"""OpenAI-compatible dense-embedding service for local BAAI/bge-m3."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer


FORMAL_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = os.getenv("BGE_M3_MODEL_PATH", str(FORMAL_ROOT / "models" / "bge-m3"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = None
MODEL = None


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    encoding_format: Literal["float"] = "float"
    dimensions: int | None = Field(default=None, ge=32, le=1024)
    max_length: int = Field(default=8192, ge=1, le=8192)
    # BGE-M3 encodes raw text; it does not use the Qwen `Instruct/Query` prefix.
    instruction: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL, TOKENIZER
    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(f"Local BGE-M3 directory not found: {MODEL_PATH}")
    kwargs = {"local_files_only": True}
    if DEVICE == "cuda":
        kwargs["torch_dtype"] = torch.float16
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    MODEL = AutoModel.from_pretrained(MODEL_PATH, **kwargs).to(DEVICE).eval()
    yield
    MODEL = None
    TOKENIZER = None
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


app = FastAPI(title="BAAI/bge-m3", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL is not None else "loading",
        "model": "BAAI/bge-m3",
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "dimensions": 1024,
        "max_length": 8192,
        "instruction_mode": "raw-input-only",
    }


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "BAAI/bge-m3", "object": "model"}]}


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    if request.instruction:
        raise HTTPException(
            status_code=400,
            detail="BAAI/bge-m3 must receive raw text; do not send an instruction prefix.",
        )
    if request.dimensions not in (None, 1024):
        raise HTTPException(status_code=400, detail="BAAI/bge-m3 has fixed 1024 dimensions")
    texts = [request.input] if isinstance(request.input, str) else request.input
    if not texts:
        raise HTTPException(status_code=400, detail="input must not be empty")
    if any(not text.strip() for text in texts):
        raise HTTPException(status_code=400, detail="input entries must not be blank")

    vectors = []
    # BGE-M3 uses the normalized CLS representation for dense retrieval.
    for start in range(0, len(texts), 4):
        batch = TOKENIZER(
            texts[start : start + 4],
            padding=True,
            truncation=True,
            max_length=request.max_length,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.inference_mode():
            cls = MODEL(**batch).last_hidden_state[:, 0]
            vectors.extend(F.normalize(cls, p=2, dim=1).float().cpu().tolist())

    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": vector, "index": index}
            for index, vector in enumerate(vectors)
        ],
        "model": "BAAI/bge-m3",
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
