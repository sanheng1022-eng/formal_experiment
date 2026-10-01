"""OpenAI-compatible embedding service for a local Qwen3-Embedding model."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# The source file lives in <workspace>/formal_experiment/src. Keep formal
# experiment assets together while allowing a caller to override the model
# directory on another machine.
FORMAL_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = os.getenv(
    "QWEN_EMBEDDING_MODEL_PATH",
    str(FORMAL_ROOT / "models" / "qwen3-embedding-0.6b"),
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = None
MODEL = None


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    encoding_format: Literal["float"] = "float"
    dimensions: int | None = Field(default=None, ge=32, le=1024)
    # For retrieval queries, set this to a task-specific English instruction.
    instruction: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL, TOKENIZER
    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(f"Local model directory not found: {MODEL_PATH}")
    kwargs = {"local_files_only": True}
    if DEVICE == "cuda":
        kwargs["torch_dtype"] = torch.float16
    TOKENIZER = AutoTokenizer.from_pretrained(
        MODEL_PATH, padding_side="left", local_files_only=True
    )
    MODEL = AutoModel.from_pretrained(MODEL_PATH, **kwargs).to(DEVICE).eval()
    yield
    MODEL = None
    TOKENIZER = None
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


app = FastAPI(title="Qwen3-Embedding-0.6B", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL is not None else "loading",
        "model": MODEL_PATH,
        "device": DEVICE,
    }


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "Qwen3-Embedding-0.6B", "object": "model"}]}


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    texts = [request.input] if isinstance(request.input, str) else request.input
    if not texts:
        raise HTTPException(status_code=400, detail="input must not be empty")
    if any(not text.strip() for text in texts):
        raise HTTPException(status_code=400, detail="input entries must not be blank")

    if request.instruction:
        texts = [f"Instruct: {request.instruction}\nQuery:{text}" for text in texts]

    vectors = []
    # A small batch size keeps this service usable on the local 4 GB GPU.
    for start in range(0, len(texts), 8):
        batch = TOKENIZER(
            texts[start : start + 8],
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.inference_mode():
            hidden_states = MODEL(**batch).last_hidden_state
            pooled = hidden_states[:, -1]  # tokenizer uses left padding
            if request.dimensions:
                pooled = pooled[:, : request.dimensions]
            vectors.extend(F.normalize(pooled, p=2, dim=1).float().cpu().tolist())
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": vector, "index": index}
            for index, vector in enumerate(vectors)
        ],
        "model": "Qwen3-Embedding-0.6B",
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
