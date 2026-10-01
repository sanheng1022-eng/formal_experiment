"""Build and audit the frozen Model-A embedding cache for Experiment 1.

This is deliberately embedding-only: it performs no retrieval, ranking,
parameter selection, or CV.  State text is read from the frozen raw dataset;
the script neither writes the dataset nor invokes a generation model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "paired_states_v2" / "paired_states.json"
CACHE = ROOT / "data" / "derived" / "embeddings" / "qwen3_embedding_0.6b.npz"
INDEX = CACHE.with_suffix(".index.json")
META = CACHE.with_suffix(".metadata.json")
REPORT = ROOT / "embedding_cache_audit_report.md"
ENDPOINT = "http://127.0.0.1:8000/v1/embeddings"
MODEL = "Qwen3-Embedding-0.6B"
DIMENSION = 1024
BATCH_SIZE = 8
QUERY_INSTRUCTION = (
    "Given a new user-state observation, retrieve historical-state invalidation "
    "conditions that are semantically compatible with the observation and indicate "
    "that the corresponding historical state may be outdated, invalid, or require "
    "re-evaluation."
)


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_existing() -> dict[str, tuple[str, np.ndarray]]:
    """Reuse only vectors whose ID, kind, source-text hash, and vector pass audit."""
    if not (CACHE.exists() and INDEX.exists() and META.exists()):
        return {}
    try:
        metadata = json.loads(META.read_text(encoding="utf-8"))
        if (
            metadata.get("model") != MODEL
            or metadata.get("embedding_dimension") != DIMENSION
            or metadata.get("query_instruction") != QUERY_INSTRUCTION
            or metadata.get("normalization") != "L2"
        ):
            return {}
        entries = json.loads(INDEX.read_text(encoding="utf-8"))["entries"]
        with np.load(CACHE, allow_pickle=False) as archive:
            vectors = archive["embeddings"]
        reusable: dict[str, tuple[str, np.ndarray]] = {}
        for entry, vector in zip(entries, vectors, strict=True):
            key = f"{entry['kind']}:{entry['id']}"
            if (
                entry.get("text_sha256")
                and vector.shape == (DIMENSION,)
                and np.isfinite(vector).all()
                # Service normalizes in float16 on CUDA then persists float32 values;
                # 1e-3 accepts that documented rounding envelope without re-normalizing.
                and np.isclose(np.linalg.norm(vector), 1.0, atol=1e-3, rtol=1e-3)
            ):
                reusable[key] = (entry["text_sha256"], vector.astype(np.float32, copy=False))
        return reusable
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return {}


def embed(texts: list[str], instruction: str | None) -> np.ndarray:
    vectors: list[list[float]] = []
    for offset in range(0, len(texts), BATCH_SIZE):
        payload: dict[str, Any] = {"input": texts[offset : offset + BATCH_SIZE], "model": MODEL}
        if instruction is not None:
            payload["instruction"] = instruction
        response = requests.post(ENDPOINT, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()["data"]
        if [item["index"] for item in data] != list(range(len(data))):
            raise RuntimeError("Embedding service returned non-contiguous batch indices.")
        vectors.extend(item["embedding"] for item in data)
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (len(texts), DIMENSION):
        raise RuntimeError(f"Unexpected embedding shape {matrix.shape}; expected {(len(texts), DIMENSION)}")
    return matrix


def validate(entries: list[dict[str, str]], matrix: np.ndarray) -> list[str]:
    errors: list[str] = []
    counts = {kind: sum(entry["kind"] == kind for entry in entries) for kind in ("state_old", "state_new", "pic")}
    if counts != {"state_old": 200, "state_new": 200, "pic": 800}:
        errors.append(f"Unexpected object counts: {counts}")
    keys = [(entry["kind"], entry["id"]) for entry in entries]
    if len(keys) != len(set(keys)):
        errors.append("Duplicate kind/ID pairs in cache index.")
    if matrix.shape != (1200, DIMENSION):
        errors.append(f"Unexpected cache matrix shape {matrix.shape}; expected (1200, {DIMENSION}).")
    if not np.isfinite(matrix).all():
        errors.append("Cache contains NaN or Inf.")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3, rtol=1e-3):
        errors.append(f"L2 normalization failed: norm range {norms.min():.9f}–{norms.max():.9f}.")
    old_ids = {entry["state_id"] for entry in entries if entry["kind"] == "state_old"}
    new_ids = {entry["query_id"] for entry in entries if entry["kind"] == "state_new"}
    pic_ids = {entry["pic_id"] for entry in entries if entry["kind"] == "pic"}
    if len(old_ids) != 200 or old_ids != new_ids:
        errors.append("state_old/state_new IDs are incomplete or do not map one-to-one.")
    if len(pic_ids) != 800:
        errors.append("PIC IDs are incomplete or duplicated.")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    rows = json.loads(RAW.read_text(encoding="utf-8"))
    if len(rows) != 200 or len({row.get("uid") for row in rows}) != 200:
        raise RuntimeError("Frozen dataset must have exactly 200 unique uids.")
    rows = sorted(rows, key=lambda row: row["uid"])
    items: list[dict[str, str]] = []
    for row in rows:
        uid = row["uid"]
        old, new = row.get("state_old", ""), row.get("state_new", "")
        pics = row.get("possible_invalidation_conditions") or []
        if not old.strip() or not new.strip() or len(pics) < 4 or any(not pic.strip() for pic in pics[:4]):
            raise RuntimeError(f"Invalid frozen text for {uid}.")
        # User-approved: cache only the first four in source-list order; source data remains unchanged.
        items.append({"kind": "state_old", "id": uid, "state_id": uid, "text": old, "text_sha256": digest_text(old)})
        items.append({"kind": "state_new", "id": uid, "query_id": uid, "text": new, "text_sha256": digest_text(new)})
        for number, pic in enumerate(pics[:4], start=1):
            items.append({"kind": "pic", "id": f"{uid}:PIC_{number}", "state_id": uid, "pic_id": f"{uid}:PIC_{number}", "text": pic, "text_sha256": digest_text(pic)})

    reusable = load_existing()
    vectors: list[np.ndarray | None] = [None] * len(items)
    groups: dict[tuple[str, str | None], list[int]] = {}
    for index, item in enumerate(items):
        key = f"{item['kind']}:{item['id']}"
        cached = reusable.get(key)
        if cached is not None and cached[0] == item["text_sha256"]:
            vectors[index] = cached[1]
        else:
            instruction = QUERY_INSTRUCTION if item["kind"] == "state_new" else None
            groups.setdefault((item["kind"], instruction), []).append(index)
    if args.verify_only and any(vector is None for vector in vectors):
        raise RuntimeError("Cache is incomplete or incompatible; --verify-only will not generate vectors.")
    generated = 0
    for (_, instruction), indices in groups.items():
        matrix = embed([items[index]["text"] for index in indices], instruction)
        for index, vector in zip(indices, matrix, strict=True):
            vectors[index] = vector
            generated += 1
    matrix = np.vstack([vector for vector in vectors if vector is not None]).astype(np.float32, copy=False)
    index_entries = [{key: value for key, value in item.items() if key != "text"} for item in items]
    errors = validate(index_entries, matrix)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not errors:
        with tempfile.NamedTemporaryFile(dir=CACHE.parent, suffix=".npz", delete=False) as handle:
            temporary_cache = Path(handle.name)
        try:
            np.savez_compressed(temporary_cache, embeddings=matrix)
            os.replace(temporary_cache, CACHE)
        finally:
            temporary_cache.unlink(missing_ok=True)
        INDEX.write_text(json.dumps({"schema_version": "experiment1-embedding-index-v1", "entries": index_entries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metadata = {
            "schema_version": "experiment1-embedding-cache-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": digest_file(RAW),
            "model": MODEL,
            "checkpoint": "models/qwen3-embedding-0.6b",
            "embedding_dimension": DIMENSION,
            "tokenizer": "AutoTokenizer(local_files_only=true, padding_side=left)",
            "pooling": "last_non_padding_token",
            "normalization": "L2",
            "device": "service-reported CUDA",
            "dtype": "float16 model inference; float32 persisted vectors",
            "batch_size": BATCH_SIZE,
            "cosine": "dot product of L2-normalized vectors",
            "state_new_instruction_only": True,
            "query_instruction": QUERY_INSTRUCTION,
            "pic_selection": "PIC_1 through PIC_4 in frozen source-list order; fifth PIC excluded only from cache by user approval",
            "generated_vectors_this_run": generated,
            "reused_vectors_this_run": len(items) - generated,
        }
        META.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = {kind: sum(entry["kind"] == kind for entry in index_entries) for kind in ("state_old", "state_new", "pic")}
    norm_min, norm_max = float(np.linalg.norm(matrix, axis=1).min()), float(np.linalg.norm(matrix, axis=1).max())
    outcome = "EMBEDDING CACHE READY" if not errors else "NOT READY"
    REPORT.write_text(
        f"# Experiment 1 embedding cache audit\n\n**Status: {outcome}**\n\n"
        f"- Cache: `{CACHE.relative_to(ROOT)}`\n- Index: `{INDEX.relative_to(ROOT)}`\n- Metadata: `{META.relative_to(ROOT)}`\n"
        f"- Counts: state_old={counts['state_old']}; state_new={counts['state_new']}; PIC={counts['pic']}\n"
        f"- Matrix shape: {tuple(matrix.shape)}; persisted dtype: float32\n"
        f"- Generated this run: {generated}; reused after audit: {len(items) - generated}\n"
        f"- NaN/Inf check: {'PASS' if np.isfinite(matrix).all() else 'FAIL'}\n"
        f"- L2 normalization check: {'PASS' if not any('normalization' in error for error in errors) else 'FAIL'}\n"
        f"- L2 norm range: {norm_min:.9f}–{norm_max:.9f}; tolerance: 1e-3 (CUDA float16 normalization, persisted as float32).\n"
        f"- PIC selection: first four source-list PICs; source texts unchanged.\n"
        f"- state_new instruction: `{QUERY_INSTRUCTION}`\n"
        + ("\n## Errors\n\n" + "\n".join(f"- {error}" for error in errors) + "\n" if errors else ""),
        encoding="utf-8",
    )
    print(outcome)
    print(f"generated={generated} reused={len(items) - generated} shape={matrix.shape}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
