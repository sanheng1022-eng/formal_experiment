# Qwen embedding service for formal experiments

## Files

- Service: `formal_experiment/src/qwen_embedding_service.py`
- Startup script: `formal_experiment/scripts/start_qwen_embedding.ps1`
- Dependencies: `formal_experiment/requirements-qwen-embedding.txt`

The startup script resolves the shared workspace paths automatically:

- Python: `<workspace>/.venv312/Scripts/python.exe`
- Model: `<workspace>/formal_experiment/models/qwen3-embedding-0.6b`

Set `QWEN_EMBEDDING_MODEL_PATH` before starting only when the model is stored elsewhere.

## First-time dependency check

From the workspace root:

```powershell
.\.venv312\Scripts\python.exe -m pip install -r .\formal_experiment\requirements-qwen-embedding.txt
```

The CUDA-compatible PyTorch build is an environment prerequisite and is intentionally not installed by the startup script.

## Start and verify

```powershell
.\formal_experiment\scripts\start_qwen_embedding.ps1
```

The service binds only to `127.0.0.1:8000`. After it reports ready:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

It provides `POST /v1/embeddings`, returning normalized embeddings. Query requests may supply an `instruction`; document and PIC embeddings normally omit one to keep retrieval views comparable with the exploratory protocol. The paired BGE-M3 service uses a different input convention; see `EMBEDDING_MODELS.md`.
