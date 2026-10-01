# BGE-M3 embedding service

The formal workspace contains a local, hash-verified copy of `BAAI/bge-m3` in `formal_experiment/models/bge-m3`. It is independent of the Qwen embedding service.

Start the service from the workspace root:

```powershell
.\formal_experiment\scripts\start_bge_m3_embedding.ps1
```

It listens on `http://127.0.0.1:8001`; use `-Port 8002` to select another port. Set `BGE_M3_MODEL_PATH` only if the model is deliberately moved elsewhere.

`POST /v1/embeddings` returns 1024-dimensional, L2-normalized dense embeddings. BGE-M3 must receive raw text: unlike Qwen3-Embedding, do not pass a Qwen-style `instruction` or an `Instruct:/Query:` prefix. The API rejects an `instruction` value so that experiment calls remain reproducible.

Example:

```powershell
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8001/v1/embeddings' -ContentType 'application/json' -Body '{"model":"BAAI/bge-m3","input":"The user prefers working in the morning."}'
```

The model supports input up to 8192 tokens. Use a smaller `max_length` request value when long context is unnecessary, since this substantially reduces GPU memory use.
