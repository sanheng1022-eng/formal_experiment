# Formal embedding-model layout

Both formal-experiment embedding models are kept under `formal_experiment/models/`. The shared Python environment remains at the workspace root as `.venv312`.

| Model | Directory | Start script | Local endpoint | Input rule |
| --- | --- | --- | --- | --- |
| Qwen3-Embedding-0.6B | `models/qwen3-embedding-0.6b` | `scripts/start_qwen_embedding.ps1` | `:8000` | Query requests may use an English task instruction. |
| BAAI/bge-m3 | `models/bge-m3` | `scripts/start_bge_m3_embedding.ps1` | `:8001` | Send raw text only; no instruction prefix. |

The machine can run both services together because they use distinct ports. Start each in its own PowerShell window from the workspace root:

```powershell
.\formal_experiment\scripts\start_qwen_embedding.ps1
.\formal_experiment\scripts\start_bge_m3_embedding.ps1
```

Both services are OpenAI-compatible at `POST /v1/embeddings` and return 1024-dimensional L2-normalized dense vectors. Record the model ID, endpoint, full request payload, and this registry's version for every experimental run. Do not compare vectors across models directly or combine them in one index: create and query a separate index for each model.

The machine has a 4 GB GPU, so keep requests modest (Qwen: 8 texts/batch; BGE-M3: 4 texts/batch) and lower `max_length` for BGE-M3 when 8192 tokens are unnecessary.
