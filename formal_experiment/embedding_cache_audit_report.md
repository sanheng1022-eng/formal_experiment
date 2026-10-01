# Experiment 1 embedding cache audit

**Status: EMBEDDING CACHE READY**

- Cache: `data\derived\embeddings\qwen3_embedding_0.6b.npz`
- Index: `data\derived\embeddings\qwen3_embedding_0.6b.index.json`
- Metadata: `data\derived\embeddings\qwen3_embedding_0.6b.metadata.json`
- Counts: state_old=200; state_new=200; PIC=800
- Matrix shape: (1200, 1024); persisted dtype: float32
- Generated this run: 0; reused after audit: 1200
- NaN/Inf check: PASS
- L2 normalization check: PASS
- L2 norm range: 0.999649048–1.000344276; tolerance: 1e-3 (CUDA float16 normalization, persisted as float32).
- PIC selection: first four source-list PICs; source texts unchanged.
- state_new instruction: `Given a new user-state observation, retrieve historical-state invalidation conditions that are semantically compatible with the observation and indicate that the corresponding historical state may be outdated, invalid, or require re-evaluation.`
