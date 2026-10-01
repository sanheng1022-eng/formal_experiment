# Experiment 11 — BGE-M3 main retrieval

| Method | R@1 | R@5 | R@10 | R@20 | R@40 | MRR | Mean Gold Rank | Median Gold Rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.020 | 0.130 | 0.250 | 0.390 | 0.565 | 0.095 | 49.780 | 30.0 |
| MultiPIC-Max | 0.055 | 0.250 | 0.335 | 0.455 | 0.640 | 0.158 | 42.260 | 23.5 |
| PIC Calibration | 0.095 | 0.305 | 0.435 | 0.550 | 0.720 | 0.208 | 33.600 | 14.0 |
| Calibration + Support | 0.100 | 0.320 | 0.445 | 0.560 | 0.740 | 0.210 | 32.360 | 15.0 |

Protocol: 200 OOF queries, fixed N=200 pool, train-only 4-fold inner selection.
