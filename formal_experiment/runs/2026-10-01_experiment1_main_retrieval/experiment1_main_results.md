# Experiment 1 — Main Retrieval Result

| Method | R@1 | R@5 | R@10 | R@20 | R@40 | MRR | Mean Gold Rank | Median Gold Rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.055 | 0.225 | 0.355 | 0.510 | 0.650 | 0.153 | 39.725 | 19.5 |
| MultiPIC-Max | 0.090 | 0.285 | 0.405 | 0.575 | 0.730 | 0.198 | 31.215 | 17.0 |
| Raw + MultiPIC-Max | 0.090 | 0.280 | 0.405 | 0.575 | 0.730 | 0.198 | 31.175 | 17.0 |
| Mean | 0.115 | 0.290 | 0.430 | 0.565 | 0.735 | 0.212 | 30.290 | 15.5 |
| Top-2 Mean | 0.105 | 0.305 | 0.405 | 0.565 | 0.740 | 0.211 | 30.145 | 15.0 |
| LogMeanExp (tau=0.1) | 0.105 | 0.300 | 0.430 | 0.565 | 0.730 | 0.208 | 29.730 | 16.0 |
| Mean-centering | 0.140 | 0.320 | 0.490 | 0.660 | 0.840 | 0.246 | 23.710 | 11.0 |
| Z-score | 0.130 | 0.350 | 0.470 | 0.650 | 0.795 | 0.251 | 24.915 | 12.0 |
| CSLS-style | 0.125 | 0.320 | 0.470 | 0.665 | 0.800 | 0.236 | 25.495 | 12.0 |
| PIC Calibration | 0.125 | 0.350 | 0.505 | 0.630 | 0.805 | 0.248 | 26.035 | 10.0 |
| Calibration + Support | 0.150 | 0.340 | 0.500 | 0.660 | 0.785 | 0.258 | 25.800 | 10.5 |

## Formal integrity checks

- 200 OOF queries; 200 candidates/query; exactly one gold/query; 40,000 candidate-level records.
- Every rank is in [1, 200]; all score fields are finite.
- Background statistics use only the corresponding inner-train / final outer-train state_new records.
- Master candidate result is frozen by the accompanying SHA-256 manifest; future analyses must derive from it without rescoring.
