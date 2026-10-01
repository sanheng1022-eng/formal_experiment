# Experiment 2 — Recall–Candidate Budget Curve

This is a read-only derivation from the frozen Experiment 1 OOF master ranks. No embedding, scoring, parameter selection, cross-validation, or candidate pool was rerun or changed.

## Recall table

| Method | R@1 | R@5 | R@10 | R@20 | R@30 | R@40 | R@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.055 | 0.225 | 0.355 | 0.510 | 0.590 | 0.650 | 0.705 |
| MultiPIC | 0.090 | 0.285 | 0.405 | 0.575 | 0.670 | 0.730 | 0.765 |
| Calibration | 0.125 | 0.350 | 0.505 | 0.630 | 0.755 | 0.805 | 0.850 |
| Calibration + Support | 0.150 | 0.340 | 0.500 | 0.660 | 0.765 | 0.785 | 0.835 |

## Pairwise changes

| K | MultiPIC − Raw | Calibration − MultiPIC | Support − Calibration |
|---:|---:|---:|---:|
| 1 | +0.035 | +0.035 | +0.025 |
| 5 | +0.060 | +0.065 | -0.010 |
| 10 | +0.050 | +0.100 | -0.005 |
| 20 | +0.065 | +0.055 | +0.030 |
| 30 | +0.080 | +0.085 | +0.010 |
| 40 | +0.080 | +0.075 | -0.020 |
| 50 | +0.060 | +0.085 | -0.015 |

## Interpretation

- MultiPIC improves over Raw at every reported budget; its largest gain is +0.080 at K=30 and K=40, and it remains +0.060 at K=50.
- Calibration retains the MultiPIC benefit and improves Recall@K over MultiPIC at every reported budget; its largest gain is +0.100 at K=10.
- Calibration + Support improves over Calibration at K=1 (+0.025), K=20 (+0.030), and K=30 (+0.010), while it is lower at K=5/10/40/50 (−0.010/−0.005/−0.020/−0.015).
