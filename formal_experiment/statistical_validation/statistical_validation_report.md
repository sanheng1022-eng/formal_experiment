# Statistical Validation

All calculations are read-only derivatives of the frozen Exp 1 master and Exp 4 crossing master. Bootstrap resampling is by `query_id` (10,000 repetitions; seed 20261001). Effects are reported as new/intervention minus baseline/reference.

## Retrieval-level paired bootstrap

| Comparison | Metric | Effect | 95% CI | Bootstrap p |
|---|---|---:|---:|---:|
| MultiPIC vs Calibration | R@20 | 0.0550 | [0.0000, 0.1150] | 0.06999 |
| MultiPIC vs Calibration | MRR | 0.0495 | [0.0195, 0.0803] | 0.00100 |
| MultiPIC vs Calibration | Mean Gold Rank | -5.1800 | [-8.5251, -1.9850] | 0.00280 |
| MultiPIC vs Calibration + Support | R@20 | 0.0850 | [0.0300, 0.1400] | 0.00200 |
| MultiPIC vs Calibration + Support | MRR | 0.0597 | [0.0281, 0.0922] | 0.00040 |
| MultiPIC vs Calibration + Support | Mean Gold Rank | -5.4150 | [-8.6650, -2.2748] | 0.00180 |

## Exact McNemar

| Comparison | Metric | both hit | both miss | base hit → new miss | base miss → new hit | exact p |
|---|---|---:|---:|---:|---:|---:|
| MultiPIC vs Calibration | R@1 | 13 | 170 | 5 | 12 | 0.14346 |
| MultiPIC vs Calibration | R@5 | 50 | 123 | 7 | 20 | 0.01916 |
| MultiPIC vs Calibration | R@10 | 74 | 92 | 7 | 27 | 0.00082 |
| MultiPIC vs Calibration | R@20 | 103 | 62 | 12 | 23 | 0.08953 |
| MultiPIC vs Calibration + Support | R@1 | 14 | 166 | 4 | 16 | 0.01182 |
| MultiPIC vs Calibration + Support | R@5 | 49 | 124 | 8 | 19 | 0.05224 |
| MultiPIC vs Calibration + Support | R@10 | 74 | 93 | 7 | 26 | 0.00132 |
| MultiPIC vs Calibration + Support | R@20 | 107 | 60 | 8 | 25 | 0.00455 |

## Crossing diagnostics

Candidate-level estimates retain all candidates within each sampled query; query-balanced estimates first average within a nonempty query/group. Thus neither treats crossing candidates as independent observations. Effects are group 1 minus group 0, as stated in each row.

| Comparison | Level | Metric | Effect | 95% CI | Bootstrap p |
|---|---|---|---:|---:|---:|
| Exp5 New overtaker - Defeated | candidate_level | top1 | 0.0424 | [0.0295, 0.0558] | 0.00020 |
| Exp5 New overtaker - Defeated | query_balanced | top1 | 0.0549 | [0.0492, 0.0604] | 0.00020 |
| Exp5 New overtaker - Defeated | candidate_level | top2 | 0.0150 | [0.0031, 0.0273] | 0.01180 |
| Exp5 New overtaker - Defeated | query_balanced | top2 | 0.0140 | [0.0077, 0.0201] | 0.00020 |
| Exp5 New overtaker - Defeated | candidate_level | gap12 | 0.0274 | [0.0246, 0.0307] | 0.00020 |
| Exp5 New overtaker - Defeated | query_balanced | gap12 | 0.0409 | [0.0368, 0.0454] | 0.00020 |
| Exp5 New overtaker - Defeated | candidate_level | mean_top2 | 0.0287 | [0.0163, 0.0413] | 0.00020 |
| Exp5 New overtaker - Defeated | query_balanced | mean_top2 | 0.0344 | [0.0289, 0.0399] | 0.00020 |
| Exp5 New overtaker - Defeated | candidate_level | mean_all | 0.0156 | [0.0038, 0.0278] | 0.00760 |
| Exp5 New overtaker - Defeated | query_balanced | mean_all | 0.0159 | [0.0104, 0.0214] | 0.00020 |
| Exp5 New overtaker - Defeated | candidate_level | support_at_002 | -0.1307 | [-0.1432, -0.1188] | 0.00020 |
| Exp5 New overtaker - Defeated | query_balanced | support_at_002 | -0.1607 | [-0.1760, -0.1451] | 0.00020 |
| Exp5 New overtaker - Defeated | candidate_level | support_at_005 | -0.2560 | [-0.2817, -0.2328] | 0.00020 |
| Exp5 New overtaker - Defeated | query_balanced | support_at_005 | -0.3236 | [-0.3468, -0.2994] | 0.00020 |
| Exp5 New overtaker - Defeated | candidate_level | winner_pic_hubness | 0.0285 | [0.0239, 0.0330] | 0.00020 |
| Exp5 New overtaker - Defeated | query_balanced | winner_pic_hubness | 0.0316 | [0.0274, 0.0357] | 0.00020 |
| Exp7 Calibration: Repaired - Not repaired | candidate_level | winner_pic_hubness | 0.0305 | [0.0251, 0.0359] | 0.00020 |
| Exp7 Calibration: Repaired - Not repaired | query_balanced | winner_pic_hubness | 0.0362 | [0.0305, 0.0418] | 0.00020 |
| Exp7 Calibration: Repaired - Not repaired | candidate_level | top1 | -0.0144 | [-0.0305, 0.0004] | 0.05839 |
| Exp7 Calibration: Repaired - Not repaired | query_balanced | top1 | -0.0185 | [-0.0262, -0.0108] | 0.00020 |
| Exp7 Calibration: Repaired - Not repaired | candidate_level | gap12 | -0.0022 | [-0.0073, 0.0022] | 0.32397 |
| Exp7 Calibration: Repaired - Not repaired | query_balanced | gap12 | 0.0009 | [-0.0062, 0.0077] | 0.79152 |
| Exp7 Calibration: Repaired - Not repaired | candidate_level | support_at_002 | -0.0093 | [-0.0259, 0.0097] | 0.33657 |
| Exp7 Calibration: Repaired - Not repaired | query_balanced | support_at_002 | -0.0160 | [-0.0333, 0.0009] | 0.06459 |
| Exp7 Calibration: Repaired - Not repaired | candidate_level | support_at_005 | 0.0024 | [-0.0272, 0.0366] | 0.84752 |
| Exp7 Calibration: Repaired - Not repaired | query_balanced | support_at_005 | -0.0021 | [-0.0343, 0.0295] | 0.90551 |
| Exp7 Calibration + Support: Repaired - Not repaired | candidate_level | winner_pic_hubness | 0.0314 | [0.0262, 0.0366] | 0.00020 |
| Exp7 Calibration + Support: Repaired - Not repaired | query_balanced | winner_pic_hubness | 0.0365 | [0.0304, 0.0425] | 0.00020 |
| Exp7 Calibration + Support: Repaired - Not repaired | candidate_level | gap12 | 0.0042 | [-0.0006, 0.0085] | 0.08319 |
| Exp7 Calibration + Support: Repaired - Not repaired | query_balanced | gap12 | 0.0089 | [0.0020, 0.0155] | 0.01020 |
| Exp7 Calibration + Support: Repaired - Not repaired | candidate_level | support_at_002 | -0.0207 | [-0.0368, -0.0024] | 0.02740 |
| Exp7 Calibration + Support: Repaired - Not repaired | query_balanced | support_at_002 | -0.0304 | [-0.0480, -0.0136] | 0.00060 |
| Exp7 Calibration + Support: Repaired - Not repaired | candidate_level | support_at_005 | -0.0342 | [-0.0644, -0.0003] | 0.04800 |
| Exp7 Calibration + Support: Repaired - Not repaired | query_balanced | support_at_005 | -0.0417 | [-0.0733, -0.0101] | 0.00800 |

- `paired_bootstrap.csv`: formal retrieval effects and percentile CIs.
- `mcnemar.csv`: exact paired hit/miss transitions.
- `crossing_cluster_bootstrap.csv`: cluster-bootstrap crossing effects, CIs, p-values, and candidate/query denominators.

No scores, ranks, candidate pools, folds, PICs, or parameter choices were modified.
