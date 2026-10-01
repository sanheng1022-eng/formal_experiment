# Experiment 11 — Second Embedding Model Replication (BGE-M3)

## Frozen protocol and embedding audit

- Model/checkpoint: `BAAI/bge-m3` from `formal_experiment/models/bge-m3`; local `AutoTokenizer`, CLS pooling, raw text input (no query prefix), L2 normalization, and cosine implemented as the dot product of normalized vectors.
- New immutable cache: 200 `state_old`, 200 `state_new`, and 800 frozen PIC vectors; 1,200 x 1,024 total. All values are finite; L2 norms range from 0.9996 to 1.0004 (float32 cache produced by the fp16 CUDA service).
- The benchmark, canonical IDs, four-PIC selection, N=200 candidate pool, outer-fold assignment, and score-descending/ID-ascending tie-break are unchanged from Experiment 1.
- BGE-M3 parameter choices were independent: four inner folds of 120/40 within each outer-train, seed 20261001, TopK {10,20,40}, beta {0.25,0.5,0.75,1.0}, selecting max R@20, then max MRR, then lower beta and TopK. The selected `(TopK,beta)` by outer fold are `(40,1.0)`, `(40,0.75)`, `(40,1.0)`, `(20,1.0)`, `(20,0.75)`.

## Retrieval result (200 complete OOF queries)

| Method | R@1 | R@5 | R@10 | R@20 | R@40 | MRR | Mean gold rank | Median gold rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | .020 | .130 | .250 | .390 | .565 | .095 | 49.780 | 30.0 |
| MultiPIC-Max | .055 | .250 | .335 | .455 | .640 | .158 | 42.260 | 23.5 |
| PIC Calibration | .095 | .305 | .435 | .550 | .720 | .208 | 33.600 | 14.0 |
| Calibration + Support | .100 | .320 | .445 | .560 | .740 | .210 | 32.360 | 15.0 |

## Crossing and mechanism replication

Raw → MultiPIC produced 5,812 defeated distractors and 4,308 new overtakers. For every one of the 200 queries, `raw_gold_rank - multipic_gold_rank = n_defeated - n_new_overtaker`; the fixed crossing definition therefore passed its identity check.

The query-cluster bootstrap (10,000 repetitions; seed 20261001) shows the same directional pattern as the primary model. New overtakers versus defeated distractors had higher winner hubness (query-balanced .663 vs .637, 95% bootstrap CI for new .659–.667), larger gap12 (.051 vs .025), and lower support@0.02 (1.221 vs 1.678) and support@0.05 (1.781 vs 2.652). These are associations, not causal claims.

## Intervention evidence

Of the 4,308 original BGE-M3 new overtakers, Calibration repaired 2,304 (53.5%) and Calibration + Support repaired 2,397 (55.6%). Calibration-repaired cases had higher mean hubness than unrepaired cases (.665 vs .644), consistent with preferential high-background repair. Support-repaired cases had lower support@0.02 (1.288 vs 1.333); its gap12 was only marginally larger (.0450 vs .0435), so the isolation/support direction is replicated more clearly than a large-gap-specific effect.

## Descriptive cross-model conclusion

| Question | Primary model direction | BGE-M3 result | Replication judgement |
|---|---|---|---|
| Raw → MultiPIC prospective coverage | R@20 .510 → .575 | .390 → .455 | direction replicated |
| Beneficial and harmful crossings coexist | 3,913 defeated; 2,211 new | 5,812 defeated; 4,308 new | direction replicated |
| New overtakers: elevated hubness / isolated evidence | higher hubness and gap; lower support | same direction for all four indicators | direction replicated |
| Calibration retrieval benefit / harm suppression | R@20 .575 → .630 | .455 → .550; 53.5% new-overtaker repair | direction replicated |
| Support additional corroboration effect | R@20 .630 → .660 | .550 → .560; 55.6% repair | direction replicated |

Absolute cosine scales and ranks differ between models, as expected. The independent replication nevertheless preserves the main representation benefit, the simultaneous distractor-amplification failure pattern, the hubness/isolation association, and the calibration/support improvements. No primary-model score, parameter choice, fold, or frozen master artifact was modified.

## Integrity and frozen artifacts

- `exp11_bgem3_master_candidate_results.csv.gz`: 40,000 records; 200 queries; 200 candidates/query; exactly one gold/query; all score values finite; all formal ranks in [1,200].
- `exp11_bgem3_crossing_master.csv.gz`: frozen Raw → MultiPIC crossing dataset used by the mechanism and repair analyses.
- The accompanying frozen manifest records source hashes, cache audit, and independent selection protocol.
