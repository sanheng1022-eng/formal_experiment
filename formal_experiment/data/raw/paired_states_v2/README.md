# Recovered paired-state dataset

This directory contains immutable copies from the frozen exploratory snapshot.

- `paired_states.json`: 200 `VALID` canonical rows, including `source.M_old`, `source.M_new`, `state_old`, `state_new`, evidence spans, and PICs. Every row has 3–5 `possible_invalidation_conditions`.
- `paired_states.jsonl`: the complete 200-row T2 construction ledger; all rows are currently `VALID`.
- `normalization_calls.jsonl`: the full construction ledger, including original prompt, input pair, final response, model metadata, and usage.

The original normalizer source file was not present in the frozen script snapshot. `../../src/normalize_memory_pairs.py` is a documented functional recovery based on the preserved v2 prompt and schema; it is not claimed to be byte-identical to the missing source.

The recovered protocol sees both memories in one request but does **not** require literal contradiction. It generates source-faithful atomic states and allows dependency-based affectedness. A strict contradiction-only protocol is a new, separate dataset-construction protocol.

## T2 completion run (2026-09-30)

The source `STALE/T1_T2_400_FULL.json.before_noise_removal_20260919` contains 200 T2 rows. The prior state ledger covered 181 rows (180 valid and one invalid). The completion run regenerated the remaining 19 missing rows plus the existing invalid row using `deepseek-flash`, with thinking enabled and `reasoning_effort=low` under the recovered original prompt.

- All 20 responses passed schema validation: 19 `VALID`, one `INVALID` (the original environmental-observation case).
- Run ledger, cache fingerprints, selection manifest, and merge summary are in `../../runs/2026-09-30_t2_state_completion/`.
- Pre-merge copies are `paired_states.json.before_t2_completion_20260930` and `paired_states.jsonl.before_t2_completion_20260930`.

## Manual revision of UID 309111b3 (2026-09-30)

The sole environmental-only invalid pair was manually revised at the memory level to make the user's location-dependent climate state explicit. The revised pair was sent to `deepseek-flash` under the same recovered prompt with thinking enabled and `reasoning_effort=low`; it returned `VALID` with confidence 0.9. The run record is `../../runs/2026-09-30_manual_revision_309111b3/`. Pre-revision copies are `paired_states.json.before_manual_revision_309111b3` and `paired_states.jsonl.before_manual_revision_309111b3`.

## PIC completion (2026-09-30)

The 20 rows introduced or revised during T2 completion initially had empty PIC lists. They were processed by the frozen original `run_paired_state_normalization.py` script using the original `state-old-hypothetical-invalidation-updates-v1` prompt, `deepseek-flash`, enabled thinking, and `reasoning_effort=low`. All 20 calls succeeded and produced 3–5 PICs. The per-UID request/response cache is `possible_invalidation_conditions_cache/` beside this dataset.
