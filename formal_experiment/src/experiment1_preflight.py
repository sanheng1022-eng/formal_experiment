"""Deterministic, non-evaluative preflight for Formal Experiment 1.

This command intentionally never calculates retrieval scores.  It creates the
frozen ID/fold/configuration records and refuses a dry-run when an immutable
input violates the SOP.  That prevents a partial result from being mistaken
for a formal Experiment 1 result.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "paired_states_v2" / "paired_states.json"
CONFIG = ROOT / "configs" / "experiment_config.yaml"
FOLDS = ROOT / "configs" / "outer_folds.json"
REPORT = ROOT / "experiment1_preflight_report.md"
PIC_SELECTION = ROOT / "data" / "derived" / "experiment1_pic_selection.json"
SEED = 20261001


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable (formal_experiment is not a Git worktree)"


def make_folds(ids: list[str]) -> dict:
    shuffled = sorted(ids)
    random.Random(SEED).shuffle(shuffled)
    folds = []
    for number in range(5):
        test = sorted(shuffled[number * 40 : (number + 1) * 40])
        test_set = set(test)
        folds.append(
            {
                "outer_fold": number,
                "train_query_ids": sorted(identifier for identifier in ids if identifier not in test_set),
                "test_query_ids": test,
            }
        )
    return {
        "schema_version": "experiment1-outer-folds-v1",
        "random_seed": SEED,
        "id_policy": "state_id and query_id are both the immutable paired_states_v2 uid",
        "candidate_order": "canonical state_id ascending (UUID lexical order)",
        "folds": folds,
    }


def yaml_config(dataset_hash: str) -> str:
    return f'''# Frozen Experiment 1 preflight configuration; generated {date.today().isoformat()}
experiment: Experiment 1 — Main Retrieval Result
execution_mode: preflight_only
dataset:
  path: data/raw/paired_states_v2/paired_states.json
  sha256: {dataset_hash}
  expected_pairs: 200
  id_policy: "state_id=query_id=paired_states_v2.uid"
  query_field: state_new
  gold_field: state_old
pic:
  expected_per_state: 4
  source_field: possible_invalidation_conditions
  selection: first_four_in_frozen_source_list_order__user_approved_2026-10-01
  frozen: true
  regeneration_allowed: false
embedding:
  model_id: Qwen3-Embedding-0.6B
  checkpoint: models/qwen3-embedding-0.6b
  dimensions: 1024
  tokenizer: AutoTokenizer(local_files_only=true, padding_side=left)
  pooling: last_non_padding_token
  normalization: L2
  device: cuda_if_available_else_cpu
  dtype: float16_on_cuda_else_model_default
  batch_size: 8
  cosine: dot_product_of_L2_normalized_vectors
  cache_path: data/derived/embeddings/qwen3_embedding_0.6b.npz
  state_new_instruction: "Given a new user-state observation, retrieve historical-state invalidation conditions that are semantically compatible with the observation and indicate that the corresponding historical state may be outdated, invalid, or require re-evaluation."
  state_old_and_pic_instruction: null
candidate_pool:
  size: 200
  pool_seed: null
  ordering: canonical_state_id_ascending
  tie_breaking: score_descending_then_canonical_state_id_ascending
outer_cv:
  folds: 5
  random_seed: {SEED}
  assignment_file: configs/outer_folds.json
inner_selection:
  train_only: true
  inner_folds: 4
  random_seed: {SEED}
  inner_split: 120_train__40_validation_per_inner_fold
  background_topk: [10, 20, 40]
  beta: [0.25, 0.5, 0.75, 1.0]
  primary_metric: R@20
  secondary_metric: MRR
  tie_breaking: highest_R@20__then_highest_MRR__then_lowest_beta__then_lowest_TopK
  alpha: 0.5
background:
  exclude_all_known_positive_updates: true
  statistics_source: inner_train_during_selection__outer_train_after_selection
  epsilon: 1e-8
methods:
  - B0 Raw State
  - B1 MultiPIC-Max
  - B2 Raw + MultiPIC-Max
  - B3 Mean
  - B3 Top-2 Mean
  - B3 Softmax / LogSumExp
  - B4 Mean-centering
  - B4 Z-score
  - B4 CSLS-style
  - M1 PIC-specific Background Calibration
  - M2 Calibration + Support
pooling:
  logmeanexp_tau: 0.1
  logmeanexp: tau_times_log_mean_exp_of_four_pic_scores
outputs:
  candidate_level_format: csv_gzip
  required_columns:
    - query_id
    - gold_state_id
    - candidate_state_id
    - is_gold
    - raw_score
    - PIC_1_score
    - PIC_2_score
    - PIC_3_score
    - PIC_4_score
    - top1
    - top2
    - gap12
    - winner_pic_id
    - winner_pic_hubness
    - calibrated_scores
    - support_score
    - raw_rank
    - multipic_rank
    - calibrated_rank
    - support_rank
    - candidate_pool_size
    - candidate_pool_seed
    - outer_fold
    - embedding_model
    - chosen_beta
    - chosen_topk
    - alpha
  sidecars: [fold_config, model_config, experiment_config, random_seeds, code_version]
'''


def main() -> int:
    rows = json.loads(RAW.read_text(encoding="utf-8"))
    ids = [row.get("uid") for row in rows]
    pic_counts = Counter(len(row.get("possible_invalidation_conditions") or []) for row in rows)
    empty_old = sum(not str(row.get("state_old") or "").strip() for row in rows)
    empty_new = sum(not str(row.get("state_new") or "").strip() for row in rows)
    duplicate_pic_rows = sum(
        len(pics := (row.get("possible_invalidation_conditions") or [])) != len(set(pics))
        for row in rows
    )
    nonfour = [row["uid"] for row in rows if len(row.get("possible_invalidation_conditions") or []) != 4]
    dataset_hash = sha256(RAW)
    selection = {
        "schema_version": "experiment1-pic-selection-v1",
        "dataset_sha256": dataset_hash,
        "selection_policy": "first four PICs in immutable source-list order; no text modified",
        "states": [
            {
                "state_id": row["uid"],
                "selected_pic_ids": [f"{row['uid']}:PIC_{number}" for number in range(1, 5)],
                "excluded_pic_ids": [
                    f"{row['uid']}:PIC_{number}"
                    for number in range(5, len(row.get("possible_invalidation_conditions") or []) + 1)
                ],
            }
            for row in sorted(rows, key=lambda item: item["uid"])
        ],
    }
    PIC_SELECTION.parent.mkdir(parents=True, exist_ok=True)
    PIC_SELECTION.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    folds = make_folds(ids)
    FOLDS.write_text(json.dumps(folds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    CONFIG.write_text(yaml_config(dataset_hash), encoding="utf-8")

    fold_tests = [set(fold["test_query_ids"]) for fold in folds["folds"]]
    fold_ok = (
        all(len(fold["train_query_ids"]) == 160 and len(fold["test_query_ids"]) == 40 for fold in folds["folds"])
        and len(set().union(*fold_tests)) == 200
        and sum(len(a & b) for i, a in enumerate(fold_tests) for b in fold_tests[i + 1 :]) == 0
    )
    cache = ROOT / "data" / "derived" / "embeddings" / "qwen3_embedding_0.6b.npz"
    index_path = cache.with_suffix(".index.json")
    metadata_path = cache.with_suffix(".metadata.json")
    cache_result = {"status": "absent"}
    if cache.exists() and index_path.exists() and metadata_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            vectors = np.load(cache, allow_pickle=False)["embeddings"]
            kind_counts = Counter(entry["kind"] for entry in index["entries"])
            unique_entries = {(entry["kind"], entry["id"]) for entry in index["entries"]}
            expected_hashes = {}
            for row in rows:
                identifier = row["uid"]
                expected_hashes[("state_old", identifier)] = hashlib.sha256(
                    row["state_old"].encode("utf-8")
                ).hexdigest()
                expected_hashes[("state_new", identifier)] = hashlib.sha256(
                    row["state_new"].encode("utf-8")
                ).hexdigest()
                for number, text in enumerate(row["possible_invalidation_conditions"][:4], start=1):
                    expected_hashes[("pic", f"{identifier}:PIC_{number}")] = hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest()
            indexed_hashes = {(entry["kind"], entry["id"]): entry["text_sha256"] for entry in index["entries"]}
            cache_result = {
                "status": "pass",
                "shape": list(vectors.shape),
                "finite": bool(np.isfinite(vectors).all()),
                "kind_counts": dict(kind_counts),
                "metadata_dataset_matches": metadata.get("dataset_sha256") == dataset_hash,
                "index_matches_vectors": len(unique_entries) == len(index["entries"]) == vectors.shape[0],
                "text_hashes_match": indexed_hashes == expected_hashes,
                "dimension_matches": vectors.ndim == 2 and vectors.shape[1] == 1024,
            }
        except (KeyError, OSError, ValueError) as error:
            cache_result = {"status": "invalid", "error": str(error)}
    blockers = []
    if len(rows) != 200 or len(set(ids)) != 200 or any(not identifier for identifier in ids):
        blockers.append("Dataset does not provide exactly 200 unique non-empty canonical IDs.")
    if empty_old or empty_new:
        blockers.append("Dataset contains blank state_old or state_new text.")
    if any(len(row.get("possible_invalidation_conditions") or []) < 4 for row in rows):
        blockers.append("At least one state has fewer than four source PICs.")
    if cache_result.get("status") != "pass" or not all(
        cache_result.get(key) for key in ("finite", "metadata_dataset_matches", "index_matches_vectors", "text_hashes_match", "dimension_matches")
    ) or cache_result.get("kind_counts") != {"state_old": 200, "state_new": 200, "pic": 800}:
        blockers.append("No auditable Qwen embedding cache exists (expected 200 state_old + 200 state_new + 800 PIC = 1,200 vectors).")
    blockers.append("No Experiment 1 retrieval/inner-CV implementation exists in formal_experiment/src to audit or dry-run.")
    report = f'''# Experiment 1 preflight report

**Status: NOT READY**

This is a preflight-only record. No complete Experiment 1 evaluation, parameter selection, or outer-test retrieval was run.

## 1. Data integrity

- Source: `data/raw/paired_states_v2/paired_states.json`
- SHA-256: `{dataset_hash}`
- Rows / VALID rows: {len(rows)} / {sum(row.get("decision") == "VALID" for row in rows)}
- Non-empty `state_old` / `state_new`: {len(rows) - empty_old} / {len(rows) - empty_new}
- Unique immutable `uid`: {len(set(ids))} / {len(rows)}.
- The raw file has no explicit `state_id` or `query_id`; the frozen configuration defines both as `uid`. This mapping is one-to-one and stable without modifying raw text.
- `dataset_index` is not a usable ID ({len(set(row.get("dataset_index") for row in rows))} unique values), so it is deliberately excluded.
- `state_new` is configured exclusively as query; its same-UID `state_old` is the gold historical state. No duplicate state text was found.

## 2. PIC audit

- Source PIC count distribution: {dict(sorted(pic_counts.items()))}; duplicate PICs within a state: {duplicate_pic_rows}.
- **PASS under approved selection policy:** `data/derived/experiment1_pic_selection.json` freezes PIC_1–PIC_4 in original source-list order for every state. For the {len(nonfour)} five-PIC rows, PIC_5 is explicitly excluded; no PIC text was edited, regenerated, reordered, or selected using retrieval results.
- Uniform state-old-only generation provenance is unavailable in the historical record. Per user instruction, this is an accepted provenance limitation, not an execution blocker.

## 3. Embedding environment and cache

- Primary registry model: `Qwen3-Embedding-0.6B`, local checkpoint `models/qwen3-embedding-0.6b`, hidden size / expected embedding dimension 1024.
- Service implementation: `AutoTokenizer(..., padding_side=left)`, last-token pooling, L2 normalization, batch size 8, CUDA float16 when available; cosine is the dot product of L2-normalized vectors.
- Host check through `.venv312`: PyTorch 2.7.1+cu118; CUDA available; NVIDIA GeForce GTX 1650.
- Cache validation result: `{json.dumps(cache_result, ensure_ascii=False)}`.
- The index and metadata declare exactly the approved first-four PIC selection. No vectors were generated by this preflight.

## 4. Fixed 5-fold outer CV

- Generated `{FOLDS.relative_to(ROOT)}` with seed `{SEED}` from lexical canonical IDs.
- Fold validation: {"PASS" if fold_ok else "FAIL"}; five folds of 160 train / 40 test; test union = 200; pairwise test overlap = 0.
- This file must be reused unchanged for every subsequent formal Experiment 1 run.

## 5–6. Inner selection and leakage audit

- Frozen parameter grid: Background TopK {{10,20,40}}; beta {{0.25,0.5,0.75,1.0}}; primary R@20; secondary MRR; alpha fixed to 0.5.
- Required data-flow is recorded in configuration: inner-train alone creates `h_cj`, `mu_cj`, `sigma_cj`, and `r_cj`; inner-validation only evaluates parameters; after selection, complete outer-train recomputes them for outer-test.
- Required background is `B_c = train state_new minus all known positive updates of c`; test-fold queries must never enter background statistics.
- No Experiment 1 scoring or inner-CV code is present under `formal_experiment/src`. Per user instruction this is accepted as pending work, but it means the future implementation has not yet been leakage-audited.

## 7. Candidate pool

- Frozen N=200 policy: every query uses the complete corpus, its same-UID gold plus 199 non-golds.
- Canonical candidate order: lexical `uid`; tie-break: score descending, then canonical ID ascending.
- This is configuration-only until a runner asserts it for every scored query.

## 8. Methods

- Required methods and formulas are listed in `configs/experiment_config.yaml`, including MultiPIC max, B3 alternatives, B4 normalization, M1 calibration, and M2 support (`top1 + 0.5 * top2`).
- **Implementation status: pending.** No additional scoring trick was added.

## 9. Output schema

- A frozen candidate-level Parquet schema is now declared in `configs/experiment_config.yaml`: IDs, gold flag, raw/PIC scores, top statistics, hubness/calibrated/support scores and ranks, candidate-pool metadata, fold/model metadata, and chosen parameters. Required sidecars are fold/model/experiment configurations, seeds, and code version.
- The schema is not yet implemented by a runner; a dry-run must reject any missing column rather than write a summary-only output.
- Code revision: `{git_revision()}`.

## 10. Dry-run

- **Not run.** The exact-four PIC selection and embedding cache now pass preflight, but the requested 2–5-query outer-test dry-run cannot run until the retrieval/inner-CV runner exists. No parameter was selected and no outer-test result was inspected.

## 11. Blocking issues and required resolution

1. **Blocking:** implement and test the Experiment 1 runner with train-only background statistics, source-fold provenance, deterministic ranking, complete candidate-level output, and assertions. Without it, the requested dry-run cannot execute.
2. Accepted limitation: PIC-generation provenance is incomplete; the approved first-four selection is frozen in a derived manifest.

## Generated artifacts

- `configs/experiment_config.yaml`
- `configs/outer_folds.json`
- `data/derived/experiment1_pic_selection.json`
- `src/experiment1_preflight.py` (re-runnable non-evaluative audit)
'''
    REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote {CONFIG}, {FOLDS}, and {REPORT}.")
    print("NOT READY")
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
