"""Leakage-safe formal runner for Experiment 1 (N=200, 5-fold OOF)."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/paired_states_v2/paired_states.json"
FOLDS = ROOT / "configs/outer_folds.json"
CACHE = ROOT / "data/derived/embeddings/qwen3_embedding_0.6b.npz"
INDEX = CACHE.with_suffix(".index.json")
META = CACHE.with_suffix(".metadata.json")
RUN = ROOT / "runs/2026-10-01_experiment1_main_retrieval"
SEED = 20261001
TOPKS = (10, 20, 40)
BETAS = (0.25, 0.5, 0.75, 1.0)
ALPHA = 0.5
TAU = 0.1
EPS = 1e-8


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(scores: np.ndarray, candidate_ids: list[str]) -> np.ndarray:
    """One-based deterministic ranks: score desc, canonical state_id asc."""
    order = sorted(range(len(candidate_ids)), key=lambda i: (-float(scores[i]), candidate_ids[i]))
    result = np.empty(len(candidate_ids), dtype=np.int16)
    result[order] = np.arange(1, len(candidate_ids) + 1, dtype=np.int16)
    return result


def topmean(values: np.ndarray, k: int) -> np.ndarray:
    k = min(k, values.shape[-1])
    return np.partition(values, values.shape[-1] - k, axis=-1)[..., -k:].mean(axis=-1)


def logmeanexp(values: np.ndarray) -> np.ndarray:
    scaled = values / TAU
    maximum = scaled.max(axis=-1, keepdims=True)
    return TAU * (maximum[:, 0] + np.log(np.exp(scaled - maximum).mean(axis=-1)))


def metrics(gold_ranks: list[int]) -> dict[str, float]:
    ranks = np.asarray(gold_ranks, dtype=float)
    result = {f"R@{k}": float((ranks <= k).mean()) for k in (1, 5, 10, 20, 40)}
    result.update(MRR=float((1.0 / ranks).mean()), mean_gold_rank=float(ranks.mean()), median_gold_rank=float(np.median(ranks)))
    return result


def load() -> tuple[list[dict], list[str], np.ndarray, np.ndarray, np.ndarray, dict]:
    rows = json.loads(RAW.read_text(encoding="utf-8"))
    ids = sorted(row["uid"] for row in rows)
    row_by_id = {row["uid"]: row for row in rows}
    index = json.loads(INDEX.read_text(encoding="utf-8"))["entries"]
    vectors = np.load(CACHE, allow_pickle=False)["embeddings"].astype(np.float64)
    if vectors.shape != (1200, 1024) or len(index) != 1200 or not np.isfinite(vectors).all():
        raise ValueError("Embedding cache fails preflight shape/finiteness contract")
    mapping = {(entry["kind"], entry["id"]): vectors[n] for n, entry in enumerate(index)}
    old = np.stack([mapping[("state_old", identifier)] for identifier in ids])
    new = np.stack([mapping[("state_new", identifier)] for identifier in ids])
    pic = np.stack([mapping[("pic", f"{identifier}:PIC_{number}")] for identifier in ids for number in range(1, 5)]).reshape(200, 4, -1)
    return [row_by_id[identifier] for identifier in ids], ids, old, new, pic, json.loads(META.read_text(encoding="utf-8"))


def background_stats(pic: np.ndarray, train_indices: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Only train state_new enters these candidate/PIC statistics."""
    # [candidate, pic, train-query]
    all_sims = np.take(PIC_SIM_ALL, train_indices, axis=2)
    h = np.empty((200, 4), dtype=np.float64)
    mu = np.empty_like(h)
    sigma = np.empty_like(h)
    train_set = set(train_indices.tolist())
    for candidate in range(200):
        keep = np.array([query != candidate for query in train_indices], dtype=bool) if candidate in train_set else np.ones(len(train_indices), dtype=bool)
        values = np.compress(keep, all_sims[candidate], axis=1)
        h[candidate] = topmean(values, topk)
        mu[candidate] = values.mean(axis=1)
        sigma[candidate] = np.maximum(values.std(axis=1, ddof=1), EPS)
    return h, mu, sigma


def evaluate_validation(validation_indices: np.ndarray, h: np.ndarray, beta: float) -> tuple[list[int], list[int]]:
    cal_ranks, multi_ranks = [], []
    for query in validation_indices:
        s = np.einsum("cpd,d->cp", PIC, NEW[query], optimize=True)
        cal = (s - beta * h).max(axis=1)
        multi = s.max(axis=1)
        cal_ranks.append(int(rank(cal, IDS)[query]))
        multi_ranks.append(int(rank(multi, IDS)[query]))
    return cal_ranks, multi_ranks


def choose_parameters(outer_fold: int, outer_train: list[str]) -> tuple[int, float, list[dict]]:
    # The same seed is applied to each distinct outer-train identifier list; it cannot use outer-test results.
    shuffled = sorted(outer_train)
    random.Random(SEED).shuffle(shuffled)
    inner_tests = [shuffled[number * 40 : (number + 1) * 40] for number in range(4)]
    records = []
    for topk in TOPKS:
        by_beta = {beta: [] for beta in BETAS}
        for validation_ids in inner_tests:
            validation = np.array([ID_TO_INDEX[identifier] for identifier in validation_ids], dtype=int)
            train = np.array([ID_TO_INDEX[identifier] for identifier in shuffled if identifier not in set(validation_ids)], dtype=int)
            h, _, _ = background_stats(PIC, train, topk)
            for beta in BETAS:
                cal, _ = evaluate_validation(validation, h, beta)
                by_beta[beta].extend(cal)
        for beta, gold_ranks in by_beta.items():
            score = metrics(gold_ranks)
            records.append({"outer_fold": outer_fold, "topk": topk, "beta": beta, **score})
    # Max R@20, max MRR, then min beta and min TopK (declared tie-break).
    winner = sorted(records, key=lambda item: (-item["R@20"], -item["MRR"], item["beta"], item["topk"]))[0]
    return winner["topk"], winner["beta"], records


def output_rows(outer_fold: int, test_ids: list[str], train_ids: list[str], topk: int, beta: float) -> list[dict]:
    train = np.array([ID_TO_INDEX[identifier] for identifier in train_ids], dtype=int)
    h, mu, sigma = background_stats(PIC, train, topk)
    result = []
    for query_id in test_ids:
        query = ID_TO_INDEX[query_id]
        s = np.einsum("cpd,d->cp", PIC, NEW[query], optimize=True)
        raw = OLD @ NEW[query]
        multi = s.max(axis=1)
        raw_multi = np.maximum(raw, multi)
        mean = s.mean(axis=1)
        top2mean = topmean(s, 2)
        lme = logmeanexp(s)
        centered = (s - mu).max(axis=1)
        zscore = ((s - mu) / sigma).max(axis=1)
        rq = float(topmean(s.reshape(1, -1), topk)[0])
        csls = (2.0 * s - h - rq).max(axis=1)
        calibrated_pics = s - beta * h
        calibration = calibrated_pics.max(axis=1)
        support = np.sort(calibrated_pics, axis=1)[:, ::-1]
        support_score = support[:, 0] + ALPHA * support[:, 1]
        scores = {
            "raw": raw, "multipic": multi, "raw_multipic": raw_multi, "mean": mean,
            "top2mean": top2mean, "logmeanexp": lme, "mean_centering": centered,
            "zscore": zscore, "csls": csls, "calibration": calibration, "support": support_score,
        }
        ranks = {name: rank(value, IDS) for name, value in scores.items()}
        order = np.argsort(-s, axis=1, kind="stable")
        winner_pic = order[:, 0]
        for candidate, candidate_id in enumerate(IDS):
            j = int(winner_pic[candidate])
            top = np.sort(s[candidate])[::-1]
            record = {
                "query_id": query_id, "gold_state_id": query_id, "candidate_state_id": candidate_id,
                "is_gold": candidate_id == query_id, "outer_fold": outer_fold,
                "raw_score": raw[candidate], "PIC_1_score": s[candidate, 0], "PIC_2_score": s[candidate, 1],
                "PIC_3_score": s[candidate, 2], "PIC_4_score": s[candidate, 3], "multipic_score": multi[candidate],
                "raw_multipic_score": raw_multi[candidate], "mean_score": mean[candidate], "top2mean_score": top2mean[candidate],
                "logmeanexp_score": lme[candidate], "mean_centering_score": centered[candidate], "zscore_score": zscore[candidate],
                "csls_score": csls[candidate], "calibrated_PIC_1_score": calibrated_pics[candidate, 0],
                "calibrated_PIC_2_score": calibrated_pics[candidate, 1], "calibrated_PIC_3_score": calibrated_pics[candidate, 2],
                "calibrated_PIC_4_score": calibrated_pics[candidate, 3], "calibration_score": calibration[candidate],
                "support_score": support_score[candidate], "top1": top[0], "top2": top[1], "gap12": top[0] - top[1],
                "winner_pic_id": f"{candidate_id}:PIC_{j + 1}", "winner_pic_hubness": h[candidate, j],
                "raw_rank": ranks["raw"][candidate], "multipic_rank": ranks["multipic"][candidate],
                "raw_multipic_rank": ranks["raw_multipic"][candidate], "mean_rank": ranks["mean"][candidate],
                "top2mean_rank": ranks["top2mean"][candidate], "logmeanexp_rank": ranks["logmeanexp"][candidate],
                "mean_centering_rank": ranks["mean_centering"][candidate], "zscore_rank": ranks["zscore"][candidate],
                "csls_rank": ranks["csls"][candidate], "calibration_rank": ranks["calibration"][candidate],
                "support_rank": ranks["support"][candidate], "chosen_beta": beta, "chosen_topk": topk,
                "alpha": ALPHA, "embedding_model": "Qwen3-Embedding-0.6B", "candidate_pool_size": 200,
                "candidate_pool_seed": "not_applicable_full_corpus", "background_source": f"outer_fold_{outer_fold}_train_160",
            }
            result.append(record)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    global IDS, ID_TO_INDEX, OLD, NEW, PIC, PIC_SIM_ALL
    if RUN.exists():
        raise FileExistsError(f"Refusing to overwrite existing formal run: {RUN}")
    ROWS, IDS, OLD, NEW, PIC, metadata = load()
    ID_TO_INDEX = {identifier: number for number, identifier in enumerate(IDS)}
    PIC_SIM_ALL = np.einsum("cpd,qd->cpq", PIC, NEW, optimize=True)
    fold_data = json.loads(FOLDS.read_text(encoding="utf-8"))["folds"]
    RUN.mkdir(parents=True)
    all_rows, selections, grid = [], [], []
    for fold in fold_data:
        number, train_ids, test_ids = fold["outer_fold"], fold["train_query_ids"], fold["test_query_ids"]
        topk, beta, inner_grid = choose_parameters(number, train_ids)
        selections.append({"outer_fold": number, "chosen_topk": topk, "chosen_beta": beta, "alpha": ALPHA, "inner_seed": SEED})
        grid.extend(inner_grid)
        all_rows.extend(output_rows(number, test_ids, train_ids, topk, beta))
    required = 200 * 200
    if len(all_rows) != required or not all(np.isfinite(float(v)) for row in all_rows for v in row.values() if isinstance(v, (float, np.floating))):
        raise AssertionError("Master result record count or finiteness invariant failed")
    counts = Counter(row["query_id"] for row in all_rows)
    golds = Counter(row["query_id"] for row in all_rows if row["is_gold"])
    rank_fields = [key for key in all_rows[0] if key.endswith("_rank")]
    if len(counts) != 200 or any(value != 200 for value in counts.values()) or any(golds[q] != 1 for q in counts) or any(not 1 <= int(row[key]) <= 200 for row in all_rows for key in rank_fields):
        raise AssertionError("Query/candidate/gold/rank invariant failed")
    master = RUN / "exp1_master_candidate_results.csv.gz"
    query_summary = RUN / "exp1_query_level_summary.csv.gz"
    write_csv(master, all_rows)
    summaries = []
    for query_id in sorted(counts):
        gold = next(row for row in all_rows if row["query_id"] == query_id and row["is_gold"])
        summaries.append({key: gold[key] for key in gold if key.endswith("_rank") or key in ("query_id", "gold_state_id", "outer_fold", "chosen_beta", "chosen_topk", "alpha")})
    write_csv(query_summary, summaries)
    method_rank = {"Raw": "raw_rank", "MultiPIC-Max": "multipic_rank", "Raw + MultiPIC-Max": "raw_multipic_rank", "Mean": "mean_rank", "Top-2 Mean": "top2mean_rank", "LogMeanExp (tau=0.1)": "logmeanexp_rank", "Mean-centering": "mean_centering_rank", "Z-score": "zscore_rank", "CSLS-style": "csls_rank", "PIC Calibration": "calibration_rank", "Calibration + Support": "support_rank"}
    results = {method: metrics([int(item[field]) for item in summaries]) for method, field in method_rank.items()}
    header = "| Method | R@1 | R@5 | R@10 | R@20 | R@40 | MRR | Mean Gold Rank | Median Gold Rank |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    table = "".join(f"| {name} | {m['R@1']:.3f} | {m['R@5']:.3f} | {m['R@10']:.3f} | {m['R@20']:.3f} | {m['R@40']:.3f} | {m['MRR']:.3f} | {m['mean_gold_rank']:.3f} | {m['median_gold_rank']:.1f} |\n" for name, m in results.items())
    report = "# Experiment 1 — Main Retrieval Result\n\n" + header + table + "\n## Formal integrity checks\n\n- 200 OOF queries; 200 candidates/query; exactly one gold/query; 40,000 candidate-level records.\n- Every rank is in [1, 200]; all score fields are finite.\n- Background statistics use only the corresponding inner-train / final outer-train state_new records.\n- Master candidate result is frozen by the accompanying SHA-256 manifest; future analyses must derive from it without rescoring.\n"
    (RUN / "experiment1_main_results.md").write_text(report, encoding="utf-8")
    (RUN / "exp1_inner_selection_grid.json").write_text(json.dumps(grid, indent=2) + "\n", encoding="utf-8")
    (RUN / "exp1_fold_parameter_selection.json").write_text(json.dumps(selections, indent=2) + "\n", encoding="utf-8")
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "master_file": master.name, "master_sha256": digest(master), "query_summary_file": query_summary.name, "query_summary_sha256": digest(query_summary), "source_hashes": {"dataset": digest(RAW), "folds": digest(FOLDS), "cache": digest(CACHE), "cache_index": digest(INDEX), "cache_metadata": digest(META)}, "configuration": {"inner_folds": 4, "inner_seed": SEED, "topks": TOPKS, "betas": BETAS, "selection": "max R@20, max MRR, min beta, min TopK", "alpha": ALPHA, "logmeanexp_tau": TAU, "epsilon": EPS}}
    (RUN / "exp1_master_candidate_results.frozen.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(ROOT / "configs/experiment_config.yaml", RUN / "experiment_config.yaml")
    shutil.copy2(FOLDS, RUN / "outer_folds.json")
    print(RUN)


if __name__ == "__main__":
    main()
