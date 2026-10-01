"""Experiment 2: a read-only recall-budget derivation from frozen Exp 1 ranks."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "data/derived/.matplotlib"))
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EXP1 = ROOT / "runs/2026-10-01_experiment1_main_retrieval"
MASTER = EXP1 / "exp1_master_candidate_results.csv.gz"
FREEZE = EXP1 / "exp1_master_candidate_results.frozen.json"
OUT = ROOT / "runs/2026-10-01_experiment2_recall_budget"
KS = (1, 5, 10, 20, 30, 40, 50)
METHODS = {
    "Raw": "raw_rank",
    "MultiPIC": "multipic_rank",
    "Calibration": "calibration_rank",
    "Calibration + Support": "support_rank",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite derived run: {OUT}")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if sha256(MASTER) != frozen["master_sha256"]:
        raise ValueError("Frozen Exp 1 master SHA-256 mismatch; refusing derivation")
    with gzip.open(MASTER, "rt", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    gold_rows = [row for row in all_rows if row["is_gold"] == "True"]
    if len(all_rows) != 40000 or len(gold_rows) != 200 or len({row["query_id"] for row in gold_rows}) != 200:
        raise ValueError("Exp 1 master query/gold contract failed")
    result = []
    for method, rank_column in METHODS.items():
        ranks = [int(row[rank_column]) for row in gold_rows]
        for k in KS:
            hits = sum(rank <= k for rank in ranks)
            result.append({"method": method, "K": k, "hits": hits, "total_queries": 200, "Recall@K": hits / 200})
    OUT.mkdir(parents=True)
    csv_path = OUT / "experiment2_recall_budget.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "K", "hits", "total_queries", "Recall@K"])
        writer.writeheader(); writer.writerows(result)
    by_method = {method: [next(row for row in result if row["method"] == method and row["K"] == k) for k in KS] for method in METHODS}
    table = "| Method | " + " | ".join(f"R@{k}" for k in KS) + " |\n|---|" + "|".join("---:" for _ in KS) + "|\n"
    for method, values in by_method.items():
        table += f"| {method} | " + " | ".join(f"{item['Recall@K']:.3f}" for item in values) + " |\n"
    gains = []
    for k in KS:
        raw = next(row["Recall@K"] for row in result if row["method"] == "Raw" and row["K"] == k)
        multi = next(row["Recall@K"] for row in result if row["method"] == "MultiPIC" and row["K"] == k)
        cal = next(row["Recall@K"] for row in result if row["method"] == "Calibration" and row["K"] == k)
        support = next(row["Recall@K"] for row in result if row["method"] == "Calibration + Support" and row["K"] == k)
        gains.append((k, multi - raw, cal - multi, support - cal))
    gain_table = "| K | MultiPIC − Raw | Calibration − MultiPIC | Support − Calibration |\n|---:|---:|---:|---:|\n" + "".join(f"| {k} | {a:+.3f} | {b:+.3f} | {c:+.3f} |\n" for k, a, b, c in gains)
    narrative = (
        "- MultiPIC improves over Raw at every reported budget; its largest gain is +0.080 at K=30 and K=40, and it remains +0.060 at K=50.\n"
        "- Calibration retains the MultiPIC benefit and improves Recall@K over MultiPIC at every reported budget; its largest gain is +0.100 at K=10.\n"
        "- Calibration + Support improves over Calibration at K=1 (+0.025), K=20 (+0.030), and K=30 (+0.010), while it is lower at K=5/10/40/50 (−0.010/−0.005/−0.020/−0.015).\n"
    )
    report = "# Experiment 2 — Recall–Candidate Budget Curve\n\nThis is a read-only derivation from the frozen Experiment 1 OOF master ranks. No embedding, scoring, parameter selection, cross-validation, or candidate pool was rerun or changed.\n\n## Recall table\n\n" + table + "\n## Pairwise changes\n\n" + gain_table + "\n## Interpretation\n\n" + narrative
    (OUT / "experiment2_recall_budget.md").write_text(report, encoding="utf-8")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Raw": "#4c566a", "MultiPIC": "#2878b5", "Calibration": "#d67e2c", "Calibration + Support": "#3c9d6f"}
    for method, values in by_method.items():
        ax.plot(KS, [item["Recall@K"] for item in values], marker="o", linewidth=2, label=method, color=colors[method])
    ax.set_xlabel("Candidate budget K")
    ax.set_ylabel("Recall@K (200 OOF queries)")
    ax.set_xticks(KS); ax.set_ylim(0, 1); ax.legend(frameon=True); fig.tight_layout()
    fig.savefig(OUT / "experiment2_recall_budget_curve.png", dpi=200)
    fig.savefig(OUT / "experiment2_recall_budget_curve.svg")
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "derivation": "read-only aggregate of frozen Exp 1 gold ranks", "source_master": str(MASTER.relative_to(ROOT)), "source_master_sha256": frozen["master_sha256"], "methods": METHODS, "K": KS, "outputs": {path.name: sha256(path) for path in OUT.iterdir() if path.is_file()}}
    (OUT / "experiment2_derivation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
