"""Read-only Exp 4–9 analysis derived from frozen Experiment 1 results."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP1 = ROOT / "runs/2026-10-01_experiment1_main_retrieval"
MASTER = EXP1 / "exp1_master_candidate_results.csv.gz"
FREEZE = EXP1 / "exp1_master_candidate_results.frozen.json"
FOLDS = EXP1 / "outer_folds.json"
PARAMS = EXP1 / "exp1_fold_parameter_selection.json"
CACHE = ROOT / "data/derived/embeddings/qwen3_embedding_0.6b.npz"
INDEX = CACHE.with_suffix(".index.json")
OUT = ROOT / "runs/2026-10-01_experiment4_9_analysis"
SEED, B = 20261001, 10_000
METRICS = ("top1", "top2", "gap12", "mean_top2", "mean_all", "support_at_002", "support_at_005", "winner_pic_hubness")


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def num(value): return float(value)
def write_csv(path: Path, rows: list[dict]):
    if not rows: return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def cluster_summary(rows: list[dict], group_key: str, groups: list[str], metrics: tuple[str, ...], rng: np.random.Generator) -> list[dict]:
    """Candidate-level and query-balanced mean with query-cluster bootstrap CIs."""
    queries = sorted({r["query_id"] for r in rows}); qidx = {q: i for i, q in enumerate(queries)}
    draws = rng.integers(0, len(queries), size=(B, len(queries)))
    result = []
    for group in groups:
        selected = [r for r in rows if r[group_key] == group]
        for metric in metrics:
            sums, counts = np.zeros(len(queries)), np.zeros(len(queries))
            bucket = defaultdict(list)
            for r in selected:
                bucket[r["query_id"]].append(num(r[metric]))
            for q, vals in bucket.items(): sums[qidx[q]], counts[qidx[q]] = sum(vals), len(vals)
            boot = sums[draws].sum(axis=1) / np.maximum(counts[draws].sum(axis=1), 1)
            result.append({"analysis_level": "candidate_level", "group": group, "metric": metric, "mean": sums.sum()/counts.sum(), "n_candidates": int(counts.sum()), "n_queries": int((counts>0).sum()), "ci95_low": np.quantile(boot,.025), "ci95_high": np.quantile(boot,.975), "bootstrap_repetitions": B, "bootstrap_seed": SEED})
            qmeans = np.full(len(queries), np.nan); valid = counts > 0; qmeans[valid] = sums[valid]/counts[valid]
            boot_q = np.nanmean(qmeans[draws], axis=1)
            result.append({"analysis_level": "query_balanced", "group": group, "metric": metric, "mean": np.nanmean(qmeans), "n_candidates": int(counts.sum()), "n_queries": int(valid.sum()), "ci95_low": np.nanquantile(boot_q,.025), "ci95_high": np.nanquantile(boot_q,.975), "bootstrap_repetitions": B, "bootstrap_seed": SEED})
    return result


def ratio_boot(rows: list[dict], flag: str, rng: np.random.Generator) -> tuple[int, int, float, float, float]:
    qs = sorted({r["query_id"] for r in rows}); qidx = {q:i for i,q in enumerate(qs)}
    numer, denom = np.zeros(len(qs)), np.zeros(len(qs))
    for r in rows: denom[qidx[r["query_id"]]] += 1; numer[qidx[r["query_id"]]] += bool(r[flag])
    draws = rng.integers(0,len(qs),size=(B,len(qs))); boot=numer[draws].sum(1)/denom[draws].sum(1)
    return int(numer.sum()), int(denom.sum()), float(numer.sum()/denom.sum()), float(np.quantile(boot,.025)), float(np.quantile(boot,.975))


def load_master():
    frozen=json.loads(FREEZE.read_text(encoding="utf-8"))
    if sha(MASTER)!=frozen["master_sha256"]: raise ValueError("Frozen Exp 1 master hash mismatch")
    with gzip.open(MASTER,"rt",encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
    if len(rows)!=40000: raise ValueError("Expected 40,000 master records")
    return rows, frozen


def crossings(master: list[dict]):
    byq=defaultdict(list)
    for r in master: byq[r["query_id"]].append(r)
    cross, qsummary=[], []
    for q, rs in sorted(byq.items()):
        gold=next(r for r in rs if r["is_gold"]=="True")
        rawg, multig=int(gold["raw_rank"]), int(gold["multipic_rank"]); defeated=new=0
        for r in rs:
            if r["is_gold"]=="True": continue
            raw_before=int(r["raw_rank"])<rawg; multi_before=int(r["multipic_rank"])<multig
            kind="Defeated" if raw_before and not multi_before else "New overtaker" if not raw_before and multi_before else None
            if not kind: continue
            defeated += kind=="Defeated"; new += kind=="New overtaker"
            scores=sorted([num(r[f"PIC_{j}_score"]) for j in range(1,5)],reverse=True); top=scores[0]
            item={"query_id":q,"gold_state_id":gold["gold_state_id"],"candidate_state_id":r["candidate_state_id"],"outer_fold":r["outer_fold"],"crossing_type":kind,"raw_gold_rank":rawg,"multipic_gold_rank":multig,"rank_gain":rawg-multig,"raw_gold_score":gold["raw_score"],"raw_candidate_score":r["raw_score"],"gold_multipic_score":gold["multipic_score"],"candidate_multipic_score":r["multipic_score"],**{f"PIC_{j}_score":r[f"PIC_{j}_score"] for j in range(1,5)},"top1":top,"top2":scores[1],"gap12":top-scores[1],"mean_top2":sum(scores[:2])/2,"mean_all":sum(scores)/4,"winner_pic_id":r["winner_pic_id"],"winner_pic_hubness":r["winner_pic_hubness"],"support_at_002":sum(x>=top-.02 for x in scores)/4,"support_at_005":sum(x>=top-.05 for x in scores)/4,"calibration_score":r["calibration_score"],"support_score":r["support_score"],"calibrated_rank":r["calibration_rank"],"support_rank":r["support_rank"],"calibration_gold_rank":gold["calibration_rank"],"support_gold_rank":gold["support_rank"]}
            item["repaired_by_calibration"]=int(item["calibrated_rank"])>int(item["calibration_gold_rank"])
            item["repaired_by_support"]=int(item["support_rank"])>int(item["support_gold_rank"])
            cross.append(item)
        qitem={"query_id":q,"outer_fold":gold["outer_fold"],"n_defeated":defeated,"n_new_overtaker":new,"raw_gold_rank":rawg,"multipic_gold_rank":multig,"rank_gain":rawg-multig,"calibration_gold_rank":gold["calibration_rank"],"support_gold_rank":gold["support_rank"]}
        if qitem["rank_gain"] != defeated-new: raise AssertionError(f"Crossing invariant failed for {q}")
        qsummary.append(qitem)
    return cross,qsummary


def taxonomy_thresholds():
    folds=json.loads(FOLDS.read_text(encoding="utf-8"))["folds"]; params={x["outer_fold"]:x for x in json.loads(PARAMS.read_text(encoding="utf-8"))}
    index=json.loads(INDEX.read_text(encoding="utf-8"))["entries"]; vec=np.load(CACHE,allow_pickle=False)["embeddings"].astype(float)
    ids=sorted({e["state_id"] for e in index if e["kind"]=="state_old"}); ix={x:i for i,x in enumerate(ids)}; mp={(e["kind"],e["id"]):vec[n] for n,e in enumerate(index)}
    news=np.stack([mp[("state_new",x)] for x in ids]); pics=np.stack([mp[("pic",f"{x}:PIC_{j}")] for x in ids for j in range(1,5)]).reshape(200,4,-1); sims=np.einsum("cpd,qd->cpq",pics,news,optimize=True)
    output=[]
    for fold in folds:
        f=fold["outer_fold"]; train=np.array([ix[x] for x in fold["train_query_ids"]]); k=params[f]["chosen_topk"]; h=np.empty((200,4))
        values=np.take(sims,train,axis=2)
        trainset=set(train.tolist())
        for c in range(200):
            keep=np.array([q!=c for q in train]) if c in trainset else np.ones(len(train),bool)
            v=np.compress(keep,values[c],axis=1); h[c]=np.partition(v,v.shape[1]-k,axis=1)[:,-k:].mean(axis=1)
        ref_h,ref_gap=[],[]
        for q in train:
            s=sims[:,:,q]; w=np.argmax(s,axis=1); order=np.sort(s,axis=1)[:,::-1]; ref_h.extend(h[np.arange(200),w]); ref_gap.extend(order[:,0]-order[:,1])
        output.append({"outer_fold":f,"chosen_topk":k,"reference_observations":len(ref_h),"hubness_median":float(np.median(ref_h)),"gap12_median":float(np.median(ref_gap)),"reference_source":"outer_train_candidate_x_query_raw_PIC_only"})
    return output


def main():
    if OUT.exists(): raise FileExistsError(f"Refusing overwrite: {OUT}")
    OUT.mkdir(parents=True); rng=np.random.default_rng(SEED); master,frozen=load_master(); cross,qsum=crossings(master)
    crossfile=OUT/"exp4_crossing_master.csv.gz"
    with gzip.open(crossfile,"wt",encoding="utf-8",newline="") as f: w=csv.DictWriter(f,fieldnames=list(cross[0])); w.writeheader(); w.writerows(cross)
    write_csv(OUT/"exp4_crossing_query_summary.csv",qsum)
    counts=Counter(x["crossing_type"] for x in cross); inv_ok=all(x["rank_gain"]==x["n_defeated"]-x["n_new_overtaker"] for x in qsum)
    (OUT/"exp4_crossing_summary.md").write_text(f"# Experiment 4 — Crossing decomposition\n\n- Defeated distractors: {counts['Defeated']}\n- New overtakers: {counts['New overtaker']}\n- Query-level rank identity verified for all 200 queries: **{inv_ok}**.\n",encoding="utf-8")
    # Freeze crossing master before all later analyses.
    cross_manifest={"source_exp1_master_sha256":frozen["master_sha256"],"crossing_master_sha256":sha(crossfile),"definition":"rank-based deterministic Exp 1 ordering; defeated: raw candidate before gold and MultiPIC candidate after gold; new overtaker: reverse","records":len(cross)}
    (OUT/"exp4_crossing_master.frozen.json").write_text(json.dumps(cross_manifest,indent=2)+"\n",encoding="utf-8")
    # Exp 5
    exp5=cluster_summary(cross,"crossing_type",["New overtaker","Defeated"],METRICS,rng); write_csv(OUT/"exp5_mechanism_analysis.csv",exp5)
    (OUT/"exp5_mechanism_analysis.md").write_text("# Experiment 5 — Mechanism analysis\n\nCandidate-level and query-balanced group means with 10,000 query-cluster bootstrap CIs are in `exp5_mechanism_analysis.csv`. Results are descriptive associations, not causal claims.\n",encoding="utf-8")
    # Exp 6: outer-train-only medians, then classify fixed crossings.
    thresholds=taxonomy_thresholds(); write_csv(OUT/"exp6_outer_train_thresholds.csv",thresholds); th={str(x["outer_fold"]):x for x in thresholds}; tax=[]
    for r in cross:
        t=th[r["outer_fold"]]; highh=num(r["winner_pic_hubness"])>=t["hubness_median"]; highg=num(r["gap12"])>=t["gap12_median"]; tax.append({"query_id":r["query_id"],"candidate_state_id":r["candidate_state_id"],"outer_fold":r["outer_fold"],"crossing_type":r["crossing_type"],"taxonomy":"High Hubness + High Gap" if highh and highg else "High Hubness + Low Gap" if highh else "Low Hubness + High Gap" if highg else "Low Hubness + Low Gap"})
    write_csv(OUT/"exp6_crossing_taxonomy_assignments.csv",tax)
    exp6=cluster_summary(tax,"taxonomy",["Low Hubness + Low Gap","Low Hubness + High Gap","High Hubness + Low Gap","High Hubness + High Gap"],("dummy",),rng) if False else []
    # Frequencies use the crossing candidates within each taxonomy as denominator, candidate-level and query-balanced.
    for level in ("candidate_level","query_balanced"):
        for group in sorted({x["taxonomy"] for x in tax}):
            members=[x for x in tax if x["taxonomy"]==group]; qs=sorted({x["query_id"] for x in members});
            for typ in ("New overtaker","Defeated"):
                if level=="candidate_level": vals=[1.0 if x["crossing_type"]==typ else 0.0 for x in members]
                else: vals=[sum(x["crossing_type"]==typ for x in members if x["query_id"]==q)/sum(x["query_id"]==q for x in members) for q in qs]
                exp6.append({"analysis_level":level,"taxonomy":group,"crossing_type":typ,"count":sum(x["crossing_type"]==typ for x in members),"denominator":len(members),"frequency":float(np.mean(vals)),"n_queries":len(qs)})
    write_csv(OUT/"exp6_failure_taxonomy.csv",exp6)
    (OUT/"exp6_failure_taxonomy.md").write_text("# Experiment 6 — Hubness × isolation taxonomy\n\nThresholds are fold-specific medians from outer-train candidate×query raw-PIC reference distributions only; assignments and frequencies derive solely from the frozen crossing master.\n",encoding="utf-8")
    # Exp 7
    new=[r for r in cross if r["crossing_type"]=="New overtaker"]
    repair=[]
    for intervention,flag,metrics in (("Calibration","repaired_by_calibration",("winner_pic_hubness","top1","gap12","support_at_002","support_at_005")),("Calibration + Support","repaired_by_support",("gap12","support_at_002","support_at_005","winner_pic_hubness"))):
        for r in new: r["repair_group"]="Repaired" if r[flag] else "Not repaired"
        for x in cluster_summary(new,"repair_group",["Repaired","Not repaired"],metrics,rng): x["intervention"]=intervention; repair.append(x)
    write_csv(OUT/"exp7_repair_analysis.csv",repair); (OUT/"exp7_repair_analysis.md").write_text("# Experiment 7 — Intervention repair analysis\n\nAll groups are original fixed MultiPIC new overtakers. Bootstrap CIs cluster by query; results describe associations.\n",encoding="utf-8")
    # Exp 8
    exp8=[]
    for intervention,flag in (("Calibration","repaired_by_calibration"),("Calibration + Support","repaired_by_support")):
        n,d,rate,lo,hi=ratio_boot(new,flag,rng); exp8.append({"intervention":intervention,"outcome":"harm_suppression","numerator":n,"denominator":d,"rate":rate,"ci95_low":lo,"ci95_high":hi,"bootstrap_repetitions":B,"bootstrap_seed":SEED})
        defeated=[r for r in cross if r["crossing_type"]=="Defeated"]
        f="repaired_by_calibration" if intervention=="Calibration" else "repaired_by_support"; n,d,rate,lo,hi=ratio_boot(defeated,f,rng); exp8.append({"intervention":intervention,"outcome":"benefit_retention","numerator":n,"denominator":d,"rate":rate,"ci95_low":lo,"ci95_high":hi,"bootstrap_repetitions":B,"bootstrap_seed":SEED})
    write_csv(OUT/"exp8_harm_benefit.csv",exp8); (OUT/"exp8_harm_benefit.md").write_text("# Experiment 8 — Harm suppression vs benefit retention\n\nRates use the fixed Exp 4 crossing sets as denominators; CIs use query-cluster bootstrap.\n",encoding="utf-8")
    # Exp 9
    critical=[q for q in qsum if int(q["raw_gold_rank"])<=20 and int(q["multipic_gold_rank"])>20]; cset={q["query_id"] for q in critical}; critical_cross=[r for r in new if r["query_id"] in cset]
    desc=[]
    for metric in ("winner_pic_hubness","gap12","support_at_002","support_at_005"):
        vals=[num(r[metric]) for r in critical_cross]; desc.append({"metric":metric,"mean":float(np.mean(vals)) if vals else np.nan,"median":float(np.median(vals)) if vals else np.nan,"n_new_overtakers":len(vals),"n_critical_queries":len(critical)})
    recovery=[]
    for label,key in (("Calibration","calibration_gold_rank"),("Calibration + Support","support_gold_rank")):
        n=sum(int(q[key])<=20 for q in critical); recovery.append({"intervention":label,"recovered_queries":n,"critical_queries":len(critical),"recovery_rate":n/len(critical) if critical else np.nan})
    write_csv(OUT/"exp9_critical_overtakers.csv",desc); write_csv(OUT/"exp9_critical_recovery.csv",recovery)
    (OUT/"exp9_critical_overtakers.md").write_text(f"# Experiment 9 — Critical overtakers\n\nCritical harmful queries (Raw rank ≤20, MultiPIC rank >20): {len(critical)} / 200 ({len(critical)/200:.3f}).\n\nRecovery results are in `exp9_critical_recovery.csv`; overtaker descriptors are in `exp9_critical_overtakers.csv`.\n",encoding="utf-8")
    summary=f"# Experiment 4–9 summary\n\n## Exp 4\n\n{counts['Defeated']} defeated distractors; {counts['New overtaker']} new overtakers. Rank identity verified for all queries: {inv_ok}.\n\n## Exp 5\n\nMechanism statistics are query-cluster-bootstrap association estimates from the frozen crossing master.\n\n## Exp 6\n\nEach of the five outer folds has independent outer-train median thresholds for hubness and gap12; outer-test crossings did not define thresholds.\n\n## Exp 7–8\n\nRepair and harm/benefit rates use the unchanged Exp 4 crossings.\n\n## Exp 9\n\n{len(critical)} critical harmful transitions at K=20.\n\nAll Exp 5–9 records trace to `exp4_crossing_master.csv.gz`; Exp 1 master was read-only throughout.\n"
    (OUT/"experiment4_9_summary.md").write_text(summary,encoding="utf-8")
    manifest={"created_at_utc":datetime.now(timezone.utc).isoformat(),"source_exp1_master_sha256":frozen["master_sha256"],"crossing_master_sha256":sha(crossfile),"bootstrap":{"repetitions":B,"seed":SEED,"cluster":"query_id"},"outer_train_taxonomy":"per-fold median hubness and gap12 reference distributions computed from frozen embeddings only; no ranks or retrieval recomputed","outputs":{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}
    (OUT/"experiment4_9_derivation_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(OUT)

if __name__=="__main__": main()
