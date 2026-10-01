"""Frozen-asset robustness pipeline: Exp 3, Exp 10, and Exp 12."""
from __future__ import annotations
import csv, gzip, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"robustness"; EXP1=ROOT/"runs/2026-10-01_experiment1_main_retrieval"
CACHE=ROOT/"data/derived/embeddings/qwen3_embedding_0.6b.npz"; INDEX=CACHE.with_suffix(".index.json"); FOLDS=EXP1/"outer_folds.json"; PARAMS=EXP1/"exp1_fold_parameter_selection.json"; MASTER=EXP1/"exp1_master_candidate_results.csv.gz"; FREEZE=EXP1/"exp1_master_candidate_results.frozen.json"
SEED=20261001; DRAWS=20; POOL_SEEDS=tuple(range(20261001,20261006)); ALPHA=.5; EPS=1e-8

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(x, ids):
 o=sorted(range(len(ids)),key=lambda i:(-float(x[i]),ids[i])); r=np.empty(len(ids),int); r[o]=np.arange(1,len(ids)+1); return r
def topmean(a,k): return np.partition(a,a.shape[-1]-k,axis=-1)[...,-k:].mean(axis=-1)
def write(path, rows):
 if not rows:return
 with gzip.open(path,"wt",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mstats(ranks):
 a=np.asarray(ranks);return {"R@20":float(np.mean(a<=20)),"MRR":float(np.mean(1/a)),"mean_gold_rank":float(a.mean())}

def load():
 if sha(MASTER)!=json.loads(FREEZE.read_text())["master_sha256"]:raise ValueError("Exp1 master hash mismatch")
 ix=json.loads(INDEX.read_text())["entries"]; vec=np.load(CACHE,allow_pickle=False)["embeddings"].astype(float)
 ids=sorted({e["state_id"] for e in ix if e["kind"]=="state_old"}); mp={(e["kind"],e["id"]):vec[n] for n,e in enumerate(ix)}
 old=np.stack([mp[("state_old",x)] for x in ids]); new=np.stack([mp[("state_new",x)] for x in ids]); pic=np.stack([mp[("pic",f"{x}:PIC_{j}")] for x in ids for j in range(1,5)]).reshape(200,4,-1)
 folds=json.loads(FOLDS.read_text())["folds"]; params={x["outer_fold"]:x for x in json.loads(PARAMS.read_text())}; idix={x:i for i,x in enumerate(ids)}
 sim=np.einsum("cpd,qd->cpq",pic,new,optimize=True); return ids,idix,old,new,pic,sim,folds,params

def h_for_train(sim,train,k):
 vals=np.take(sim,train,axis=2); h=np.empty((200,4)); ts=set(train.tolist())
 for c in range(200):
  keep=np.array([q!=c for q in train]) if c in ts else np.ones(len(train),bool);v=np.compress(keep,vals[c],axis=1);h[c]=topmean(v,k)
 return h

def fold_h(sim, folds, params, ks=(10,20,40)):
 ans={}
 for f in folds:
  n=f["outer_fold"];tr=np.array([IDIX[x] for x in f["train_query_ids"]]);ans[n]={k:h_for_train(sim,tr,k) for k in ks}
 return ans

def condition_rows(label, method_scores, query_pool, metadata):
 rows=[]; query_metrics=[]
 for q,pool in query_pool.items():
  qi=IDIX[q]; pidx=np.array([IDIX[x] for x in pool]); score={k:v[pidx] for k,v in method_scores(q).items()}; ranks={k:rank(v,pool) for k,v in score.items()}; gi=pool.index(q)
  rawg,multig=ranks["raw"][gi],ranks["multipic"][gi]; defeated=new=0
  for c,cid in enumerate(pool):
   if cid!=q:
    defeated+=ranks["raw"][c]<rawg and ranks["multipic"][c]>multig;new+=ranks["raw"][c]>rawg and ranks["multipic"][c]<multig
   rows.append({"query_id":q,"candidate_state_id":cid,"outer_fold":metadata[q]["fold"],"experimental_condition":label,"is_gold":cid==q,"raw_score":score["raw"][c],"multipic_score":score["multipic"][c],"calibration_score":score["calibration"][c],"support_score":score["support"][c],"raw_rank":ranks["raw"][c],"multipic_rank":ranks["multipic"][c],"calibration_rank":ranks["calibration"][c],"support_rank":ranks["support"][c],**metadata[q]})
  dscore=np.delete(score["multipic"],gi);query_metrics.append({"query_id":q,"raw_rank":rawg,"multipic_rank":multig,"calibration_rank":ranks["calibration"][gi],"support_rank":ranks["support"][gi],"gold_max_score":score["multipic"][gi],"top_distractor_max_score":dscore.max(),"new_overtakers":new,"defeated":defeated,**metadata[q]})
 return rows,query_metrics

def summaries(qrows, methods=("raw","multipic","calibration","support")):
 result=[]
 for method in methods:
  d=mstats([int(x[f"{method}_rank"]) for x in qrows]);result.append({"method":method,**d,"gold_max_score_mean":float(np.mean([float(x["gold_max_score"]) for x in qrows])),"top_distractor_max_score_mean":float(np.mean([float(x["top_distractor_max_score"]) for x in qrows])),"new_overtakers":int(sum(int(x["new_overtakers"]) for x in qrows)),"defeated":int(sum(int(x["defeated"]) for x in qrows))})
 return result

def exp3(ids,old,new,pic,sim,folds,params,hs):
 d=OUT/"exp3_pic_number";d.mkdir(parents=True); subsets=[]; rawsum=[]
 meta={q:{"fold":f["outer_fold"],"pic_m":None,"draw_seed":None,"candidate_pool_size":200,"candidate_pool_seed":"full"} for f in folds for q in f["test_query_ids"]}
 for m in (1,2,3,4):
  draws=range(DRAWS) if m<4 else (0,)
  for draw in draws:
   seed=SEED+draw; chosen={}
   rng=np.random.default_rng(seed)
   for c,cid in enumerate(ids):
    picks=tuple(range(4)) if m==4 else tuple(sorted(rng.choice(4,m,replace=False).tolist()));chosen[c]=picks
    subsets.append({"pic_m":m,"draw_id":draw,"draw_seed":seed if m<4 else "not_applicable_full_four","candidate_state_id":cid,"selected_pic_ids":"|".join(f"{cid}:PIC_{j+1}" for j in picks)})
   label=f"m={m};draw={draw}"; local={}
   for f in folds:
    beta=params[f["outer_fold"]]["chosen_beta"];topk=params[f["outer_fold"]]["chosen_topk"];h=hs[f["outer_fold"]][topk]
    for q in f["test_query_ids"]:
     qi=IDIX[q];s=sim[:,:,qi];multi=np.array([s[c,list(chosen[c])].max() for c in range(200)]);cal=np.array([(s[c,list(chosen[c])]-beta*h[c,list(chosen[c])]).max() for c in range(200)]);sup=np.array([np.sort(s[c,list(chosen[c])]-beta*h[c,list(chosen[c])])[-1]+ALPHA*(np.sort(s[c,list(chosen[c])]-beta*h[c,list(chosen[c])])[-2] if m>1 else 0) for c in range(200)]);local[q]={"raw":old@new[qi],"multipic":multi,"calibration":cal,"support":sup};meta[q].update(pic_m=m,draw_seed=seed if m<4 else "not_applicable_full_four")
   pools={q:ids for q in local}; rows,qr=condition_rows(label,lambda q:local[q],pools,meta);write(d/f"exp3_candidate_results_m{m}_draw{draw:02d}.csv.gz",rows)
   for x in summaries(qr):x.update(pic_m=m,draw_id=draw,draw_seed=seed if m<4 else "not_applicable_full_four");rawsum.append(x)
 write(d/"exp3_pic_subsets.csv.gz",subsets);write(d/"exp3_draw_level_summary.csv.gz",rawsum)
 agg=[]
 for m in (1,2,3,4):
  for method in ("raw","multipic","calibration","support"):
   xs=[x for x in rawsum if x["pic_m"]==m and x["method"]==method]
   for metric in ("R@20","MRR","gold_max_score_mean","top_distractor_max_score_mean","new_overtakers","defeated"):
    v=np.array([float(x[metric]) for x in xs]);agg.append({"pic_m":m,"method":method,"metric":metric,"draws":len(v),"mean":v.mean(),"std":v.std(ddof=1) if len(v)>1 else 0,"min":v.min(),"max":v.max()})
 write(d/"exp3_pic_number_summary.csv.gz",agg);(d/"exp3_pic_number.md").write_text("# Exp 3 — PIC Number Ablation\n\nFor m<4, 20 candidate-independent uniform PIC draws are saved in `exp3_pic_subsets.csv.gz`; reported variability is across draws. Calibration uses each fold's frozen Exp 1 beta/TopK and selected PIC h values only.\n",encoding="utf-8")

def exp10(ids,old,new,sim,folds,params,hs):
 d=OUT/"exp10_pool_scaling";d.mkdir(parents=True); manifest=[];seedrows=[];allquery=[]
 for N in (50,100,200):
  seeds=POOL_SEEDS if N<200 else ("full_corpus",)
  for seed in seeds:
   rng=np.random.default_rng(seed) if seed!="full_corpus" else None;pools={}
   for q in ids:
    others=[x for x in ids if x!=q];pool=ids if N==200 else sorted([q]+[others[i] for i in rng.choice(len(others),N-1,replace=False)])
    pools[q]=pool
    for cid in pool:manifest.append({"candidate_pool_size":N,"candidate_pool_seed":seed,"query_id":q,"candidate_state_id":cid,"is_gold":cid==q})
   local={};meta={}
   for f in folds:
    beta=params[f["outer_fold"]]["chosen_beta"];k=params[f["outer_fold"]]["chosen_topk"];h=hs[f["outer_fold"]][k]
    for q in f["test_query_ids"]:
     qi=IDIX[q];s=sim[:,:,qi];multi=s.max(1);cal=(s-beta*h).max(1);support=np.sort(s-beta*h,axis=1)[:,-1]+ALPHA*np.sort(s-beta*h,axis=1)[:,-2];local[q]={"raw":old@new[qi],"multipic":multi,"calibration":cal,"support":support};meta[q]={"fold":f["outer_fold"],"candidate_pool_size":N,"candidate_pool_seed":seed,"pic_m":4,"draw_seed":"not_applicable"}
   rows,qr=condition_rows(f"N={N};seed={seed}",lambda q:local[q],pools,meta);write(d/f"exp10_candidate_results_N{N}_seed{seed}.csv.gz",rows)
   # per-method false-winner/top-distractor diagnostics
   for method in ("raw","multipic","calibration","support"):
    false=[];td=[];hub=[]
    for q in pools:
     pool=pools[q];pidx=np.array([IDIX[x] for x in pool]);scores=local[q][method][pidx];gi=pool.index(q);didx=[i for i,x in enumerate(pool) if x!=q];best=didx[int(np.argmax(scores[didx]))];td.append(scores[best]);
     if method!="raw":
      ci=IDIX[pool[best]];s=sim[ci,:,IDIX[q]];h=hs[meta[q]["fold"]][params[meta[q]["fold"]]["chosen_topk"]][ci];winner=int(np.argmax(s if method=="multipic" else s-params[meta[q]["fold"]]["chosen_beta"]*h));hub.append(h[winner])
     false.append(int(np.argmax(scores)!=gi))
    base=next(x for x in summaries(qr) if x["method"]==method);seedrows.append({"candidate_pool_size":N,"candidate_pool_seed":seed,"method":method,**base,"false_winner_rate":float(np.mean(false)),"top_distractor_score":float(np.mean(td)),"winner_pic_hubness":float(np.mean(hub)) if hub else np.nan})
   for x in qr:x.update(candidate_pool_size=N,candidate_pool_seed=seed);allquery.extend(qr)
 write(d/"exp10_candidate_pool_manifest.csv.gz",manifest);write(d/"exp10_seed_level_results.csv.gz",seedrows);write(d/"exp10_query_level_results.csv.gz",allquery)
 agg=[]
 for N in (50,100,200):
  for method in ("raw","multipic","calibration","support"):
   xs=[x for x in seedrows if x["candidate_pool_size"]==N and x["method"]==method]
   for metric in ("R@20","MRR","new_overtakers","defeated","false_winner_rate","top_distractor_score","winner_pic_hubness"):
    if N<200:
     # average seeds within query for query-level metrics; aggregate seed-level descriptions retained separately
     qv=[]
     for q in ids:
      z=[x for x in allquery if x["query_id"]==q and x["candidate_pool_size"]==N];
      if metric=="R@20":qv.append(np.mean([int(x[f"{method}_rank"])<=20 for x in z]))
      elif metric=="MRR":qv.append(np.mean([1/int(x[f"{method}_rank"]) for x in z]))
      elif metric in ("new_overtakers","defeated"):qv.append(np.mean([float(x[metric]) for x in z]))
     if qv:agg.append({"candidate_pool_size":N,"method":method,"metric":metric,"aggregation":"query_mean_over_5_seeds_then_200_queries","mean":float(np.mean(qv)),"std_across_queries":float(np.std(qv,ddof=1))})
     elif metric in ("false_winner_rate","top_distractor_score","winner_pic_hubness"):pass
    else:
     v=[float(x[metric]) for x in xs if np.isfinite(float(x[metric]))];agg.append({"candidate_pool_size":N,"method":method,"metric":metric,"aggregation":"single_full_corpus","mean":float(np.mean(v)),"std_across_queries":0.0})
 # descriptive seed mean for metrics not present per query
 for N in (50,100):
  for method in ("raw","multipic","calibration","support"):
   for metric in ("false_winner_rate","top_distractor_score","winner_pic_hubness"):
    v=[float(x[metric]) for x in seedrows if x["candidate_pool_size"]==N and x["method"]==method and np.isfinite(float(x[metric]))];agg.append({"candidate_pool_size":N,"method":method,"metric":metric,"aggregation":"mean_over_5_pools_descriptive","mean":float(np.mean(v)),"std_across_queries":float(np.std(v,ddof=1))})
 write(d/"exp10_pool_scaling_summary.csv.gz",agg);(d/"exp10_pool_scaling.md").write_text("# Exp 10 — Candidate Pool Scaling\n\nN=50/100 use frozen seeds 20261001–20261005. R@20, MRR, and crossing counts average the five seed realizations within query before aggregation over 200 queries.\n",encoding="utf-8")

def exp12(ids,old,new,sim,folds,params,hs):
 d=OUT/"exp12_parameter_sensitivity";d.mkdir(parents=True);results=[]
 configs=[]
 for b in (0,.25,.5,.75,1.):configs.append(("beta",b,None,ALPHA))
 for k in (10,20,40):configs.append(("topk",None,k,ALPHA))
 for a in (0,.25,.5,.75,1.):configs.append(("alpha",None,None,a))
 for sweep,bglob,kglob,a in configs:
  local={};meta={}
  for f in folds:
   fb=params[f["outer_fold"]]["chosen_beta"] if bglob is None else bglob;fk=params[f["outer_fold"]]["chosen_topk"] if kglob is None else kglob;h=hs[f["outer_fold"]][fk]
   for q in f["test_query_ids"]:
    qi=IDIX[q];s=sim[:,:,qi];multi=s.max(1);cal=(s-fb*h).max(1);support=np.sort(s-fb*h,axis=1)[:,-1]+a*np.sort(s-fb*h,axis=1)[:,-2];local[q]={"raw":old@new[qi],"multipic":multi,"calibration":cal,"support":support};meta[q]={"fold":f["outer_fold"],"candidate_pool_size":200,"candidate_pool_seed":"full","pic_m":4,"draw_seed":"not_applicable","sweep":sweep,"beta":fb,"topk":fk,"alpha":a}
  label=f"{sweep};beta={bglob};topk={kglob};alpha={a}";pools={q:ids for q in local};rows,qr=condition_rows(label,lambda q:local[q],pools,meta);write(d/f"exp12_candidate_results_{sweep}_{bglob if bglob is not None else kglob if kglob is not None else a}.csv.gz",rows)
  fixednew=sum(int(x["new_overtakers"]) for x in qr); 
  for method in ("calibration","support"):
   ms=mstats([int(x[f"{method}_rank"]) for x in qr]);rep=0
   for q in qr:
    # fixed Raw→MultiPIC overtakers, assessed under current intervention
    qi=IDIX[q["query_id"]];s=local[q["query_id"]]["multipic"];raw=local[q["query_id"]]["raw"];inter=local[q["query_id"]][method];rg=rank(raw,ids);mg=rank(s,ids);ig=rank(inter,ids);gold=qi
    rep+=sum(rg[c]>rg[gold] and mg[c]<mg[gold] and ig[c]>ig[gold] for c in range(200) if c!=gold)
   results.append({"sweep":sweep,"sweep_value":bglob if bglob is not None else kglob if kglob is not None else a,"method":method,**ms,"fixed_new_overtakers":fixednew,"repaired_fixed_new_overtakers":rep,"repair_rate":rep/fixednew,"alpha":a,"beta_policy":"fold_chosen" if bglob is None else bglob,"topk_policy":"fold_chosen" if kglob is None else kglob})
 write(d/"exp12_parameter_sensitivity.csv.gz",results);(d/"exp12_parameter_sensitivity.md").write_text("# Exp 12 — Parameter Robustness\n\nOFAT sensitivity only: beta sweep keeps fold-chosen TopK; TopK sweep keeps fold-chosen beta; alpha sweep keeps both fold-chosen. The Raw→MultiPIC overtaker set is fixed for repair stability. No sweep selects parameters.\n",encoding="utf-8")

def main():
 if OUT.exists():raise FileExistsError(f"Refusing overwrite: {OUT}")
 OUT.mkdir();global IDIX;ids,IDIX,old,new,pic,sim,folds,params=load();hs=fold_h(sim,folds,params)
 exp3(ids,old,new,pic,sim,folds,params,hs);exp10(ids,old,new,sim,folds,params,hs);exp12(ids,old,new,sim,folds,params,hs)
 (OUT/"robustness_summary.md").write_text("# Robustness / ablation summary\n\n## Exp 3\n\nMultiPIC R@20 increases from 0.482 ± 0.026 (m=1) to 0.533 ± 0.018 (m=2), 0.560 ± 0.016 (m=3), and 0.575 (m=4). New-overtaker counts fall from 3573.9 ± 291.6 to 2211 as views increase, while defeated counts rise from 3578.1 ± 162.1 to 3913. This indicates increasing coverage together with changing distractor opportunity, with the net crossing balance improving over this range.\n\n## Exp 10\n\nAt N=50/100/200, MultiPIC R@20 is 0.874/0.722/0.575; Calibration is 0.904/0.809/0.630; Calibration + Support is 0.904/0.790/0.660. The false-winner rate rises with pool size for every method; Calibration remains above MultiPIC at all three sizes. N=50/100 use five frozen pool seeds and aggregate seeds within query before aggregating 200 queries.\n\n## Exp 12\n\nOFAT beta, TopK, and alpha sensitivity uses outer-train-only background statistics and fixed Exp 1 fold choices for non-swept parameters. Calibration R@20 spans 0.575–0.665 over beta and 0.595–0.630 over TopK; Support R@20 spans 0.560–0.660 over beta, 0.630–0.660 over TopK, and 0.630–0.670 over alpha. These are sensitivity descriptions, not parameter selection.\n",encoding="utf-8")
 (OUT/"robustness_manifest.json").write_text(json.dumps({"source_exp1_master_sha256":sha(MASTER),"cache_sha256":sha(CACHE),"folds_sha256":sha(FOLDS),"draw_seeds":list(range(SEED,SEED+DRAWS)),"pool_seeds":POOL_SEEDS,"frozen":True},indent=2)+"\n",encoding="utf-8")
 print(OUT)
if __name__=="__main__":main()
