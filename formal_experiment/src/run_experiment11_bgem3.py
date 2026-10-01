"""Independent BGE-M3 replication of Experiment 1, plus Exp-11 diagnostics."""
from __future__ import annotations
import csv, gzip, hashlib, json, random, shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import run_experiment1 as core

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/paired_states_v2/paired_states.json'
FOLDS=ROOT/'configs/outer_folds.json'
RUN=ROOT/'runs/2026-10-01_exp11_bgem3'
CACHE=RUN/'embeddings/bge_m3.npz'; INDEX=RUN/'embeddings/bge_m3.index.json'; META=RUN/'embeddings/bge_m3.metadata.json'
SEED=20261001; TOPKS=(10,20,40); BETAS=(.25,.5,.75,1.); ALPHA=.5; EPS=1e-8

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,rows):
    with gzip.open(path,'wt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)
def read(path):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def metrics(rs): return core.metrics([int(x) for x in rs])

def load():
    rows=json.loads(RAW.read_text(encoding='utf-8')); ids=sorted(x['uid'] for x in rows)
    idx=json.loads(INDEX.read_text(encoding='utf-8'))['entries']; vec=np.load(CACHE,allow_pickle=False)['embeddings'].astype(np.float64)
    norms=np.linalg.norm(vec,axis=1)
    if vec.shape!=(1200,1024) or len(idx)!=1200 or not np.isfinite(vec).all() or not np.allclose(norms,1,atol=1e-3): raise ValueError('BGE cache preflight failed')
    m={(e['kind'],e['id']):vec[n] for n,e in enumerate(idx)}
    old=np.stack([m[('state_old',x)] for x in ids]); new=np.stack([m[('state_new',x)] for x in ids])
    pic=np.stack([m[('pic',f'{x}:PIC_{j}')] for x in ids for j in range(1,5)]).reshape(200,4,-1)
    return ids,old,new,pic,{'norm_min':float(norms.min()),'norm_max':float(norms.max())}

def background(pic_sim,train,k):
    a=np.take(pic_sim,train,axis=2); h=np.empty((200,4))
    ts=set(train.tolist())
    for c in range(200):
        keep=np.array([q!=c for q in train]) if c in ts else np.ones(len(train),bool)
        h[c]=core.topmean(np.compress(keep,a[c],axis=1),k)
    return h
def choose(fold, train_ids):
    shuffled=sorted(train_ids);random.Random(SEED).shuffle(shuffled); grid=[]
    for k in TOPKS:
      scores={b:[] for b in BETAS}
      for n in range(4):
        val=shuffled[n*40:(n+1)*40]; tr=[x for x in shuffled if x not in set(val)]
        h=background(core.PIC_SIM_ALL,np.array([core.ID_TO_INDEX[x] for x in tr]),k)
        for b in BETAS:
          for qid in val:
            q=core.ID_TO_INDEX[qid]; s=np.einsum('cpd,d->cp',core.PIC,core.NEW[q]); scores[b].append(int(core.rank((s-b*h).max(1),core.IDS)[q]))
      for b,rs in scores.items(): grid.append({'outer_fold':fold,'topk':k,'beta':b,**metrics(rs)})
    winner=sorted(grid,key=lambda z:(-z['R@20'],-z['MRR'],z['beta'],z['topk']))[0]
    return winner['topk'],winner['beta'],grid
def rows_for_fold(fold,test_ids,train_ids,k,b):
    h=background(core.PIC_SIM_ALL,np.array([core.ID_TO_INDEX[x] for x in train_ids]),k); result=[]
    for qid in test_ids:
      q=core.ID_TO_INDEX[qid]; s=np.einsum('cpd,d->cp',core.PIC,core.NEW[q]); raw=core.OLD@core.NEW[q]; multi=s.max(1); calpics=s-b*h; cal=calpics.max(1); sortedcal=np.sort(calpics,axis=1)[:,::-1]; sup=sortedcal[:,0]+ALPHA*sortedcal[:,1]
      scores={'raw':raw,'multipic':multi,'calibration':cal,'support':sup}; ranks={n:core.rank(x,core.IDS) for n,x in scores.items()}; wp=np.argmax(s,axis=1)
      for c,cid in enumerate(core.IDS):
        top=np.sort(s[c])[::-1]; j=int(wp[c])
        result.append({'query_id':qid,'gold_state_id':qid,'candidate_state_id':cid,'is_gold':cid==qid,'outer_fold':fold,'raw_score':raw[c],'PIC_1_score':s[c,0],'PIC_2_score':s[c,1],'PIC_3_score':s[c,2],'PIC_4_score':s[c,3],'multipic_score':multi[c],'top1':top[0],'top2':top[1],'gap12':top[0]-top[1],'winner_pic_id':f'{cid}:PIC_{j+1}','winner_pic_hubness':h[c,j],'calibrated_PIC_1_score':calpics[c,0],'calibrated_PIC_2_score':calpics[c,1],'calibrated_PIC_3_score':calpics[c,2],'calibrated_PIC_4_score':calpics[c,3],'calibration_score':cal[c],'support_score':sup[c],'raw_rank':ranks['raw'][c],'multipic_rank':ranks['multipic'][c],'calibration_rank':ranks['calibration'][c],'support_rank':ranks['support'][c],'chosen_beta':b,'chosen_topk':k,'alpha':ALPHA,'embedding_model':'BAAI/bge-m3','candidate_pool_size':200,'candidate_pool_seed':'not_applicable_full_corpus','background_source':f'outer_fold_{fold}_train_160'})
    return result
def crossing(master):
    out=[]; qs=[]
    byq={}
    for r in master:byq.setdefault(r['query_id'],[]).append(r)
    for q,rs in byq.items():
      gold=next(x for x in rs if x['is_gold']); graw=int(gold['raw_rank']);gmulti=int(gold['multipic_rank']); nd=no=0
      for r in rs:
        if r is gold:continue
        raw=int(r['raw_rank'])<graw; multi=int(r['multipic_rank'])<gmulti
        typ='defeated_distractor' if raw and not multi else ('new_overtaker' if not raw and multi else None)
        if not typ:continue
        nd+=typ=='defeated_distractor';no+=typ=='new_overtaker'
        z={k:r[k] for k in r};z.update(crossing_type=typ,raw_gold_rank=graw,multipic_gold_rank=gmulti,rank_gain=graw-gmulti,calibration_gold_rank=gold['calibration_rank'],support_gold_rank=gold['support_rank'],raw_gold_score=gold['raw_score'],raw_candidate_score=r['raw_score'],gold_multipic_score=gold['multipic_score'],candidate_multipic_score=r['multipic_score'],mean_top2=(float(r['top1'])+float(r['top2']))/2,mean_all=np.mean([float(r[f'PIC_{j}_score']) for j in range(1,5)]),**{'support@0.02':sum(float(r['PIC_'+str(j)+'_score'])>=float(r['top1'])-.02 for j in range(1,5)),'support@0.05':sum(float(r['PIC_'+str(j)+'_score'])>=float(r['top1'])-.05 for j in range(1,5))})
        out.append(z)
      qs.append({'query_id':q,'outer_fold':gold['outer_fold'],'raw_gold_rank':graw,'multipic_gold_rank':gmulti,'rank_gain':graw-gmulti,'n_defeated':nd,'n_new_overtaker':no,'identity_holds':graw-gmulti==nd-no,'calibration_gold_rank':gold['calibration_rank'],'support_gold_rank':gold['support_rank']})
    if not all(x['identity_holds'] for x in qs):raise AssertionError('crossing identity failed')
    return out,qs
def bootstrap(cross,metric,typ):
    byq={};
    for r in cross:
      if r['crossing_type']==typ:byq.setdefault(r['query_id'],[]).append(float(r[metric]))
    vals=np.array([np.mean(v) for v in byq.values()]); rng=np.random.default_rng(SEED); means=np.array([rng.choice(vals,len(vals),replace=True).mean() for _ in range(10000)])
    return float(vals.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975)),len(vals)
def main():
    global_ids,old,new,pic,audit=load();core.IDS=global_ids;core.ID_TO_INDEX={x:i for i,x in enumerate(global_ids)};core.OLD=old;core.NEW=new;core.PIC=pic;core.PIC_SIM_ALL=np.einsum('cpd,qd->cpq',pic,new,optimize=True)
    folds=json.loads(FOLDS.read_text(encoding='utf-8'))['folds']; allrows=[];sel=[];grid=[]
    for f in folds:
      k,b,g=choose(f['outer_fold'],f['train_query_ids']);sel.append({'outer_fold':f['outer_fold'],'chosen_topk':k,'chosen_beta':b,'alpha':ALPHA,'inner_seed':SEED});grid+=g;allrows+=rows_for_fold(f['outer_fold'],f['test_query_ids'],f['train_query_ids'],k,b)
    counts=Counter(x['query_id'] for x in allrows);golds=Counter(x['query_id'] for x in allrows if x['is_gold']);
    if len(allrows)!=40000 or len(counts)!=200 or any(v!=200 for v in counts.values()) or any(golds[x]!=1 for x in counts) or not all(np.isfinite(float(v)) for r in allrows for v in r.values() if isinstance(v,(float,np.floating))):raise AssertionError('master integrity failed')
    master=RUN/'exp11_bgem3_master_candidate_results.csv.gz'; write(master,allrows)
    qsum=[next(r for r in allrows if r['query_id']==q and r['is_gold']) for q in sorted(counts)];write(RUN/'exp11_bgem3_query_summary.csv.gz',qsum)
    method={'Raw':'raw_rank','MultiPIC-Max':'multipic_rank','PIC Calibration':'calibration_rank','Calibration + Support':'support_rank'}; ms={n:metrics([r[f] for r in qsum]) for n,f in method.items()}
    table='| Method | R@1 | R@5 | R@10 | R@20 | R@40 | MRR | Mean Gold Rank | Median Gold Rank |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'+''.join(f"| {n} | {x['R@1']:.3f} | {x['R@5']:.3f} | {x['R@10']:.3f} | {x['R@20']:.3f} | {x['R@40']:.3f} | {x['MRR']:.3f} | {x['mean_gold_rank']:.3f} | {x['median_gold_rank']:.1f} |\n" for n,x in ms.items())
    (RUN/'exp11_bgem3_main_results.md').write_text('# Experiment 11 — BGE-M3 main retrieval\n\n'+table+'\nProtocol: 200 OOF queries, fixed N=200 pool, train-only 4-fold inner selection.\n',encoding='utf-8')
    (RUN/'exp11_bgem3_inner_selection_grid.json').write_text(json.dumps(grid,indent=2)+'\n');(RUN/'exp11_bgem3_fold_parameter_selection.json').write_text(json.dumps(sel,indent=2)+'\n')
    cross,cqs=crossing(allrows);write(RUN/'exp11_bgem3_crossing_master.csv.gz',cross);write(RUN/'exp11_bgem3_crossing_query_summary.csv.gz',cqs)
    mech=[]
    for metric in ('winner_pic_hubness','top1','top2','gap12','support@0.02','support@0.05'):
      for typ in ('new_overtaker','defeated_distractor'):
        a,lo,hi,n=bootstrap(cross,metric,typ);vals=[float(r[metric]) for r in cross if r['crossing_type']==typ];mech.append({'metric':metric,'crossing_type':typ,'candidate_n':len(vals),'candidate_mean':float(np.mean(vals)),'query_balanced_mean':a,'cluster_bootstrap_ci_low':lo,'cluster_bootstrap_ci_high':hi,'query_n':n})
    write(RUN/'exp11_bgem3_mechanism_analysis.csv.gz',mech)
    repair=[]
    for intervention,field in [('Calibration','calibration_rank'),('Calibration + Support','support_rank')]:
      for repaired in (True,False):
        subset=[r for r in cross if r['crossing_type']=='new_overtaker' and ((int(r[field])>=int(r['calibration_gold_rank'] if intervention=='Calibration' else r['support_gold_rank']))==repaired)]
        for metric in ('winner_pic_hubness','top1','gap12','support@0.02','support@0.05'):
          repair.append({'intervention':intervention,'repaired':repaired,'metric':metric,'n':len(subset),'mean':float(np.mean([float(r[metric]) for r in subset])) if subset else None})
    write(RUN/'exp11_bgem3_repair_analysis.csv.gz',repair)
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'model':'BAAI/bge-m3','cache_audit':audit,'master_sha256':digest(master),'crossing_sha256':digest(RUN/'exp11_bgem3_crossing_master.csv.gz'),'source_hashes':{'dataset':digest(RAW),'folds':digest(FOLDS),'cache':digest(CACHE)},'protocol':{'inner_folds':4,'inner_seed':SEED,'topks':TOPKS,'betas':BETAS,'selection':'max R@20, max MRR, min beta, min TopK','alpha':ALPHA,'epsilon':EPS}}
    (RUN/'exp11_bgem3_master_candidate_results.frozen.json').write_text(json.dumps(manifest,indent=2)+'\n');shutil.copy2(FOLDS,RUN/'outer_folds.json')
    print(RUN)
if __name__=='__main__':main()
