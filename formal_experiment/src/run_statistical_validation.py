"""Read-only statistical validation for frozen Exp 1 and Exp 4–9 artifacts."""
from __future__ import annotations
import csv, gzip, json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EXP1=ROOT/'runs/2026-10-01_experiment1_main_retrieval'
EXP49=ROOT/'runs/2026-10-01_experiment4_9_analysis'
OUT=ROOT/'statistical_validation'
SEED=20261001; B=10_000

def readgz(p):
 with gzip.open(p,'rt',encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 with open(p,'w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ci(x):return float(np.quantile(x,.025)),float(np.quantile(x,.975))
def pboot(x):
 # Two-sided, finite-sample corrected tail probability for a null effect of zero.
 return min(1.,2*min((np.sum(x<=0)+1)/(len(x)+1),(np.sum(x>=0)+1)/(len(x)+1)))
def binom_exact(b,c):
 n=b+c
 if not n:return 1.
 k=min(b,c);obs=math.comb(n,k)*.5**n
 return min(1.,sum(math.comb(n,i)*.5**n for i in range(n+1) if math.comb(n,i)*.5**n<=obs+1e-15))
def main():
 OUT.mkdir(exist_ok=False)
 master=readgz(EXP1/'exp1_master_candidate_results.csv.gz')
 byq={}
 for r in master:
  if r['is_gold']=='True':byq[r['query_id']]=r
 qs=sorted(byq); rng=np.random.default_rng(SEED); draws=rng.integers(0,len(qs),size=(B,len(qs)))
 paired=[]
 comparisons=[('MultiPIC','Calibration','multipic_rank','calibration_rank'),('MultiPIC','Calibration + Support','multipic_rank','support_rank')]
 for base,new,bf,nf in comparisons:
  a=np.array([float(byq[q][bf]) for q in qs]);z=np.array([float(byq[q][nf]) for q in qs])
  for name,fun in [('R@20',lambda x:(x<=20).mean()),('MRR',lambda x:(1/x).mean()),('Mean Gold Rank',lambda x:x.mean())]:
   point=fun(z)-fun(a);boot=np.array([fun(z[d])-fun(a[d]) for d in draws]);lo,hi=ci(boot)
   paired.append({'comparison':f'{base} vs {new}','baseline':base,'new':new,'metric':name,'effect_definition':'new - baseline','effect_size':point,'ci95_low':lo,'ci95_high':hi,'bootstrap_p_value':pboot(boot),'unit':'query_id','n_queries':len(qs),'bootstrap_repetitions':B,'seed':SEED})
 write(OUT/'paired_bootstrap.csv',paired)
 mc=[]
 for base,new,bf,nf in comparisons:
  a=np.array([int(byq[q][bf]) for q in qs]);z=np.array([int(byq[q][nf]) for q in qs])
  for k in (1,5,10,20):
   ah=a<=k;nh=z<=k;b=int(np.sum(ah&nh));m=int(np.sum(~ah&~nh));loss=int(np.sum(ah&~nh));gain=int(np.sum(~ah&nh))
   mc.append({'comparison':f'{base} vs {new}','baseline':base,'new':new,'metric':f'R@{k}','both_hit':b,'both_miss':m,'baseline_hit_to_new_miss':loss,'baseline_miss_to_new_hit':gain,'discordant_total':loss+gain,'exact_mcnemar_p_value':binom_exact(loss,gain),'n_queries':len(qs)})
 write(OUT/'mcnemar.csv',mc)
 cross=readgz(EXP49/'exp4_crossing_master.csv.gz')
 allq=sorted({r['query_id'] for r in master})
 # Each comparison retains query clusters, including empty groups. Candidate-level weights every candidate in a sampled query; query-balanced weighs a present group-query once.
 specs=[]
 for metric in ('top1','top2','gap12','mean_top2','mean_all','support_at_002','support_at_005','winner_pic_hubness'):
  specs.append(('Exp5 New overtaker - Defeated','New overtaker','Defeated',metric))
 for intervention,metrics in [('Calibration',('winner_pic_hubness','top1','gap12','support_at_002','support_at_005')),('Calibration + Support',('winner_pic_hubness','gap12','support_at_002','support_at_005'))]:
  for metric in metrics:specs.append((f'Exp7 {intervention}: Repaired - Not repaired','Repaired','Not repaired',metric,intervention))
 result=[]
 for spec in specs:
  label,g1,g0,metric,*iv=spec; iv=iv[0] if iv else None
  maps={g:{} for g in (g1,g0)}
  for r in cross:
   group=r['crossing_type']
   if label.startswith('Exp5'):
    group='New overtaker' if group=='New overtaker' else 'Defeated'
   else:
    if group!='New overtaker':continue
    field='calibrated_rank' if iv=='Calibration' else 'support_rank'; goldfield='calibration_gold_rank' if iv=='Calibration' else 'support_gold_rank'
    group='Repaired' if int(r[field])>=int(r[goldfield]) else 'Not repaired'
   if group in maps:maps[group].setdefault(r['query_id'],[]).append(float(r[metric]))
  for level in ('candidate_level','query_balanced'):
   # Build per-query numerators/denominators once, then resample the entire
   # query clusters by array indexing. This is algebraically identical to
   # resampling the candidate lists, but avoids treating candidates as IID.
   numer=[];denom=[]
   for group in (g1,g0):
    sums=np.array([sum(maps[group].get(q,[])) for q in allq],float)
    counts=np.array([len(maps[group].get(q,[])) for q in allq],float)
    if level=='query_balanced':
     sums=np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0);counts=(counts>0).astype(float)
    numer.append(sums);denom.append(counts)
   def stat(sample,index):
    d=denom[index][sample].sum(axis=1) if np.ndim(sample)==2 else denom[index][sample].sum()
    n=numer[index][sample].sum(axis=1) if np.ndim(sample)==2 else numer[index][sample].sum()
    return np.divide(n,d,out=np.full_like(np.asarray(n,dtype=float),np.nan),where=d>0)
   point=float(stat(np.arange(len(allq)),0)-stat(np.arange(len(allq)),1))
   boot=stat(draws,0)-stat(draws,1);lo,hi=ci(boot)
   n1=sum(len(x) for x in maps[g1].values());n0=sum(len(x) for x in maps[g0].values())
   q1=len(maps[g1]);q0=len(maps[g0])
   result.append({'comparison':label,'analysis_level':level,'metric':metric,'effect_definition':f'{g1} - {g0}','effect_size':point,'ci95_low':lo,'ci95_high':hi,'bootstrap_p_value':pboot(boot),'group1_candidates':n1,'group0_candidates':n0,'group1_queries':q1,'group0_queries':q0,'cluster_universe_queries':len(allq),'bootstrap_repetitions':B,'seed':SEED})
 write(OUT/'crossing_cluster_bootstrap.csv',result)
 report=['# Statistical Validation','', 'All calculations are read-only derivatives of the frozen Exp 1 master and Exp 4 crossing master. Bootstrap resampling is by `query_id` (10,000 repetitions; seed 20261001). Effects are reported as new/intervention minus baseline/reference.','', '## Retrieval-level paired bootstrap','', '| Comparison | Metric | Effect | 95% CI | Bootstrap p |','|---|---|---:|---:|---:|']
 for x in paired:report.append(f"| {x['comparison']} | {x['metric']} | {x['effect_size']:.4f} | [{x['ci95_low']:.4f}, {x['ci95_high']:.4f}] | {x['bootstrap_p_value']:.5f} |")
 report+=['','## Exact McNemar','', '| Comparison | Metric | both hit | both miss | base hit → new miss | base miss → new hit | exact p |','|---|---|---:|---:|---:|---:|---:|']
 for x in mc:report.append(f"| {x['comparison']} | {x['metric']} | {x['both_hit']} | {x['both_miss']} | {x['baseline_hit_to_new_miss']} | {x['baseline_miss_to_new_hit']} | {x['exact_mcnemar_p_value']:.5f} |")
 report+=['','## Crossing diagnostics','', 'Candidate-level estimates retain all candidates within each sampled query; query-balanced estimates first average within a nonempty query/group. Thus neither treats crossing candidates as independent observations. Effects are group 1 minus group 0, as stated in each row.','','| Comparison | Level | Metric | Effect | 95% CI | Bootstrap p |','|---|---|---|---:|---:|---:|']
 for x in result:report.append(f"| {x['comparison']} | {x['analysis_level']} | {x['metric']} | {x['effect_size']:.4f} | [{x['ci95_low']:.4f}, {x['ci95_high']:.4f}] | {x['bootstrap_p_value']:.5f} |")
 report+=['','- `paired_bootstrap.csv`: formal retrieval effects and percentile CIs.','- `mcnemar.csv`: exact paired hit/miss transitions.','- `crossing_cluster_bootstrap.csv`: cluster-bootstrap crossing effects, CIs, p-values, and candidate/query denominators.','','No scores, ranks, candidate pools, folds, PICs, or parameter choices were modified.']
 (OUT/'statistical_validation_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
 (OUT/'validation_manifest.json').write_text(json.dumps({'seed':SEED,'bootstrap_repetitions':B,'source_exp1_master_sha256':__import__('hashlib').sha256((EXP1/'exp1_master_candidate_results.csv.gz').read_bytes()).hexdigest(),'source_crossing_master_sha256':__import__('hashlib').sha256((EXP49/'exp4_crossing_master.csv.gz').read_bytes()).hexdigest()},indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
