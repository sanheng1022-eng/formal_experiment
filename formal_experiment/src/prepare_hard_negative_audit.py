"""Build the fixed, read-only hard-negative audit packet from frozen formal artifacts."""
from __future__ import annotations
import csv, gzip, hashlib, json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/paired_states_v2/paired_states.json'
EXP1=ROOT/'runs/2026-10-01_experiment1_main_retrieval'
EXP49=ROOT/'runs/2026-10-01_experiment4_9_analysis'
OUT=ROOT/'hard_negative_audit'
ALLOWED=('clearly unaffected','plausibly affected / possible missing secondary gold','uncertain')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readgz(p):
 with gzip.open(p,'rt',encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 with open(p,'w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 if OUT.exists():raise FileExistsError(f'Refusing to overwrite existing audit packet: {OUT}')
 raw={r['uid']:r for r in json.loads(RAW.read_text(encoding='utf-8'))}
 master=readgz(EXP1/'exp1_master_candidate_results.csv.gz');cross=readgz(EXP49/'exp4_crossing_master.csv.gz')
 gold={r['query_id']:r for r in master if r['is_gold']=='True'}
 critical={q for q,r in gold.items() if int(r['raw_rank'])<=20 and int(r['multipic_rank'])>20}
 # Critical transitions are the New-overtaker crossing candidates within a critical query.
 chosen=defaultdict(set);crossinfo={(r['query_id'],r['candidate_state_id']):r for r in cross}
 for r in cross:
  if r['crossing_type']=='New overtaker' and r['query_id'] in critical:chosen[(r['query_id'],r['candidate_state_id'])].add('critical_harmful_transition')
 non=[r for r in master if r['is_gold']=='False']
 for tag,field in [('multipic_global_top30','multipic_score'),('calibration_global_top30','calibration_score')]:
  for r in sorted(non,key=lambda x:(-float(x[field]),x['query_id'],x['candidate_state_id']))[:30]:chosen[(r['query_id'],r['candidate_state_id'])].add(tag)
 lookup={(r['query_id'],r['candidate_state_id']):r for r in master}
 rows=[]
 for q,c in sorted(chosen):
  r=lookup[(q,c)];g=gold[q];x=crossinfo.get((q,c),{}) ; cand=raw[c];source=raw[q]
  rows.append({'query_id':q,'outer_fold':r['outer_fold'],'candidate_state_id':c,'gold_state_id':q,'selection_sources':' | '.join(sorted(chosen[(q,c)])),'is_critical_query':q in critical,'state_new':source['state_new'],'candidate_state_old':cand['state_old'],'gold_state_old':source['state_old'],'PIC_1':cand['possible_invalidation_conditions'][0],'PIC_2':cand['possible_invalidation_conditions'][1],'PIC_3':cand['possible_invalidation_conditions'][2],'PIC_4':cand['possible_invalidation_conditions'][3],'raw_score':r['raw_score'],'raw_rank':r['raw_rank'],'multipic_score':r['multipic_score'],'multipic_rank':r['multipic_rank'],'calibration_score':r['calibration_score'],'calibration_rank':r['calibration_rank'],'support_score':r['support_score'],'support_rank':r['support_rank'],'winner_pic_id':r['winner_pic_id'],'winner_pic_hubness':r['winner_pic_hubness'],'top1':r['top1'],'top2':r['top2'],'gap12':r['gap12'],'crossing_type':x.get('crossing_type','not_a_raw_to_multipic_crossing'),'raw_gold_rank':x.get('raw_gold_rank',g['raw_rank']),'multipic_gold_rank':x.get('multipic_gold_rank',g['multipic_rank']),'calibration_gold_rank':x.get('calibration_gold_rank',g['calibration_rank']),'support_gold_rank':x.get('support_gold_rank',g['support_rank']),'repaired_by_calibration':x.get('repaired_by_calibration','not_applicable'),'repaired_by_support':x.get('repaired_by_support','not_applicable')})
 OUT.mkdir();write(OUT/'audit_candidates.csv',rows)
 labels=[{'query_id':r['query_id'],'candidate_state_id':r['candidate_state_id'],'selection_sources':r['selection_sources'],'audit_label':'','label_options':' | '.join(ALLOWED),'reviewer_id':'','review_notes':''} for r in rows]
 write(OUT/'audit_labels.csv',labels)
 manifest={'selection_rule':'all New-overtaker candidates in critical harmful queries + global top 30 MultiPIC-score non-golds + global top 30 Calibration-score non-golds; deduplicate by query_id + candidate_state_id','critical_query_definition':'raw_gold_rank <= 20 AND multipic_gold_rank > 20','critical_query_count':len(critical),'critical_new_overtaker_records':sum('critical_harmful_transition' in x['selection_sources'] for x in rows),'multipic_top30_records':sum('multipic_global_top30' in x['selection_sources'] for x in rows),'calibration_top30_records':sum('calibration_global_top30' in x['selection_sources'] for x in rows),'deduplicated_audit_records':len(rows),'source_hashes':{'raw_pairs':sha(RAW),'exp1_master':sha(EXP1/'exp1_master_candidate_results.csv.gz'),'exp4_crossing_master':sha(EXP49/'exp4_crossing_master.csv.gz')}}
 (OUT/'audit_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 report=f'''# Hard-negative Label Audit\n\n## Fixed sampling packet\n\n- Critical harmful queries: {len(critical)}; their New-overtaker records: {manifest['critical_new_overtaker_records']}.\n- Global false-score additions: top 30 MultiPIC and top 30 Calibration non-gold query-candidate pairs.\n- After fixed-key deduplication, the packet contains **{len(rows)}** samples.\n- Selection is frozen in `audit_manifest.json`; no observation may alter it.\n\n## Human-label procedure\n\nComplete `audit_labels.csv` with exactly one label per row:\n\n- `clearly unaffected`\n- `plausibly affected / possible missing secondary gold`\n- `uncertain`\n\nThe corresponding `audit_candidates.csv` provides state_new, candidate and gold historical states, all four frozen PICs, scores/ranks, and crossing/critical metadata. Do not edit benchmark labels or result files.\n\n## Status\n\n**AWAITING HUMAN LABELS.** Counts, proportions, and representative labelled cases cannot be truthfully reported until an authorized human reviewer supplies labels.\n'''
 (OUT/'hard_negative_audit_report.md').write_text(report,encoding='utf-8')
 print(OUT)
if __name__=='__main__':main()
