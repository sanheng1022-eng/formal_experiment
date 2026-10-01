"""Export the frozen hard-negative packet as one self-contained LLM task per row."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; AUDIT=ROOT/'hard_negative_audit'
SYSTEM='''You are an independent label-quality reviewer for an affected-state retrieval benchmark. Decide whether the candidate historical state is semantically affected by the future update in state_new, even though it is not the benchmark's designated gold state. Do not infer affectedness from retrieval score alone. Compare the update's concrete entities, behaviors, constraints, APIs, configuration, and semantics against candidate_state_old and its four PICs. The supplied gold state is context only; do not assume that a candidate must be unaffected merely because a different gold exists. Return exactly one JSON object and no markdown. audit_label must be exactly one of: clearly unaffected; plausibly affected / possible missing secondary gold; uncertain. Use clearly unaffected only if the candidate is substantively unrelated. Use plausibly affected / possible missing secondary gold when a defensible direct or indirect impact exists. Use uncertain when the supplied text cannot support a reliable semantic decision. rationale must be <=80 words and identify concrete textual evidence. Do not propose changes to gold labels or retrieval methods.'''
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 candidates=list(csv.DictReader(open(AUDIT/'audit_candidates.csv',encoding='utf-8',newline='')))
 out=AUDIT/'llm_audit_tasks.jsonl'
 if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
 with open(out,'w',encoding='utf-8') as f:
  for n,r in enumerate(candidates,1):
   evidence={k:r[k] for k in ('query_id','candidate_state_id','gold_state_id','selection_sources','is_critical_query','state_new','candidate_state_old','gold_state_old','PIC_1','PIC_2','PIC_3','PIC_4','raw_score','multipic_score','calibration_score','raw_rank','multipic_rank','calibration_rank','crossing_type','raw_gold_rank','multipic_gold_rank','calibration_gold_rank','repaired_by_calibration')}
   schema={'query_id':r['query_id'],'candidate_state_id':r['candidate_state_id'],'audit_label':'one allowed label','rationale':'<=80 words','evidence_fields':['concrete semantic overlap or lack thereof'],'confidence':'low | medium | high'}
   task={'task_id':f'hna-{n:04d}','system_prompt':SYSTEM,'input':evidence,'required_json_schema':schema,'frozen_packet_sha256':sha(AUDIT/'audit_candidates.csv')}
   f.write(json.dumps(task,ensure_ascii=False,separators=(',',':'))+'\n')
 print(out)
if __name__=='__main__':main()
