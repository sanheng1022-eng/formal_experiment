"""Validate completed human labels and render the final hard-negative audit summary."""
from __future__ import annotations
import csv
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'hard_negative_audit'
ALLOWED=('clearly_unaffected','plausibly_affected','uncertain')
DISPLAY={'clearly_unaffected':'clearly unaffected','plausibly_affected':'plausibly affected / possible missing secondary gold','uncertain':'uncertain'}
def read(name):
 with open(OUT/name,encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def main():
 candidates={(r['query_id'],r['candidate_state_id']):r for r in read('audit_candidates.csv')};labels=read('audit_labels.csv')
 if len(labels)!=len(candidates) or {(r['query_id'],r['candidate_state_id']) for r in labels}!=set(candidates):raise ValueError('Label sheet keys do not exactly match the frozen audit packet')
 invalid=[r for r in labels if r['audit_label'] not in ALLOWED]
 if invalid:raise ValueError(f'All {len(candidates)} rows require one permitted human label; invalid/blank rows: {len(invalid)}')
 counts=Counter(r['audit_label'] for r in labels);total=len(labels)
 lines=['# LLM-assisted Hard-negative Label Audit — Completed Summary','',f'LLM-reviewed audit samples: **{total}**. The frozen sample definition and source hashes are in `audit_manifest.json`.','', '| Label | Count | Proportion |','|---|---:|---:|']
 for label in ALLOWED:lines.append(f'| {DISPLAY[label]} | {counts[label]} | {counts[label]/total:.1%} |')
 lines+=['','## Representative labelled records','']
 for label in ALLOWED:
  lines+=['### '+DISPLAY[label],'']
  for x in [r for r in labels if r['audit_label']==label][:3]:
   c=candidates[(x['query_id'],x['candidate_state_id'])]
   lines.append(f"- `{c['query_id']}` → `{c['candidate_state_id']}`; sources: {c['selection_sources']}; reviewer note: {x['review_notes'] or '(none)'}. ")
  lines.append('')
 lines+=['## Interpretation constraint','', 'If the plausibly_affected category is substantial, interpret distractor-amplification claims conservatively because some hard negatives may not be truly unaffected. If clearly_unaffected cases dominate, the audit supplies model-based label-quality support for the failure analysis. This audit never changes benchmark gold labels or formal retrieval results.']
 (OUT/'hard_negative_audit_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(OUT/'hard_negative_audit_report.md')
if __name__=='__main__':main()
