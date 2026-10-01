"""Run the frozen hard-negative label audit through the workspace GPT-6 relay."""
from __future__ import annotations
import csv, hashlib, json, os, tempfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from gpt6_relay import GPT6Relay

ROOT=Path(__file__).resolve().parents[1]; AUDIT=ROOT/'hard_negative_audit'
ALLOWED={'clearly_unaffected','plausibly_affected','uncertain'}
REQUEST_TIMEOUT_SECONDS=60
SYSTEM_PROMPT=r'''You are an independent semantic auditor for an affected-state localization benchmark.

Your task is to determine whether a non-gold candidate historical state could genuinely be affected, invalidated, contradicted, or require revision because of the new state update.

You are performing **label auditing**, not retrieval evaluation.

## Input

For each case you may receive:

- `state_new`: the new update / query.
- `gold_state_old`: the benchmark's labeled affected historical state, provided only as contextual reference.
- `candidate_state_old`: a non-gold historical state being audited.
- `candidate_PICs`: hypothetical prospective invalidation views generated only from `candidate_state_old`.

PICs describe possible future developments that *could* affect the candidate. They are auxiliary semantic cues only.

**A PIC matching `state_new` is not by itself sufficient evidence that the candidate is truly affected.**

## Core question

Judge:

> If `state_new` becomes true, would the factual validity, applicability, interpretation, or current status of `candidate_state_old` reasonably need to change?

Focus on the semantic relationship between `state_new` and `candidate_state_old`.

A candidate may be affected even when they share few surface words, but there must be a defensible semantic dependency.

Examples of meaningful dependency include:

- direct contradiction or replacement;
- update of the same underlying entity, event, plan, preference, status, or relationship;
- a new fact that invalidates an assumption required by the candidate;
- a development that makes the candidate outdated or no longer applicable;
- an implicit but concrete causal or relational connection.

Do **not** mark a candidate as affected merely because:

- it is topically similar;
- it mentions the same broad domain;
- a PIC happens to resemble `state_new`;
- the two statements could coexist without changing the candidate;
- the relationship requires substantial unsupported speculation.

## Labels

Return exactly one label:

### `clearly_unaffected`

`state_new` can reasonably coexist with `candidate_state_old` and does not require the candidate to be revised, invalidated, or reinterpreted.

### `plausibly_affected`

There is a concrete and defensible semantic relation by which `state_new` could make `candidate_state_old` outdated, invalid, contradicted, or require revision.

Use this label even if the benchmark does not currently mark the candidate as gold.

### `uncertain`

There is a possible relation, but the supplied information is insufficient to determine whether the candidate is genuinely affected without making important assumptions.

Do not force ambiguous cases into either extreme.

## Important rules

1. Judge each candidate independently.
2. Do not assume the benchmark gold annotation is complete.
3. Do not assume a non-gold candidate is unaffected simply because it is labeled non-gold.
4. Do not treat PICs as ground truth.
5. Do not infer facts that are not contained in the supplied text.
6. Prefer semantic necessity over topical similarity.
7. Ignore retrieval scores, ranks, method names, or whether a method retrieved the candidate.
8. The question is not whether `candidate_state_old` is similar to `state_new`; the question is whether the new update actually changes the validity or applicability of that historical state.

## Output

Return valid JSON only:
```json
{
  "label": "clearly_unaffected | plausibly_affected | uncertain",
  "relation_type": "contradiction | replacement | status_update | dependency | implicit_relation | topical_only | none | unclear",
  "reason": "Brief explanation based only on the supplied state_new and candidate_state_old.",
  "pic_consistency": "supports_relation | does_not_support_relation | mixed | not_needed",
  "confidence": "high | medium | low"
}
```

Keep `reason` concise.

The final label must primarily follow the relationship between `state_new` and `candidate_state_old`; PICs may help interpret the candidate's prospective semantics but must never override the underlying state relationship.'''
def atomic_text(path:Path,text:str):
 fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent);os.close(fd)
 try:
  Path(tmp).write_text(text,encoding='utf-8');os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def prompt_hash(system,user):
 return hashlib.sha256((system+'\n\n'+user).encode('utf-8')).hexdigest()
def main():
 out=AUDIT/'gpt6_audit_responses.jsonl'; labels=AUDIT/'audit_labels.csv'
 if out.exists():raise FileExistsError(f'Refusing to overwrite raw LLM responses: {out}')
 rows=list(csv.DictReader(open(AUDIT/'audit_candidates.csv',encoding='utf-8',newline='')))
 row_by_key={(r['query_id'],r['candidate_state_id']):r for r in rows}
 def one(n,r):
   payload={'state_new':r['state_new'],'gold_state_old':r['gold_state_old'],'candidate_state_old':r['candidate_state_old'],'candidate_PICs':[r[f'PIC_{j}'] for j in range(1,5)]}
   os.environ['GPT6_RELAY_TIMEOUT']=str(REQUEST_TIMEOUT_SECONDS)
   with GPT6Relay.from_env_file(ROOT.parent/'.env.llm') as relay:
    for attempt in range(3):
     started=time.monotonic();print(f'start hna-{n:04d} attempt {attempt + 1}/3',flush=True)
     try:
      result=relay.ask(SYSTEM_PROMPT,json.dumps(payload,ensure_ascii=False),temperature=0,json_object=True,max_output_tokens=500);answer=json.loads(result.content)
      valid=answer.get('label') in ALLOWED and answer.get('relation_type') in {'contradiction','replacement','status_update','dependency','implicit_relation','topical_only','none','unclear'} and answer.get('pic_consistency') in {'supports_relation','does_not_support_relation','mixed','not_needed'} and answer.get('confidence') in {'high','medium','low'} and isinstance(answer.get('reason'),str) and len(answer['reason'].split())<=80
      if not valid:raise ValueError('schema validation failed')
      print(f'done hna-{n:04d} elapsed={time.monotonic()-started:.1f}s',flush=True);return n,r,payload,result,answer
     except Exception as exc:
      print(f'retry hna-{n:04d} attempt {attempt + 1}/3 after {time.monotonic()-started:.1f}s: {type(exc).__name__}',flush=True)
      if attempt==2:raise
      time.sleep(2**attempt)
 checkpoint=AUDIT/'gpt6_audit_responses.partial.jsonl';responses=[]
 if checkpoint.exists():
  for line in checkpoint.read_text(encoding='utf-8').splitlines():
   try:
    item=json.loads(line)
    if item.get('task_id') and item.get('response'):responses.append(item)
   except json.JSONDecodeError:pass
  atomic_text(checkpoint,''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in sorted({x['task_id']:x for x in responses}.values(),key=lambda x:x['task_id'])))
 completed={x['task_id'] for x in responses}
 starting_done=len(completed)
 next_task=next((f'hna-{n:04d}' for n in range(1,len(rows)+1) if f'hna-{n:04d}' not in completed),'none')
 print(f'resume checkpoint: {starting_done}/{len(rows)} complete; next={next_task}',flush=True)
 # The configured relay is serial-only. Retain the executor/checkpoint structure
 # for deterministic one-at-a-time execution and crash-visible progress.
 with open(checkpoint,'a',encoding='utf-8') as partial, ThreadPoolExecutor(max_workers=1) as pool:
  futures=[pool.submit(one,n,r) for n,r in enumerate(rows,1) if f'hna-{n:04d}' not in completed]
  for done,future in enumerate(as_completed(futures),1):
   n,r,payload,result,answer=future.result()
   user_prompt=json.dumps(payload,ensure_ascii=False)
   item={'task_id':f'hna-{n:04d}','query_id':r['query_id'],'candidate_state_id':r['candidate_state_id'],'selection_sources':r['selection_sources'],'system_prompt':SYSTEM_PROMPT,'user_prompt':user_prompt,'prompt_sha256':prompt_hash(SYSTEM_PROMPT,user_prompt),'input':payload,'response':answer,'raw_response':result.content,'model':result.model,'finish_reason':result.finish_reason,'prompt_tokens':result.prompt_tokens,'completion_tokens':result.completion_tokens,'total_tokens':result.total_tokens,'temperature':0,'timestamp_utc':datetime.now(timezone.utc).isoformat()}
   partial.write(json.dumps(item,ensure_ascii=False)+'\n');partial.flush();responses.append(item)
   print(f'{starting_done + done}/{len(rows)}',flush=True)
 responses=sorted({x['task_id']:x for x in responses}.values(),key=lambda x:x['task_id'])
 atomic_text(out,''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in responses))
 labelrows=[]
 for x in responses:
  a=x['response'];labelrows.append({'query_id':x['query_id'],'candidate_state_id':x['candidate_state_id'],'selection_sources':x['selection_sources'],'audit_label':a['label'],'relation_type':a['relation_type'],'pic_consistency':a['pic_consistency'],'confidence':a['confidence'],'reviewer_id':f"GPT6Relay:{x['model']}",'review_notes':a['reason']})
 fd,tmp=tempfile.mkstemp(prefix=labels.name+'.',suffix='.tmp',dir=labels.parent);os.close(fd)
 try:
  with open(tmp,'w',encoding='utf-8',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(labelrows[0]));w.writeheader();w.writerows(labelrows)
  os.replace(tmp,labels)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
if __name__=='__main__':main()
