# LLM-assisted Hard-negative Audit — Handoff

## Scope and frozen inputs

- Review exactly the 494 records in `llm_audit_tasks.jsonl`, one line/task at a time.
- The sample is fixed: all 441 New overtakers in the 17 critical harmful queries, plus the globally highest-scoring 30 MultiPIC and 30 Calibration non-gold records, deduplicated by `(query_id, candidate_state_id)`.
- `audit_candidates.csv` and `audit_manifest.json` are read-only. Do not change sampling, benchmark gold labels, scores, ranks, PIC text, folds, parameters, or methods.
- This must be described in any paper/report as an **LLM-assisted label-quality audit**, not a human audit.

## Recommended execution protocol

1. Use one declared model/version for every task; record provider, model ID, release/version, date, decoding settings, and any seed.
2. Use temperature 0 (and top_p 1, if supported), structured JSON output, and the exact task-specific system prompt included in each JSONL record.
3. Preserve each unedited raw model response in a separate JSONL log keyed by `task_id`.
4. Validate that every response has the matching IDs, one permitted label, and a rationale of at most 80 words. Reject/retry malformed responses without changing task content.
5. For stronger reliability, obtain two independent model passes and send disagreements to a fixed third-pass adjudication prompt. If one pass is used, state this limitation prominently.
6. Copy only validated `audit_label` and reviewer/model provenance into the writable `audit_labels.csv`; keep rationales in a separate response log or add them to `review_notes` without editing other columns.
7. Run `summarize_hard_negative_audit.py` only after all 494 labels are populated and validated.

## Permitted labels

| Label | Decision rule |
|---|---|
| `clearly unaffected` | Candidate state is substantively unrelated to the update. |
| `plausibly affected / possible missing secondary gold` | The text supports a defensible direct or indirect impact on the candidate. |
| `uncertain` | Available text cannot support a reliable decision. |

Scores and ranks are retrieval context, not evidence of affectedness. The model should ground the rationale in concrete entities, APIs, behavior, constraints, configuration, or semantics found in `state_new`, candidate state, and PICs.

## Required response shape

```json
{
  "query_id": "exact input value",
  "candidate_state_id": "exact input value",
  "audit_label": "clearly unaffected | plausibly affected / possible missing secondary gold | uncertain",
  "rationale": "80 words or fewer, grounded in the supplied text",
  "evidence_fields": ["brief concrete evidence"],
  "confidence": "low | medium | high"
}
```

## Interpretation

Do not revise the benchmark after this audit. Report the three label proportions and distinguish possible missing-secondary-gold records from confirmed unrelated distractors. A large possible-missing-secondary-gold proportion warrants cautious interpretation of distractor amplification; predominance of clearly-unaffected records supports the formal failure analysis. LLM labels are evidence from a model-based audit and do not substitute for independent human/domain-expert adjudication.
