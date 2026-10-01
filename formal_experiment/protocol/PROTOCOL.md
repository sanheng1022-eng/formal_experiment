# Formal experiment protocol — draft to freeze

**Status:** DRAFT — not yet executable or citable as a formal result.

## Research question

State the primary question, the target population, and the exact retrieval setting.

## Hypotheses and outcomes

| ID | Hypothesis | Primary metric | Decision rule |
|---|---|---|---|
| H1 | TODO | TODO | TODO |

Declare secondary and diagnostic metrics separately. Do not promote a diagnostic to a primary outcome after inspecting test results.

## Data and splits

- Raw dataset identity and SHA-256: TODO
- Train / validation / final-test rule: TODO
- Candidate-pool construction and seed(s): TODO
- Exclusions and missing-data policy: TODO

The final test set must be independent from the exploratory 40-query held-out set, or this limitation must be explicitly reported.

## Methods

- Baselines: TODO
- Proposed method: TODO
- Parameter search space and validation-only selection rule: TODO
- Fixed random seeds and number of replications: TODO

Exploratory parameters (`TopK=20`, `beta=0.5`, `lambda=0.5`) may be evaluated as pre-specified candidates, but must not be selected using the formal final-test set.

## Execution and reporting

- Every run saves its resolved config, input hashes, code revision, stdout/stderr, and result tables.
- A failed run is retained with a short failure note; reruns receive new run IDs.
- The final-test command is run only after configuration freeze and produces a dated final-result record.

## Freeze record

- Protocol version/hash: TODO
- Date frozen: TODO
- Person responsible: TODO
- Final-test authorization: TODO
