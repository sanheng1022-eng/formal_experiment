# Formal experiment workspace

This is a clean workspace for the formal study. Do not modify or execute code from `../archive/` in place. Port only reviewed, minimal components into `src/`, and document the provenance in the relevant run record.

## Start here

1. Complete and freeze `protocol/PROTOCOL.md` before creating a final test result.
2. Put immutable source data in `data/raw/`; derive all transformed data into `data/derived/`.
3. Store each run's fully resolved configuration and outputs in a new timestamped directory under `runs/`.
4. Record every transferred exploratory component and its archived SHA-256 in that run's `PROVENANCE.md`.

`configs/` is the only permitted source of tunable configuration. Test data must not influence a configuration change.
