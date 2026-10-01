#!/usr/bin/env python3
"""Build canonical state pairs from paired M_old / M_new memories.

Functional recovery of the missing exploratory normalizer. The v2 prompt was
recovered verbatim from the frozen normalization_calls.jsonl cache. It does not
force literal contradiction: each state must be faithful to its source memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from deepseek_flash import DeepSeekFlash


PROMPT_VERSION = "paired-state-normalizer-v2"
SYSTEM_PROMPT = '''You construct canonical state pairs for an affected-state retrieval benchmark.

You receive one benchmark pair consisting of M_old and M_new. The source
benchmark has already indicated that the two memories are related. Your task is
to identify and canonicalize the most relevant user-related state expressed by
each side. The goal is dataset construction, NOT affectedness reasoning.

Requirements:
1. Produce exactly one state_old and one state_new when the pair is valid.
2. Both states must be faithful to their own source: state_old may contain only
   information supported by M_old, and state_new only information supported by M_new.
3. Always write canonical states in third person using "The user".
4. Each state must be a standalone atomic proposition expressing one primary
   user-specific fact, preference, goal, habit, relationship, possession,
   capability, constraint, plan, routine, location, or condition.
5. Preserve wants versus actions, plans versus current actions, obligations
   versus facts, temporary versus persistent scope, uncertainty, negation, and time.
6. Do NOT rewrite either state to make the pair contradictory; insert a causal
   or dependency relationship; infer consequences from the other memory; add
   unsupported facts; or replace a concrete state with a broad topic label.
7. Affectedness need not be a direct contradiction: valid states can be linked
   only through an implicit dependency.
8. Reject the pair if either side cannot be represented as a meaningful
   user-related state without substantial inference. Normally reject general
   world facts, unrelated people's events, environmental descriptions without a
   clear user relation, and text requiring the other memory to interpret it.

Return valid JSON only:
{
  "decision": "VALID" or "INVALID",
  "state_old": "canonical atomic state or null",
  "state_new": "canonical atomic state or null",
  "old_evidence_text": "verbatim supporting span from M_old or null",
  "new_evidence_text": "verbatim supporting span from M_new or null",
  "invalid_reason": "reason or null",
  "confidence": 0.0
}
'''


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    parsed = json.loads(text) if path.suffix == ".json" else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(parsed, list):
        raise ValueError("input must be a JSON array or JSONL")
    result, seen = [], set()
    for number, row in enumerate(parsed, 1):
        source = row.get("source", row) if isinstance(row, dict) else None
        uid = row.get("uid") if isinstance(row, dict) else None
        if not isinstance(uid, str) or not uid or uid in seen:
            raise ValueError(f"row {number}: uid must be unique and nonempty")
        if not isinstance(source, dict) or not all(isinstance(source.get(key), str) and source[key].strip() for key in ("M_old", "M_new")):
            raise ValueError(f"row {number}: M_old and M_new must be nonempty strings")
        seen.add(uid)
        result.append({"uid": uid, "source": {"M_old": source["M_old"].strip(), "M_new": source["M_new"].strip()}})
    return result


def validate(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("decision") not in {"VALID", "INVALID"}:
        raise ValueError("response decision must be VALID or INVALID")
    result = {"decision": value["decision"]}
    for key in ("state_old", "state_new", "old_evidence_text", "new_evidence_text", "invalid_reason"):
        item = value.get(key)
        if item is not None and not isinstance(item, str):
            raise ValueError(f"response {key} must be string or null")
        result[key] = item.strip() if isinstance(item, str) else None
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("response confidence must be in [0, 1]")
    result["confidence"] = confidence
    if result["decision"] == "VALID" and not all(result[key] for key in ("state_old", "state_new", "old_evidence_text", "new_evidence_text")):
        raise ValueError("VALID response requires both states and both evidence spans")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSONL output; refuses overwrite.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env.llm"))
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), default="low")
    parser.add_argument("--max-output-tokens", type=int, default=16_384)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.limit < 0:
        parser.error("output must not exist and --limit must be non-negative")
    rows = load_rows(args.input)
    rows = rows[:args.limit] if args.limit else rows
    if args.dry_run:
        print(json.dumps({"input": str(args.input), "rows": len(rows), "prompt_version": PROMPT_VERSION}, ensure_ascii=False, indent=2))
        return
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    output = []
    with DeepSeekFlash.from_env_file(args.env_file, model=args.model) as client:
        for row in rows:
            fingerprint = {"prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_hash, "input_sha256": digest(row["source"]), "model": args.model, "thinking": not args.disable_thinking, "reasoning_effort": args.reasoning_effort}
            cache = args.cache_dir / f"{row['uid']}.paired-state.json"
            cached = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() and not args.refresh_cache else None
            if cached and cached.get("status") == "success" and cached.get("fingerprint") == fingerprint:
                result, cache_hit = cached["result"], True
            else:
                response = client.ask_json(SYSTEM_PROMPT, json.dumps(row["source"], ensure_ascii=False, indent=2), thinking=not args.disable_thinking, reasoning_effort=args.reasoning_effort, max_output_tokens=args.max_output_tokens)
                result, cache_hit = validate(response), False
                cache.write_text(json.dumps({"status": "success", "fingerprint": fingerprint, "result": result}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            output.append({"schema_version": "paired_states_v2", "uid": row["uid"], "source": row["source"], **result, "status": "success", "cache_hit": cache_hit})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(output), "cache_dir": str(args.cache_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
