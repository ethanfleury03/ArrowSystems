#!/usr/bin/env python3
"""
verify_raw_response_schema.py

Verifies that ticket_judgements.raw_response_json matches the contract you expect,
and that effective cache-eligible tickets are safe to index as "solution cards".

Run:
  python verify_raw_response_schema.py --db data/tickets.db
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


VALID_OUTCOMES = {
    "resolved_remotely_actionable",
    "needs_onsite",
    "needs_replacement",
    "denied",
    "unclear",
    "no_fix_provided",
    "workaround_only",
}

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){1}\d{3}[-.\s]?\d{4}\b")


def redact_pii(s: str) -> str:
    s = EMAIL_RE.sub("[REDACTED_EMAIL]", s)
    s = PHONE_RE.sub("[REDACTED_PHONE]", s)
    return s


def safe_trunc(x: Any, n: int = 240) -> str:
    try:
        s = json.dumps(x, ensure_ascii=False)
    except Exception:
        s = str(x)
    s = redact_pii(s)
    return s if len(s) <= n else s[: n - 3] + "..."


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


EFFECTIVE_SQL = """
SELECT DISTINCT j.ticket_id
FROM ticket_judgements j
LEFT JOIN ticket_manual_reviews m ON j.ticket_id = m.ticket_id
WHERE (
    (j.review_status = 'approved' OR (j.review_status IS NULL AND j.cache_eligible = 1))
    OR (m.manual_status = 'approved')
)
AND (m.manual_status IS NULL OR m.manual_status != 'rejected')
ORDER BY j.ticket_id
"""


AUTO_SAMPLE_SQL = """
SELECT ticket_id
FROM ticket_judgements
WHERE review_status = 'approved'
ORDER BY ticket_id
LIMIT ?
"""

MANUAL_SAMPLE_SQL = """
SELECT j.ticket_id
FROM ticket_judgements j
JOIN ticket_manual_reviews m ON j.ticket_id = m.ticket_id
WHERE m.manual_status = 'approved'
ORDER BY j.ticket_id
LIMIT ?
"""


TICKET_ROW_SQL = """
SELECT
  j.ticket_id,
  j.cache_eligible,
  j.confidence,
  j.problem AS problem_col,
  j.resolution_steps_json AS resolution_steps_json_col,
  j.confirmation AS confirmation_col,
  j.evidence_json AS evidence_json_col,
  j.blockers_json AS blockers_json_col,
  j.model,
  j.prompt_version,
  j.judged_at,
  j.raw_response_json,
  j.review_status,
  j.review_reason,
  j.review_reasons_json,
  j.reviewed_at,
  m.manual_status,
  m.manual_reason,
  m.manual_confirmation_quote,
  m.reviewer,
  m.reviewed_at AS manual_reviewed_at
FROM ticket_judgements j
LEFT JOIN ticket_manual_reviews m ON j.ticket_id = m.ticket_id
WHERE j.ticket_id = ?
"""


def json_load_maybe(s: Optional[str]) -> Any:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_problem(raw: Dict[str, Any]) -> Optional[str]:
    p = raw.get("problem")
    if isinstance(p, str):
        return p.strip() or None
    if isinstance(p, dict):
        for k in ("summary", "text", "problem", "description"):
            v = p.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def extract_steps(raw: Dict[str, Any]) -> List[str]:
    # Most expected: raw["resolution"]["steps"]
    res = raw.get("resolution")
    if isinstance(res, dict):
        steps = res.get("steps")
        if isinstance(steps, list):
            out = []
            for s in steps:
                if isinstance(s, str) and s.strip():
                    out.append(s.strip())
            return out
    # Fallback: other possible shapes
    for k in ("resolution_steps", "steps"):
        steps = raw.get(k)
        if isinstance(steps, list):
            out = []
            for s in steps:
                if isinstance(s, str) and s.strip():
                    out.append(s.strip())
            return out
    return []


def extract_confirmation(raw: Dict[str, Any]) -> Tuple[Optional[bool], Optional[Dict[str, Any]]]:
    c = raw.get("confirmation")
    if isinstance(c, dict):
        confirmed = c.get("confirmed")
        evidence = c.get("evidence")
        return (bool(confirmed) if confirmed is not None else None, evidence if isinstance(evidence, dict) else None)
    return (None, None)


def normalize_str(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/tickets.db")
    ap.add_argument("--sample-auto", type=int, default=5)
    ap.add_argument("--sample-manual", type=int, default=5)
    ap.add_argument("--max-fail-ids", type=int, default=10)
    ap.add_argument("--json-out", default=None, help="Optional: write full report JSON to file")
    args = ap.parse_args()

    conn = connect(args.db)
    cur = conn.cursor()

    # Effective set
    effective_ids = [r["ticket_id"] for r in cur.execute(EFFECTIVE_SQL).fetchall()]

    # Samples
    auto_ids = [r["ticket_id"] for r in cur.execute(AUTO_SAMPLE_SQL, (args.sample_auto,)).fetchall()]
    manual_ids = [r["ticket_id"] for r in cur.execute(MANUAL_SAMPLE_SQL, (args.sample_manual,)).fetchall()]

    sample_ids = []
    for tid in auto_ids + manual_ids:
        if tid not in sample_ids:
            sample_ids.append(tid)

    failures: Dict[str, List[str]] = {
        "raw_json_parse_failed": [],
        "problem_missing": [],
        "steps_missing": [],
        "confirmation_not_true": [],
        "confirmation_quote_missing": [],
        "confidence_invalid": [],
        "outcome_invalid": [],
        "col_raw_problem_mismatch": [],
        "col_raw_steps_mismatch": [],
        "col_raw_evidence_mismatch": [],
    }

    per_ticket_reports: List[Dict[str, Any]] = []

    def check_ticket(ticket_id: str, collect_report: bool = False) -> None:
        row = cur.execute(TICKET_ROW_SQL, (ticket_id,)).fetchone()
        if not row:
            return

        raw_text = row["raw_response_json"]
        raw = None
        try:
            raw = json.loads(raw_text)
            if not isinstance(raw, dict):
                raise ValueError("raw_response_json not an object")
        except Exception:
            failures["raw_json_parse_failed"].append(ticket_id)
            return

        outcome = raw.get("outcome")
        confidence = raw.get("confidence")

        problem = extract_problem(raw)
        steps = extract_steps(raw)
        confirmed, evidence = extract_confirmation(raw)
        quote = None
        if isinstance(evidence, dict):
            q = evidence.get("quote")
            if isinstance(q, str) and q.strip():
                quote = q.strip()

        # Checks
        if not problem:
            failures["problem_missing"].append(ticket_id)
        if not steps:
            failures["steps_missing"].append(ticket_id)

        if confirmed is not True:
            failures["confirmation_not_true"].append(ticket_id)
        if confirmed is True and not quote:
            failures["confirmation_quote_missing"].append(ticket_id)

        if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
            failures["confidence_invalid"].append(ticket_id)

        if outcome not in VALID_OUTCOMES:
            failures["outcome_invalid"].append(ticket_id)

        # Column vs raw consistency (best-effort; tolerate formatting differences)
        problem_col = row["problem_col"]
        if isinstance(problem_col, str) and problem and normalize_str(problem_col) and normalize_str(problem_col) != normalize_str(problem):
            failures["col_raw_problem_mismatch"].append(ticket_id)

        steps_col = json_load_maybe(row["resolution_steps_json_col"])
        if isinstance(steps_col, list) and steps:
            steps_col_norm = [normalize_str(s) for s in steps_col if isinstance(s, str)]
            steps_norm = [normalize_str(s) for s in steps]
            if steps_col_norm and steps_norm and steps_col_norm != steps_norm:
                failures["col_raw_steps_mismatch"].append(ticket_id)

        ev_col = json_load_maybe(row["evidence_json_col"])
        if isinstance(ev_col, dict) and isinstance(evidence, dict):
            # Only compare quote if present on both sides
            evq = ev_col.get("quote")
            if isinstance(evq, str) and quote and normalize_str(evq) != normalize_str(quote):
                failures["col_raw_evidence_mismatch"].append(ticket_id)

        if collect_report:
            per_ticket_reports.append({
                "ticket_id": ticket_id,
                "review_status": row["review_status"],
                "review_reason": row["review_reason"],
                "manual_status": row["manual_status"],
                "model": row["model"],
                "prompt_version": row["prompt_version"],
                "judged_at": row["judged_at"],
                "raw_top_level_keys": list(raw.keys()),
                "raw_outcome": outcome,
                "raw_confidence": confidence,
                "raw_problem_extracted": safe_trunc(problem),
                "raw_steps_extracted": steps[:8],
                "raw_confirmation_confirmed": confirmed,
                "raw_confirmation_evidence": safe_trunc(evidence),
                "col_problem": safe_trunc(problem_col),
                "col_resolution_steps_json": safe_trunc(json_load_maybe(row["resolution_steps_json_col"])),
                "col_evidence_json": safe_trunc(json_load_maybe(row["evidence_json_col"])),
            })

    # Run checks over ALL effective IDs
    for tid in effective_ids:
        check_ticket(tid, collect_report=False)

    # Collect sample reports
    for tid in sample_ids:
        check_ticket(tid, collect_report=True)

    # Print report
    print("=" * 90)
    print("Raw Response Schema Verification")
    print("=" * 90)
    print(f"DB: {args.db}")
    print(f"Effective cache-eligible tickets: {len(effective_ids)}")
    print(f"Sample auto-approved: {len(auto_ids)} | sample manual-approved: {len(manual_ids)}")
    print()

    print("Sample Tickets (schema + key fields)")
    print("-" * 90)
    for r in per_ticket_reports:
        print(f"ticket_id={r['ticket_id']}  review_status={r['review_status']}  manual_status={r['manual_status']}")
        print(f"  keys={r['raw_top_level_keys']}")
        print(f"  outcome={r['raw_outcome']}  confidence={r['raw_confidence']}")
        print(f"  problem={r['raw_problem_extracted']}")
        print(f"  steps={r['raw_steps_extracted']}")
        print(f"  confirmation.confirmed={r['raw_confirmation_confirmed']}")
        print(f"  confirmation.evidence={r['raw_confirmation_evidence']}")
        print()

    print("Consistency Checks (effective cache-eligible set)")
    print("-" * 90)
    summary = {}
    for k, ids in failures.items():
        summary[k] = len(set(ids))
        if summary[k] == 0:
            continue
        uniq = sorted(set(ids))[: args.max_fail_ids]
        print(f"{k}: {summary[k]}  examples={uniq}")

    if all(v == 0 for v in summary.values()):
        print("All checks PASSED for effective cache-eligible set.")
    else:
        print("Some checks FAILED. Review failures above before integrating.")

    report_obj = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "db": args.db,
        "effective_cache_eligible_count": len(effective_ids),
        "sample_ids": sample_ids,
        "failures": {k: sorted(set(v)) for k, v in failures.items()},
        "sample_reports": per_ticket_reports,
    }

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report_obj, f, ensure_ascii=False, indent=2)
        print()
        print(f"Wrote JSON report: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
