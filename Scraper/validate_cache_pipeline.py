#!/usr/bin/env python3
"""
DB-wide validator for cache eligibility pipeline correctness.

Scans ticket_judgements and validates:
- review_status/cache_eligible consistency
- Approved tickets meet approval criteria
- Hard-blocked tickets have matching trigger phrases in transcript
- Confirmation evidence is grounded

Example usage:
  python Scraper/validate_cache_pipeline.py
  python Scraper/validate_cache_pipeline.py --fix-inconsistencies
"""

import argparse
import json
import sys
from typing import Dict, List, Set

import db


def validate_review_status_cache_eligible_consistency(conn) -> Dict[str, List[str]]:
    """
    Find tickets where review_status and cache_eligible are inconsistent.
    
    Returns:
        Dict with keys: "approved_but_not_eligible", "not_approved_but_eligible"
    """
    cursor = conn.cursor()
    
    # Approved but cache_eligible != 1
    cursor.execute("""
        SELECT ticket_id
        FROM ticket_judgements
        WHERE review_status = 'approved' AND cache_eligible != 1
    """)
    approved_but_not_eligible = [row["ticket_id"] for row in cursor.fetchall()]
    
    # Not approved but cache_eligible == 1
    cursor.execute("""
        SELECT ticket_id
        FROM ticket_judgements
        WHERE review_status != 'approved' AND cache_eligible = 1
    """)
    not_approved_but_eligible = [row["ticket_id"] for row in cursor.fetchall()]
    
    return {
        "approved_but_not_eligible": approved_but_not_eligible,
        "not_approved_but_eligible": not_approved_but_eligible
    }


def validate_approved_criteria(conn, approve_min_confidence: float = 0.90, require_requester_confirmation: bool = True) -> Dict[str, List[Dict]]:
    """
    Validate that approved tickets meet all approval criteria.
    
    Returns:
        Dict with violation types -> list of ticket dicts with violation details
    """
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            ticket_id,
            review_status,
            cache_eligible,
            confidence,
            raw_response_json,
            model
        FROM ticket_judgements
        WHERE review_status = 'approved'
    """)
    
    violations = {
        "wrong_outcome": [],
        "low_confidence": [],
        "missing_confirmation": [],
        "missing_steps": []
    }
    
    for row in cursor.fetchall():
        ticket_id = row["ticket_id"]
        confidence = row["confidence"]
        raw_response_json = dict(row).get("raw_response_json", "{}")
        model = dict(row).get("model", "")
        
        try:
            raw_json = json.loads(raw_response_json)
            outcome = raw_json.get("outcome", "unclear")
            
            # Extract confirmation
            confirmation_obj = raw_json.get("confirmation", {})
            if isinstance(confirmation_obj, dict):
                confirmation_confirmed = confirmation_obj.get("confirmed", False)
            else:
                confirmation_confirmed = bool(confirmation_obj)
            
            # Extract steps
            resolution_obj = raw_json.get("resolution", {})
            if isinstance(resolution_obj, dict):
                resolution_steps = resolution_obj.get("steps", [])
            else:
                resolution_steps = raw_json.get("resolution_steps", [])
            
            if not isinstance(resolution_steps, list):
                resolution_steps = []
            
            # Check violations
            if outcome != "resolved_remotely_actionable":
                violations["wrong_outcome"].append({
                    "ticket_id": ticket_id,
                    "outcome": outcome,
                    "model": model
                })
            
            if confidence < approve_min_confidence:
                violations["low_confidence"].append({
                    "ticket_id": ticket_id,
                    "confidence": confidence,
                    "threshold": approve_min_confidence
                })
            
            if require_requester_confirmation and not confirmation_confirmed:
                violations["missing_confirmation"].append({
                    "ticket_id": ticket_id,
                    "confirmed": confirmation_confirmed
                })
            
            if outcome == "resolved_remotely_actionable" and len(resolution_steps) < 1:
                violations["missing_steps"].append({
                    "ticket_id": ticket_id,
                    "steps_count": len(resolution_steps)
                })
        
        except (json.JSONDecodeError, TypeError, AttributeError):
            violations["wrong_outcome"].append({
                "ticket_id": ticket_id,
                "outcome": "parse_error",
                "error": "Invalid raw_response_json"
            })
    
    return violations


def validate_hard_block_phrases(conn) -> Dict[str, List[Dict]]:
    """
    Validate that hard-blocked tickets have matching trigger phrases in their transcript.
    
    Returns:
        Dict with "missing_phrase" -> list of ticket dicts
    """
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            j.ticket_id,
            j.model,
            j.raw_response_json,
            j.blockers_json
        FROM ticket_judgements j
        WHERE j.model = 'hard_block'
    """)
    
    violations = {
        "missing_phrase": []
    }
    
    for row in cursor.fetchall():
        ticket_id = row["ticket_id"]
        raw_response_json = dict(row).get("raw_response_json", "{}")
        blockers_json = dict(row).get("blockers_json", "[]")
        
        try:
            raw_json = json.loads(raw_response_json)
            outcome = raw_json.get("outcome", "unclear")
            
            blockers = json.loads(blockers_json) if blockers_json else []
            
            # Get conversation transcript
            conversation = db.get_ticket_detail_json(conn, ticket_id)
            if not conversation:
                continue
            
            # Build transcript text (same as hard_block uses)
            text_parts = []
            request = conversation.get("request", {})
            subject = request.get("subject", "")
            if subject:
                text_parts.append(subject.lower())
            
            messages = conversation.get("messages", [])
            if not isinstance(messages, list):
                messages = []
            
            messages.sort(key=lambda m: m.get("created_at", ""))
            
            for msg in messages:
                text = msg.get("text", "").strip()
                if text:
                    text_parts.append(text.lower())
            
            full_text = " ".join(text_parts)
            
            # Check if blocker phrase appears in transcript
            # Extract outcome category from blockers
            blocker_text = " ".join(blockers).lower()
            expected_phrases = {
                "denied": ["denied", "rejected", "warranty", "claim"],
                "needs_onsite": ["onsite", "on site", "on-site", "site visit", "field service", "dispatch"],
                "needs_replacement": ["rma", "replacement", "send back", "requires return"],
                "no_fix_provided": ["no fix", "unable to reproduce", "cannot reproduce"],
                "workaround_only": ["workaround"]
            }
            
            if outcome in expected_phrases:
                phrases_to_check = expected_phrases[outcome]
                found = any(phrase in full_text for phrase in phrases_to_check)
                
                if not found:
                    violations["missing_phrase"].append({
                        "ticket_id": ticket_id,
                        "outcome": outcome,
                        "blockers": blockers[:2]  # First 2 blockers
                    })
        
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    
    return violations


def validate_confirmation_evidence(conn, require_requester_confirmation: bool = True) -> Dict[str, List[Dict]]:
    """
    Validate that confirmed=True tickets have grounded evidence from requester-authored messages.
    
    Returns:
        Dict with violation types -> list of ticket dicts
    """
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            j.ticket_id,
            j.raw_response_json
        FROM ticket_judgements j
        WHERE json_extract(j.raw_response_json, '$.confirmation.confirmed') = 1
    """)
    
    violations = {
        "missing_evidence": [],
        "quote_not_found": [],
        "wrong_role": [],
        "not_requester_authored": []
    }
    
    for row in cursor.fetchall():
        ticket_id = row["ticket_id"]
        raw_response_json = dict(row).get("raw_response_json", "{}")
        
        try:
            raw_json = json.loads(raw_response_json)
            confirmation_obj = raw_json.get("confirmation", {})
            
            if not isinstance(confirmation_obj, dict):
                continue
            
            confirmation_confirmed = confirmation_obj.get("confirmed", False)
            if not confirmation_confirmed:
                continue
            
            confirmation_evidence = confirmation_obj.get("evidence", {})
            if not confirmation_evidence:
                violations["missing_evidence"].append({
                    "ticket_id": ticket_id,
                    "reason": "evidence object missing"
                })
                continue
            
            evidence_quote = confirmation_evidence.get("quote", "")
            evidence_role = confirmation_evidence.get("author_role", "")
            evidence_author_id = confirmation_evidence.get("author_id", "")
            message_index = confirmation_evidence.get("message_index")
            
            if not evidence_quote:
                violations["missing_evidence"].append({
                    "ticket_id": ticket_id,
                    "reason": "evidence.quote missing"
                })
                continue
            
            # Check if quote exists in transcript
            conversation = db.get_ticket_detail_json(conn, ticket_id)
            if not conversation:
                violations["quote_not_found"].append({
                    "ticket_id": ticket_id,
                    "reason": "conversation not found"
                })
                continue
            
            # Build transcript text
            text_parts = []
            messages = conversation.get("messages", [])
            if not isinstance(messages, list):
                messages = []
            
            messages.sort(key=lambda m: m.get("created_at", ""))
            
            for msg in messages:
                text = msg.get("text", "").strip()
                if text:
                    text_parts.append(text.lower())
            
            full_text = " ".join(text_parts)
            
            # Check if quote appears (case-insensitive substring)
            if evidence_quote.lower() not in full_text:
                violations["quote_not_found"].append({
                    "ticket_id": ticket_id,
                    "quote_preview": evidence_quote[:50]
                })
            
            # Check role and author_id if required
            if require_requester_confirmation:
                outcome = raw_json.get("outcome", "")
                if outcome == "resolved_remotely_actionable":
                    # Identify requester author IDs
                    # Import here to avoid circular dependency
                    import sys
                    import os
                    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                    from judge_ticket_cache_eligibility import get_requester_author_ids
                    requester_author_ids = get_requester_author_ids(conversation, messages)
                    
                    # Check if message is from requester
                    if message_index is not None and message_index < len(messages):
                        msg = messages[message_index]
                        actual_role = msg.get("role", "").lower()
                        actual_author_id = msg.get("author_id", "")
                        
                        # Check by author_id first (more reliable)
                        is_requester = False
                        if requester_author_ids and actual_author_id:
                            is_requester = str(actual_author_id) in requester_author_ids
                        elif actual_role in ["requester", "user", "customer"]:
                            is_requester = True
                        
                        if not is_requester:
                            violations["not_requester_authored"].append({
                                "ticket_id": ticket_id,
                                "evidence_role": evidence_role,
                                "actual_role": actual_role,
                                "actual_author_id": actual_author_id,
                                "requester_ids": list(requester_author_ids) if requester_author_ids else []
                            })
                    elif evidence_role and evidence_role.lower() != "requester":
                        # Fallback: check evidence_role if message_index not available
                        violations["wrong_role"].append({
                            "ticket_id": ticket_id,
                            "evidence_role": evidence_role
                        })
        
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            continue
    
    return violations


def main():
    parser = argparse.ArgumentParser(
        description="Validate cache eligibility pipeline correctness",
        epilog="""
Examples:
  python Scraper/validate_cache_pipeline.py
  python Scraper/validate_cache_pipeline.py --approve-min-confidence 0.90
        """
    )
    parser.add_argument("--approve-min-confidence", type=float, default=0.90, help="Minimum confidence for approval (default: 0.90)")
    parser.add_argument("--require-requester-confirmation", action="store_true", default=True, help="Require requester confirmation (default: True)")
    parser.add_argument("--db", type=str, help="Database path override")
    parser.add_argument("--fix-inconsistencies", action="store_true", help="Fix review_status/cache_eligible inconsistencies (sets cache_eligible based on review_status)")
    
    args = parser.parse_args()
    
    # Initialize DB
    db_path = args.db or db.DEFAULT_DB_PATH
    db.init_db(db_path)
    conn = db.get_connection(db_path)
    
    try:
        print("=" * 70)
        print("Cache Eligibility Pipeline Validator")
        print("=" * 70)
        print()
        
        # 1. Check review_status/cache_eligible consistency
        print("1. Checking review_status/cache_eligible consistency...")
        consistency_issues = validate_review_status_cache_eligible_consistency(conn)
        
        approved_but_not = consistency_issues["approved_but_not_eligible"]
        not_approved_but = consistency_issues["not_approved_but_eligible"]
        
        if approved_but_not:
            print(f"  ERROR: APPROVED_BUT_NOT_ELIGIBLE: {len(approved_but_not)} -> {' '.join(approved_but_not)}")
        else:
            print(f"  OK: No approved tickets with cache_eligible != 1")
        
        if not_approved_but:
            print(f"  ERROR: NOT_APPROVED_BUT_ELIGIBLE: {len(not_approved_but)} -> {' '.join(not_approved_but)}")
        else:
            print(f"  OK: No non-approved tickets with cache_eligible == 1")
        
        print()
        
        # 2. Validate approved tickets meet criteria
        print("2. Validating approved tickets meet approval criteria...")
        approved_violations = validate_approved_criteria(
            conn, 
            approve_min_confidence=args.approve_min_confidence,
            require_requester_confirmation=args.require_requester_confirmation
        )
        
        total_violations = sum(len(v) for v in approved_violations.values())
        if total_violations > 0:
            print(f"  ERROR: Found {total_violations} violations:")
            for violation_type, tickets in approved_violations.items():
                if tickets:
                    ticket_ids = [t["ticket_id"] for t in tickets]
                    print(f"    - {violation_type.upper()}: {len(tickets)} -> {' '.join(ticket_ids)}")
        else:
            print(f"  OK: All approved tickets meet criteria")
        
        print()
        
        # 3. Validate hard-block phrases
        print("3. Validating hard-block trigger phrases...")
        hard_block_violations = validate_hard_block_phrases(conn)
        
        missing_phrases = hard_block_violations["missing_phrase"]
        if missing_phrases:
            print(f"  ERROR: MISSING_PHRASE: {len(missing_phrases)} -> {' '.join([t['ticket_id'] for t in missing_phrases])}")
        else:
            print(f"  OK: All hard-blocked tickets have matching phrases")
        
        print()
        
        # 4. Validate confirmation evidence
        print("4. Validating confirmation evidence grounding...")
        confirmation_violations = validate_confirmation_evidence(
            conn,
            require_requester_confirmation=args.require_requester_confirmation
        )
        
        total_conf_violations = sum(len(v) for v in confirmation_violations.values())
        if total_conf_violations > 0:
            print(f"  ERROR: Found {total_conf_violations} confirmation evidence violations:")
            for violation_type, tickets in confirmation_violations.items():
                if tickets:
                    ticket_ids = [t["ticket_id"] for t in tickets]
                    print(f"    - {violation_type.upper()}: {len(tickets)} -> {' '.join(ticket_ids)}")
        else:
            print(f"  OK: All confirmations have grounded evidence")
        
        print()
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        
        total_issues = (
            len(approved_but_not) + len(not_approved_but) +
            total_violations + len(missing_phrases) + total_conf_violations
        )
        
        if total_issues == 0:
            print("OK: All validations passed!")
        else:
            print(f"ERROR: Found {total_issues} total issues")
            print()
            print("Run --relabel-from-db to recompute review_status/cache_eligible")
            print("or manually review the listed tickets.")
        
        # Fix inconsistencies if requested
        if args.fix_inconsistencies:
            print()
            print("Fixing inconsistencies...")
            cursor = conn.cursor()
            
            fixed = 0
            for ticket_id in approved_but_not:
                cursor.execute("""
                    UPDATE ticket_judgements
                    SET cache_eligible = 1
                    WHERE ticket_id = ? AND review_status = 'approved'
                """, (ticket_id,))
                fixed += cursor.rowcount
            
            for ticket_id in not_approved_but:
                cursor.execute("""
                    UPDATE ticket_judgements
                    SET cache_eligible = 0
                    WHERE ticket_id = ? AND review_status != 'approved'
                """, (ticket_id,))
                fixed += cursor.rowcount
            
            conn.commit()
            print(f"Fixed {fixed} inconsistencies")
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()

