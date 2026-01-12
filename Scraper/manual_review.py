#!/usr/bin/env python3
"""
Manual review override tool for cache eligibility pipeline.

Allows reviewers to manually approve/reject tickets without modifying raw_response_json.
Manual overrides take precedence over automated judgments.

Usage:
  # Interactive review mode (recommended)
  python manual_review.py --interactive --reviewer "Ethan"
  
  # Non-interactive: Approve a ticket
  python manual_review.py --id 3599 --approve --reason "Confirmed working via email"
  
  # Non-interactive: Reject a ticket
  python manual_review.py --id 4246 --reject --reason "Not actually resolved"
  
  # Approve with confirmation quote
  python manual_review.py --id 3688 --approve --confirmation-quote "Fixed, now working" --reviewer "John Doe"
  
  # List manual reviews
  python manual_review.py --list
  
  # Remove manual override (revert to automated judgment)
  python manual_review.py --id 3599 --remove
"""
import argparse
import json
import sys
import db


def fetch_tickets_needing_review(conn, resume: bool = True):
    """
    Fetch tickets that need manual review.
    
    Args:
        conn: Database connection
        resume: If True, skip tickets that already have manual reviews
        
    Returns:
        List of ticket dicts, sorted by priority
    """
    cursor = conn.cursor()
    
    # Build query: needs_review tickets, optionally excluding already-reviewed
    if resume:
        query = """
            SELECT 
                j.ticket_id,
                j.raw_response_json,
                j.confidence,
                j.review_reason,
                j.review_reasons_json,
                j.blockers_json,
                j.judged_at,
                j.model
            FROM ticket_judgements j
            LEFT JOIN ticket_manual_reviews m ON j.ticket_id = m.ticket_id
            WHERE j.review_status = 'needs_review'
            AND m.ticket_id IS NULL
        """
    else:
        query = """
            SELECT 
                ticket_id,
                raw_response_json,
                confidence,
                review_reason,
                review_reasons_json,
                blockers_json,
                judged_at,
                model
            FROM ticket_judgements
            WHERE review_status = 'needs_review'
        """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    # Process and enrich rows
    enriched = []
    for row in rows:
        row_dict = dict(row)
        ticket_id = row_dict['ticket_id']
        raw_json_str = row_dict.get('raw_response_json', '{}')
        
        try:
            raw_json = json.loads(raw_json_str) if raw_json_str else {}
        except (json.JSONDecodeError, TypeError):
            raw_json = {}
        
        # Extract fields
        outcome = raw_json.get('outcome', 'unclear')
        confirmation_obj = raw_json.get('confirmation', {})
        confirmation_confirmed = confirmation_obj.get('confirmed', False) if isinstance(confirmation_obj, dict) else bool(confirmation_obj)
        confirmation_evidence = confirmation_obj.get('evidence', {}) if isinstance(confirmation_obj, dict) else {}
        confirmation_quote = confirmation_evidence.get('quote', '') if confirmation_evidence else ''
        
        resolution_obj = raw_json.get('resolution', {})
        resolution_steps = resolution_obj.get('steps', []) if isinstance(resolution_obj, dict) else []
        
        # Get conversation for transcript
        conversation = db.get_ticket_detail_json(conn, ticket_id)
        transcript_excerpt = ''
        if conversation:
            messages = conversation.get('messages', [])
            if isinstance(messages, list):
                sorted_messages = sorted(messages, key=lambda m: m.get('created_at', ''))
                last_messages = sorted_messages[-10:] if len(sorted_messages) > 10 else sorted_messages
                excerpt_parts = []
                for msg in last_messages:
                    role = msg.get('role', 'unknown')
                    author_id = msg.get('author_id', '')
                    text = msg.get('text', '')[:200]
                    excerpt_parts.append(f"[{role}|{author_id}]: {text}")
                transcript_excerpt = '\n'.join(excerpt_parts)
        
        # Parse blockers
        blockers = []
        try:
            blockers = json.loads(row_dict.get('blockers_json', '[]') or '[]')
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Parse review_reasons
        review_reasons = []
        try:
            review_reasons = json.loads(row_dict.get('review_reasons_json', '[]') or '[]')
        except (json.JSONDecodeError, TypeError):
            pass
        
        enriched.append({
            'ticket_id': ticket_id,
            'outcome': outcome,
            'confidence': row_dict.get('confidence', 0.0),
            'review_reason': row_dict.get('review_reason', ''),
            'review_reasons': review_reasons,
            'confirmation_confirmed': confirmation_confirmed,
            'confirmation_quote': confirmation_quote,
            'resolution_steps': resolution_steps,
            'blockers': blockers,
            'transcript_excerpt': transcript_excerpt,
            'model': row_dict.get('model', ''),
            'judged_at': row_dict.get('judged_at', '')
        })
    
    # Sort: missing_confirmation first, then borderline_eligible by confidence desc, then others
    def sort_key(r):
        reason = r['review_reason']
        conf = r['confidence']
        if reason == 'missing_confirmation':
            return (0, -conf)
        elif reason == 'borderline_eligible':
            return (1, -conf)
        else:
            return (2, -conf)
    
    enriched.sort(key=sort_key)
    return enriched


def display_ticket(ticket, index, total):
    """Display a ticket's full review information."""
    print("\n" + "=" * 80)
    print(f"Ticket {index}/{total}: {ticket['ticket_id']}")
    print("=" * 80)
    print(f"Outcome: {ticket['outcome']}")
    print(f"Confidence: {ticket['confidence']:.2f}")
    print(f"Review Reason: {ticket['review_reason']}")
    
    if ticket['review_reasons']:
        print(f"Review Reasons: {', '.join(ticket['review_reasons'])}")
    
    if ticket['blockers']:
        print(f"\nBlockers:")
        for blocker in ticket['blockers'][:5]:
            print(f"  - {blocker}")
        if len(ticket['blockers']) > 5:
            print(f"  ... and {len(ticket['blockers']) - 5} more")
    
    print(f"\nConfirmation:")
    print(f"  Confirmed: {ticket['confirmation_confirmed']}")
    if ticket['confirmation_quote']:
        print(f"  Quote: {ticket['confirmation_quote'][:300]}")
    
    if ticket['resolution_steps']:
        print(f"\nResolution Steps ({len(ticket['resolution_steps'])}):")
        for i, step in enumerate(ticket['resolution_steps'][:5], 1):
            print(f"  {i}. {step[:200]}")
        if len(ticket['resolution_steps']) > 5:
            print(f"  ... and {len(ticket['resolution_steps']) - 5} more")
    
    if ticket['transcript_excerpt']:
        print(f"\nLast Messages:")
        print(ticket['transcript_excerpt'][:1000])
        if len(ticket['transcript_excerpt']) > 1000:
            print("  ... (truncated)")
    
    print(f"\nModel: {ticket['model']}")
    print(f"Judged At: {ticket['judged_at']}")
    print("=" * 80)


def interactive_review(conn, reviewer: str):
    """Interactive review mode: show tickets one by one."""
    tickets = fetch_tickets_needing_review(conn, resume=True)
    
    if not tickets:
        print("No tickets need review (all have been reviewed or none are in needs_review status).")
        return
    
    print(f"\nFound {len(tickets)} ticket(s) needing review")
    print("Order: missing_confirmation → borderline_eligible → others (by confidence)")
    print("\nCommands:")
    print("  [y]es  - Approve (cache eligible)")
    print("  [n]o   - Reject (not cache eligible)")
    print("  [s]kip - Skip this ticket (review later)")
    print("  [q]uit - Exit and save progress")
    print()
    
    approved_count = 0
    rejected_count = 0
    skipped_count = 0
    
    for i, ticket in enumerate(tickets, 1):
        display_ticket(ticket, i, len(tickets))
        
        while True:
            response = input("\nCache eligible? [y]es / [n]o / [s]kip / [q]uit: ").strip().lower()
            
            if response in ['q', 'quit']:
                print(f"\nExiting. Progress saved.")
                print(f"  Reviewed: {approved_count} approved, {rejected_count} rejected, {skipped_count} skipped")
                print(f"  Remaining: {len(tickets) - i + 1}")
                return
            
            elif response in ['s', 'skip']:
                skipped_count += 1
                print(f"Skipped ticket {ticket['ticket_id']}")
                break
            
            elif response in ['y', 'yes']:
                # Ask for optional reason and confirmation quote
                reason = input("Reason (optional, press Enter to skip): ").strip() or None
                quote = input("Confirmation quote (optional, press Enter to skip): ").strip() or None
                
                db.upsert_manual_review(
                    conn,
                    ticket['ticket_id'],
                    'approved',
                    manual_reason=reason,
                    manual_confirmation_quote=quote,
                    reviewer=reviewer
                )
                
                # Update ticket_judgements to reflect manual status
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE ticket_judgements
                    SET review_status = 'approved',
                        cache_eligible = 1,
                        review_reason = ?,
                        reviewed_at = datetime('now')
                    WHERE ticket_id = ?
                """, (
                    f"manual_approved" + (f": {reason}" if reason else ""),
                    ticket['ticket_id']
                ))
                conn.commit()
                
                approved_count += 1
                print(f"✓ Approved ticket {ticket['ticket_id']}")
                break
            
            elif response in ['n', 'no']:
                # Ask for optional reason
                reason = input("Reason (optional, press Enter to skip): ").strip() or None
                
                db.upsert_manual_review(
                    conn,
                    ticket['ticket_id'],
                    'rejected',
                    manual_reason=reason,
                    reviewer=reviewer
                )
                
                # Update ticket_judgements to reflect manual status
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE ticket_judgements
                    SET review_status = 'rejected',
                        cache_eligible = 0,
                        review_reason = ?,
                        reviewed_at = datetime('now')
                    WHERE ticket_id = ?
                """, (
                    f"manual_rejected" + (f": {reason}" if reason else ""),
                    ticket['ticket_id']
                ))
                conn.commit()
                
                rejected_count += 1
                print(f"✓ Rejected ticket {ticket['ticket_id']}")
                break
            
            else:
                print("Invalid input. Please enter y/n/s/q")
    
    print(f"\n{'=' * 80}")
    print("Review Complete!")
    print(f"{'=' * 80}")
    print(f"Total reviewed: {len(tickets)}")
    print(f"  Approved: {approved_count}")
    print(f"  Rejected: {rejected_count}")
    print(f"  Skipped: {skipped_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Manual review override for cache eligibility",
        epilog=__doc__
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive review mode (show tickets one by one)")
    parser.add_argument("--reviewer", type=str, help="Reviewer name/identifier (required for --interactive)")
    parser.add_argument("--id", type=str, help="Ticket ID to review (for non-interactive mode)")
    parser.add_argument("--approve", action="store_true", help="Manually approve ticket")
    parser.add_argument("--reject", action="store_true", help="Manually reject ticket")
    parser.add_argument("--reason", type=str, help="Reason for manual decision")
    parser.add_argument("--confirmation-quote", type=str, help="Confirmation quote (for approvals)")
    parser.add_argument("--remove", action="store_true", help="Remove manual override (revert to automated)")
    parser.add_argument("--list", action="store_true", help="List all manual reviews")
    parser.add_argument("--db", type=str, help="Database path override")
    
    args = parser.parse_args()
    
    # Initialize DB
    db_path = args.db or db.DEFAULT_DB_PATH
    db.init_db(db_path)
    conn = db.get_connection(db_path)
    
    try:
        if args.interactive:
            # Interactive review mode
            if not args.reviewer:
                print("Error: --reviewer required for --interactive mode", file=sys.stderr)
                sys.exit(1)
            interactive_review(conn, args.reviewer)
        
        elif args.list:
            # List all manual reviews
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ticket_id, manual_status, manual_reason, reviewer, reviewed_at
                FROM ticket_manual_reviews
                ORDER BY reviewed_at DESC
            """)
            rows = cursor.fetchall()
            
            if not rows:
                print("No manual reviews found.")
                return
            
            print(f"Found {len(rows)} manual reviews:")
            print()
            for row in rows:
                print(f"  Ticket {row['ticket_id']}: {row['manual_status'].upper()}")
                if row['manual_reason']:
                    print(f"    Reason: {row['manual_reason']}")
                if row['reviewer']:
                    print(f"    Reviewer: {row['reviewer']}")
                print(f"    Reviewed: {row['reviewed_at']}")
                print()
        
        elif args.remove:
            # Remove manual override
            if not args.id:
                print("Error: --id required for --remove", file=sys.stderr)
                sys.exit(1)
            
            cursor = conn.cursor()
            cursor.execute("DELETE FROM ticket_manual_reviews WHERE ticket_id = ?", (args.id,))
            conn.commit()
            
            if cursor.rowcount > 0:
                print(f"Removed manual override for ticket {args.id}")
            else:
                print(f"No manual override found for ticket {args.id}")
        
        elif args.approve or args.reject:
            # Non-interactive: Add/update manual review
            if not args.id:
                print("Error: --id required", file=sys.stderr)
                sys.exit(1)
            
            if args.approve and args.reject:
                print("Error: Cannot both approve and reject", file=sys.stderr)
                sys.exit(1)
            
            manual_status = "approved" if args.approve else "rejected"
            
            # Check if ticket exists in judgments
            cursor = conn.cursor()
            cursor.execute("SELECT ticket_id FROM ticket_judgements WHERE ticket_id = ?", (args.id,))
            if not cursor.fetchone():
                print(f"Error: Ticket {args.id} not found in ticket_judgements", file=sys.stderr)
                sys.exit(1)
            
            db.upsert_manual_review(
                conn,
                args.id,
                manual_status,
                manual_reason=args.reason,
                manual_confirmation_quote=args.confirmation_quote if args.approve else None,
                reviewer=args.reviewer
            )
            
            print(f"Manual {manual_status} recorded for ticket {args.id}")
            print(f"  Reason: {args.reason or '(none)'}")
            if args.approve and args.confirmation_quote:
                print(f"  Confirmation quote: {args.confirmation_quote}")
            if args.reviewer:
                print(f"  Reviewer: {args.reviewer}")
            
            # Update ticket_judgements to reflect manual status
            cursor.execute("""
                UPDATE ticket_judgements
                SET review_status = ?,
                    cache_eligible = ?,
                    review_reason = ?,
                    reviewed_at = datetime('now')
                WHERE ticket_id = ?
            """, (
                manual_status,
                1 if args.approve else 0,
                f"manual_{manual_status}" + (f": {args.reason}" if args.reason else ""),
                args.id
            ))
            conn.commit()
            
            print(f"Updated ticket_judgements.review_status to '{manual_status}'")
        
        else:
            parser.print_help()
            sys.exit(1)
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()
