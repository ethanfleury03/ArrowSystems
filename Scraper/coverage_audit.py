#!/usr/bin/env python3
"""
Coverage audit: Check that all tickets are accounted for.
"""
import sqlite3
import sys
import db

def main():
    db_path = db.DEFAULT_DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("=" * 70)
    print("COVERAGE AUDIT")
    print("=" * 70)
    print()
    
    # 1. Tickets ingested vs judged
    cursor.execute("SELECT COUNT(*) FROM tickets_detail")
    ingested_count = cursor.fetchone()[0]
    print(f"1. Tickets ingested: {ingested_count}")
    
    cursor.execute("SELECT COUNT(DISTINCT ticket_id) FROM ticket_judgements")
    judged_count = cursor.fetchone()[0]
    print(f"   Tickets judged: {judged_count}")
    print(f"   Coverage: {judged_count}/{ingested_count} ({100*judged_count/max(ingested_count,1):.1f}%)")
    print()
    
    # 2. Missing judgments
    cursor.execute("""
        SELECT d.ticket_id 
        FROM tickets_detail d 
        LEFT JOIN ticket_judgements j ON d.ticket_id = j.ticket_id 
        WHERE j.ticket_id IS NULL
    """)
    missing = cursor.fetchall()
    print(f"2. Missing judgments: {len(missing)}")
    if missing:
        print(f"   Sample missing IDs: {[m['ticket_id'] for m in missing[:10]]}")
    print()
    
    # 3. Duplicate judgments
    cursor.execute("""
        SELECT ticket_id, COUNT(*) c 
        FROM ticket_judgements 
        GROUP BY ticket_id 
        HAVING c > 1
    """)
    dups = cursor.fetchall()
    print(f"3. Duplicate judgments: {len(dups)}")
    if dups:
        for dup in dups[:10]:
            print(f"   Ticket {dup['ticket_id']}: {dup['c']} judgments")
    print()
    
    # 4. Missing/empty transcripts
    cursor.execute("""
        SELECT ticket_id 
        FROM tickets_detail 
        WHERE conversation_json IS NULL OR conversation_json = '' 
        LIMIT 50
    """)
    empty = cursor.fetchall()
    print(f"4. Empty transcripts: {len(empty)}")
    if empty:
        print(f"   Sample empty IDs: {[e['ticket_id'] for e in empty[:10]]}")
    print()
    
    # 5. Needs review breakdown
    cursor.execute("""
        SELECT review_reason, COUNT(*) 
        FROM ticket_judgements 
        WHERE review_status = 'needs_review' 
        GROUP BY review_reason 
        ORDER BY COUNT(*) DESC
    """)
    needs_review = cursor.fetchall()
    print("5. Needs review breakdown:")
    total_needs_review = sum(r['COUNT(*)'] for r in needs_review)
    print(f"   Total needs_review: {total_needs_review}")
    for row in needs_review:
        print(f"   - {row['review_reason']}: {row['COUNT(*)']}")
    print()
    
    # 6. Overall status breakdown
    cursor.execute("""
        SELECT review_status, COUNT(*) 
        FROM ticket_judgements 
        GROUP BY review_status 
        ORDER BY COUNT(*) DESC
    """)
    status_breakdown = cursor.fetchall()
    print("6. Overall status breakdown:")
    for row in status_breakdown:
        print(f"   - {row['review_status']}: {row['COUNT(*)']}")
    print()
    
    conn.close()
    
    # Summary
    print("=" * 70)
    if len(missing) == 0 and len(dups) == 0 and len(empty) == 0:
        print("[PASS] Coverage audit PASSED: All tickets accounted for")
    else:
        print("[ISSUES] Coverage audit ISSUES FOUND:")
        if len(missing) > 0:
            print(f"  - {len(missing)} tickets missing judgments")
        if len(dups) > 0:
            print(f"  - {len(dups)} tickets have duplicate judgments")
        if len(empty) > 0:
            print(f"  - {len(empty)} tickets have empty transcripts")
    print("=" * 70)

if __name__ == "__main__":
    main()
