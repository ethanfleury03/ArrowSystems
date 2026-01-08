#!/usr/bin/env python3
"""
Test script to process a single ticket end-to-end.

Usage:
    python test_one_ticket.py <path_to_raw_ticket.json>
    
Example:
    python test_one_ticket.py data/tickets_raw/TICKET-001.json
"""

import json
import sys
from pathlib import Path

from Ticket import Ticket


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_one_ticket.py <path_to_raw_ticket.json>")
        sys.exit(1)
    
    ticket_path = Path(sys.argv[1])
    
    if not ticket_path.exists():
        print(f"Error: File not found: {ticket_path}")
        sys.exit(1)
    
    # Load raw ticket JSON
    print(f"Loading ticket from: {ticket_path}")
    with open(ticket_path, 'r', encoding='utf-8') as f:
        raw_ticket = json.load(f)
    
    # Extract ticket_id from the JSON or filename
    ticket_id = raw_ticket.get("ticket_id")
    if not ticket_id:
        # Try to get from metadata
        ticket_id = raw_ticket.get("metadata", {}).get("ticket_id")
    if not ticket_id:
        # Fallback to filename
        ticket_id = ticket_path.stem
    
    # Extract ticket_content
    # Handle different structures
    if "messages" in raw_ticket:
        ticket_content = raw_ticket
    elif "html" in raw_ticket:
        ticket_content = raw_ticket
    else:
        # Assume the whole thing is the content
        ticket_content = raw_ticket
    
    # Construct Ticket
    print(f"\nConstructing Ticket with ID: {ticket_id}")
    ticket = Ticket(ticket_id=ticket_id, ticket_content=ticket_content)
    
    # Print deterministic fields
    print("\n" + "=" * 60)
    print("Deterministic Fields:")
    print("=" * 60)
    print(f"Message Count: {ticket.message_count}")
    print(f"Error Codes: {ticket.error_codes}")
    print(f"Combined Text Length: {len(ticket.combined_text)} characters")
    print(f"Combined Text Preview (first 200 chars):")
    print(ticket.combined_text[:200] + "..." if len(ticket.combined_text) > 200 else ticket.combined_text)
    
    # Derive LLM fields
    print("\n" + "=" * 60)
    print("Deriving LLM fields...")
    print("=" * 60)
    try:
        ticket.derive()
        print("✓ LLM derivation successful")
    except Exception as e:
        print(f"✗ LLM derivation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print derived fields
    print("\n" + "=" * 60)
    print("LLM-Derived Fields:")
    print("=" * 60)
    print(f"Title: {ticket.ticket_title}")
    print(f"Description: {ticket.ticket_description}")
    print(f"Is Resolved: {ticket.is_resolved}")
    print(f"Solution Description: {ticket.solution_description}")
    
    # Save derived ticket
    print("\n" + "=" * 60)
    print("Saving derived ticket...")
    print("=" * 60)
    try:
        saved_path = ticket.save_derived()
        print(f"✓ Saved to: {saved_path}")
    except Exception as e:
        print(f"✗ Save failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print full dict
    print("\n" + "=" * 60)
    print("Full Ticket Dictionary:")
    print("=" * 60)
    print(json.dumps(ticket.to_dict(), indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 60)
    print("✓ Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


