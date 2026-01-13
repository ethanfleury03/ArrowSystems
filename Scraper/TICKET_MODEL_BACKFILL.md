# Ticket Machine Model Backfill

One-time backfill script that auto-assigns machine models to tickets by scanning ticket conversation text for machine model names/aliases.

## Overview

This script:
- Scans ticket conversation text (subject + description + messages) for machine model mentions
- Matches against machine models from the RAG app database
- Auto-assigns models when exactly 1 match is found
- Flags ambiguous cases (multiple matches) for manual review
- Leaves tickets unassigned if no matches found

## Prerequisites

1. **Tickets database**: `Scraper/data/tickets.db` must exist with populated `tickets_index` and `tickets_detail` tables
2. **Machine models source**: Either:
   - **Cloud SQL access**: `DATABASE_URL` in root `.env` file (automatically detected)
     - For Cloud SQL: Use Cloud SQL Proxy or ensure database is accessible
     - For local: Ensure PostgreSQL is running
   - **OR exported JSON file**: `machine_models.json` (use `--models-source json`)

## Usage

### Basic Usage (Dry-Run)

```bash
# Dry-run mode (default, safe, no DB writes)
python scripts/backfill_ticket_machine_models.py

# Dry-run with limit (test on first 100 tickets)
python scripts/backfill_ticket_machine_models.py --limit 100

# Dry-run for specific ticket
python scripts/backfill_ticket_machine_models.py --ticket-id 12345
```

### Write Mode (Apply Changes)

```bash
# Enable writes (requires --write flag)
python scripts/backfill_ticket_machine_models.py --write

# Write with limit (safe testing)
python scripts/backfill_ticket_machine_models.py --write --limit 100
```

### Machine Models Source

**Option 1: Cloud SQL (default)**
```bash
# Set DATABASE_URL environment variable
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
python scripts/backfill_ticket_machine_models.py --write
```

**Option 2: JSON File**
```bash
# Export models first (from backend):
# python -c "from backend.utils.db import SessionLocal, MachineModel; import json; ..."
# Then use JSON file:
python scripts/backfill_ticket_machine_models.py \
    --models-source json \
    --models-json path/to/machine_models.json \
    --write
```

### Self-Test Mode

```bash
# Test matcher on sample strings (no DB required)
python scripts/backfill_ticket_machine_models.py --self-test
```

## Command-Line Options

```
--db PATH                 Path to tickets.db (default: Scraper/data/tickets.db)
--outdir DIR              Output directory (default: out/)
--dry-run                 Dry-run mode (default: True)
--write                   Enable database writes (overrides --dry-run)
--limit N                 Limit number of tickets to process
--ticket-id ID            Process specific ticket ID only
--min-score X             Minimum match score threshold (default: 50)
--ambiguous-threshold X    Ambiguity threshold 0.0-1.0 (default: 0.1)
--models-source SOURCE     Source: "cloudsql" or "json" (default: cloudsql)
--models-json PATH        Path to machine_models.json (if models-source=json)
--self-test               Run self-test mode
```

## Matching Logic

The matcher uses deterministic string matching (no LLM):

1. **Normalization**: Lowercase, collapse whitespace, remove punctuation
2. **Scoring**:
   - Exact full name match = 100 points
   - Alias match (spacing/case variants) = 80 points
   - Partial token match (unique) = 50 points
3. **Word-boundary matching**: Prevents false positives
4. **False positive prevention**:
   - Minimum token length: 3 characters
   - Excludes common English words
   - Requires word boundaries

## Assignment Rules

- **Assigned**: Exactly 1 match found → auto-assign
- **Ambiguous**: Multiple matches with scores within 10% → flag for review
- **Unassigned**: No matches found → leave unassigned

## Output Files

The script always exports two files (even in dry-run):

1. **`out/ticket_machine_model_backfill.jsonl`**
   - One JSON object per ticket
   - Includes matches, assignment, conversation length

2. **`out/ticket_machine_model_backfill.csv`**
   - CSV format for manual review
   - Columns: ticket_id, status, confidence, machine_model_ids, match_count, evidence_snippet

## Database Schema

The script creates two tables:

### `ticket_machine_model_matches`
One row per match:
- `ticket_id` (TEXT)
- `machine_model_id` (INTEGER)
- `machine_model_name` (TEXT)
- `match_source` (TEXT: "name" | "alias" | "token")
- `score` (INTEGER)
- `evidence_snippet` (TEXT)
- `created_at` (TEXT)

### `ticket_machine_model_assignment`
One row per ticket (summary):
- `ticket_id` (TEXT PRIMARY KEY)
- `machine_model_ids` (TEXT JSON array)
- `status` (TEXT: "unassigned" | "assigned" | "ambiguous")
- `confidence` (REAL 0.0-1.0)
- `method` (TEXT: "regex_match_v1")
- `updated_at` (TEXT)

## Example Output

```json
{
  "ticket_id": "12345",
  "matches": [
    {
      "model_id": 1,
      "model_name": "DuraFlex",
      "match_source": "name",
      "score": 100,
      "evidence_snippet": "...I have a DuraFlex machine that's not working..."
    }
  ],
  "assignment": {
    "machine_model_ids": [1],
    "status": "assigned",
    "confidence": 1.0,
    "method": "regex_match_v1"
  },
  "conversation_length": 1250
}
```

## Safety Features

- **Dry-run by default**: Must use `--write` to enable database writes
- **Idempotent**: Safe to run multiple times (uses INSERT OR REPLACE)
- **Non-destructive**: Only adds new tables/columns, doesn't modify existing ticket data
- **Comprehensive logging**: Shows progress and summary statistics
- **Error handling**: Continues processing if individual tickets fail

## Troubleshooting

**Error: "Tickets database not found"**
- Ensure `Scraper/data/tickets.db` exists
- Or use `--db` to specify custom path

**Error: "DATABASE_URL environment variable not set"**
- Set `DATABASE_URL` for Cloud SQL access
- OR use `--models-source json --models-json path/to/file.json`

**No matches found**
- Check that machine model names match common variations in ticket text
- Try lowering `--min-score` threshold
- Review `out/ticket_machine_model_backfill.csv` for false negatives

**Too many ambiguous matches**
- Increase `--ambiguous-threshold` (e.g., 0.2 for 20% threshold)
- Review ambiguous tickets in CSV output

## Next Steps

After running the backfill:

1. **Review ambiguous tickets**: Check `out/ticket_machine_model_backfill.csv` for tickets with `status=ambiguous`
2. **Manual review**: Use CSV to identify tickets needing manual assignment
3. **Verify assignments**: Spot-check assigned tickets to ensure accuracy
4. **Re-run if needed**: Script is idempotent, safe to re-run after fixing issues
