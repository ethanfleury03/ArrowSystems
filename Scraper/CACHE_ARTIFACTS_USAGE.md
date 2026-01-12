# Ticket Cache Artifacts Export & Ingestion

This document describes how to export cache-eligible Zendesk tickets and ingest them into the RAG system.

## Overview

Cache-eligible tickets are converted into RAG-ready artifacts (JSONL format) that can be indexed alongside technical documentation. This enables the RAG system to surface proven solutions from past support tickets.

## Export Cache Artifacts

Export cache-eligible tickets from `tickets.db`:

```bash
# Export all cache-eligible tickets
python export_cache_artifacts.py --db data/tickets.db --out out/cache_artifacts.jsonl

# Export specific ticket
python export_cache_artifacts.py --db data/tickets.db --out out/cache_artifacts.jsonl --ticket-id 12345

# Export first 10 tickets (for testing)
python export_cache_artifacts.py --db data/tickets.db --out out/cache_artifacts.jsonl --limit 10

# Overwrite existing file
python export_cache_artifacts.py --db data/tickets.db --out out/cache_artifacts.jsonl --force
```

The exporter:
- Queries `ticket_judgements` for tickets with `cache_eligible=1` AND (auto-approved OR manual-approved)
- Does NOT filter on `confirmation.confirmed` (only `cache_eligible` matters)
- Converts each ticket's `raw_response_json` into a standardized artifact format
- Writes JSONL file: `out/cache_artifacts.jsonl`

## Ingest into RAG Index

Ingest the exported artifacts into the LlamaIndex RAG system:

```bash
# Ingest all artifacts
python -m backend.scripts.ingest_ticket_cache_artifacts --jsonl out/cache_artifacts.jsonl

# Dry run (validate only, don't insert)
python -m backend.scripts.ingest_ticket_cache_artifacts --jsonl out/cache_artifacts.jsonl --dry-run

# Limit to first 10 artifacts
python -m backend.scripts.ingest_ticket_cache_artifacts --jsonl out/cache_artifacts.jsonl --limit 10

# Skip artifacts that already exist in index
python -m backend.scripts.ingest_ticket_cache_artifacts --jsonl out/cache_artifacts.jsonl --skip-existing
```

The ingestion script:
- Reads JSONL file and validates artifacts
- Converts artifacts to LlamaIndex `TextNode` objects
- Inserts nodes into existing RAG index (incremental)
- Uses `node_id=artifact.id` for deduplication
- Persists updated index

## Artifact Format

Each artifact in the JSONL file has this structure:

```json
{
  "id": "ticket:12345",
  "text": "Problem: ...\nResolution Steps:\n1. ...\n2. ...\nOutcome: ...\nRationale: ...",
  "metadata": {
    "document_id": "ticket:12345",
    "file_name": "ticket_12345.md",
    "content_type": "ticket_cache",
    "source": "zendesk_ticket",
    "ticket_id": "12345",
    "outcome": "resolved_remotely_actionable",
    "confidence": 0.95,
    "cache_eligible": 1,
    "confirmed": true,
    "rationale": "...",
    "blockers": [],
    "machine_model_ids": [],
    "machine_model_names": [],
    "machine_models": [],
    "machine_model": []
  }
}
```

## Notes

- **Deterministic**: Text generation uses a stable template (no LLM calls)
- **Deduplication**: Artifacts use `ticket:{ticket_id}` as unique ID
- **Incremental**: Ingestion adds to existing index without rebuilding
- **Safe**: Dry-run mode available for validation before insertion
