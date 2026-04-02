#!/usr/bin/env python3
"""One-shot CLI query runner for ArrowSystems RAG."""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# ---- env bootstrap BEFORE backend imports ----
os.environ.setdefault("ENV", "dev")
os.environ.setdefault("DEV_SKIP_DB", "true")
os.environ.setdefault("DEV_SKIP_GCS", "true")
os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@127.0.0.1:5432/db")
os.environ.setdefault("TICKET_CACHE_ENABLED", "false")

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.rag_pipeline import RAGPipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Query local RAG index once and exit")
    parser.add_argument("-q", "--query", required=True, help="Question to ask")
    parser.add_argument("--storage-dir", default="latest_model")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--cache-dir", default=os.getenv("HF_HOME", "/tmp/hf_cache"))
    args = parser.parse_args()

    pipeline = RAGPipeline(cache_dir=args.cache_dir, db_manager=None)
    ok = pipeline.initialize(storage_dir=args.storage_dir)
    if not ok:
        print("ERROR: Failed to initialize RAG pipeline", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    response = pipeline.query(
        query=args.query,
        top_k=args.top_k,
        alpha=args.alpha,
        role="ADMIN",
        user_machine_models=None,
        machine_confirmation=False,
    )
    elapsed = time.time() - t0

    if args.format == "json":
        print(json.dumps({
            "query": response.query,
            "answer": response.answer,
            "confidence": response.confidence,
            "sources": response.sources,
            "elapsed_seconds": round(elapsed, 2),
        }, indent=2, default=str))
    else:
        print(pipeline.format_response(response))
        print(f"\n[{elapsed:.2f}s]")


if __name__ == "__main__":
    main()
