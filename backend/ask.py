#!/usr/bin/env python3
"""One-shot CLI query runner for ArrowSystems RAG.

Loads a local LlamaIndex vector store from disk and answers a single query.
No database, no GCS, no web server — strictly local index only.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# ---- env: force local-only mode BEFORE any backend imports ----
os.environ["ENV"] = "dev"
os.environ["DEV_SKIP_DB"] = "true"
os.environ["DEV_SKIP_GCS"] = "true"
os.environ["TICKET_CACHE_ENABLED"] = "false"

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

_REQUIRED_INDEX_FILES = ("docstore.json", "index_store.json", "default__vector_store.json")


def _preflight(storage_dir: str) -> None:
    """Verify required index files exist before heavy imports."""
    storage_path = Path(storage_dir)
    if not storage_path.is_absolute():
        storage_path = storage_path.resolve()
    missing = [f for f in _REQUIRED_INDEX_FILES if not (storage_path / f).exists()]
    if missing:
        print(
            f"ERROR: Required index files missing from {storage_path}:\n"
            + "\n".join(f"  - {f}" for f in missing),
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Query local RAG index once and exit")
    parser.add_argument("-q", "--query", required=True, help="Question to ask")
    parser.add_argument("--storage-dir", default="latest_model")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--cache-dir", default=os.getenv("HF_HOME", "/tmp/hf_cache"))
    args = parser.parse_args()

    _preflight(args.storage_dir)

    from backend.rag_pipeline import RAGPipeline  # noqa: E402

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
