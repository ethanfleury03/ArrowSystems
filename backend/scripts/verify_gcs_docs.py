#!/usr/bin/env python3
"""
Quick GCS docs inventory (no HTTP, no DB).

Usage:
  python -m backend.scripts.verify_gcs_docs

Env:
  GCS_DOCS_BUCKET (default arrow-rag-support-prod-docs)
  GCS_DOCS_PREFIX (default "" (bucket root); supports "ROOT")
"""

from __future__ import annotations

import os
from pathlib import Path

from backend.config.env import normalize_gcs_prefix
from backend.utils.gcs_client import list_objects


def main() -> int:
    bucket = os.getenv("GCS_DOCS_BUCKET", "arrow-rag-support-prod-docs").strip()
    prefix = normalize_gcs_prefix(os.environ.get("GCS_DOCS_PREFIX"))
    supported = {".pdf", ".docx", ".md", ".markdown"}

    objs = list_objects(bucket, prefix)
    files = [o for o in objs if o.name and not o.name.endswith("/") and Path(o.name).suffix.lower() in supported]

    print(f"GCS: gs://{bucket}/{prefix}" if prefix else f"GCS: gs://{bucket}/ (bucket root)")
    print(f"total_objects_under_prefix={len(objs)}")
    print(f"supported_docs_under_prefix={len(files)}")
    print("first_50_supported:")
    for o in files[:50]:
        print(f"- {o.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


