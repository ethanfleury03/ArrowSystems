#!/usr/bin/env python3
"""
Ingestion Confirmation Runbook

A) Build-only ingestion (local index build + strict metadata verification)

export PROMOTE_INDEX=false
export GCS_DOCS_BUCKET=arrow-rag-support-prod-docs
export DOCS_GCS_PREFIX=ROOT
export INGEST_WORKDIR=/workspace/ingest_work
python -m backend.ingest
python -m backend.scripts.audit_ingestion_metadata

B) Promote flow (ONLY after audit passes)

export PROMOTE_INDEX=true
python -m backend.ingest

What PASS looks like
- backend/ingest.py prints "✅ Local index verification passed"
- This script exits 0 and prints 0 missing/invalid required fields
- Ingestion logs include:
  - event=ingest_resolved_doc_metadata_sample with correct gs://arrow-rag-support-prod-docs/... paths + machine models
  - event=chunk_metadata_sample with correct per-node metadata (no stale overrides)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _default_workdir() -> Path:
    # Match backend/ingest.py defaults
    if os.name == "nt":
        return (_repo_root() / "ingest_work").resolve()
    return Path("/workspace/ingest_work").resolve()


def _resolve_workdir() -> Path:
    return Path(os.getenv("INGEST_WORKDIR", str(_default_workdir()))).resolve()


def _resolve_index_dir(workdir: Path) -> Path:
    candidates = [
        workdir / "index_artifact",
        workdir / "index",
        workdir / "latest_model",
    ]
    env_override = os.getenv("INDEX_OUT_DIR") or os.getenv("RAG_INDEX_LOCAL_DIR")
    if env_override:
        candidates.insert(0, Path(env_override))
    for c in candidates:
        c = c.resolve()
        if c.exists() and c.is_dir():
            return c
    return (workdir / "index_artifact").resolve()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_list_of_str(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(v, str) for v in x)


def _is_list_of_int(x: Any) -> bool:
    # bool is an int subclass; exclude it
    return isinstance(x, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in x)


def main() -> int:
    workdir = _resolve_workdir()
    index_dir = _resolve_index_dir(workdir)

    docs_bucket = os.getenv("GCS_DOCS_BUCKET") or os.getenv("DOCS_GCS_BUCKET") or "arrow-rag-support-prod-docs"
    required_source_prefix = f"gs://{docs_bucket}/"

    docstore_path = index_dir / "docstore.json"
    if not docstore_path.exists():
        print(f"[AUDIT] ❌ Missing docstore.json: {docstore_path}", file=sys.stderr, flush=True)
        return 1

    docstore = _load_json(docstore_path)
    nodes = docstore.get("docstore/data") if isinstance(docstore, dict) else None
    if not isinstance(nodes, dict) or not nodes:
        print("[AUDIT] ❌ docstore/data missing or empty", file=sys.stderr, flush=True)
        return 1

    required_keys = [
        "document_id",
        "source_gcs",
        "machine_model",
        "machine_model_ids",
        "machine_model_names",
    ]

    missing_key_counts = {k: 0 for k in required_keys}
    invalid_type_counts = {k: 0 for k in required_keys}

    total = 0
    samples: list[dict[str, Any]] = []

    for wrapped in nodes.values():
        total += 1
        data = wrapped.get("__data__") if isinstance(wrapped, dict) else None
        meta = data.get("metadata") if isinstance(data, dict) else None
        if not isinstance(meta, dict):
            for k in required_keys:
                missing_key_counts[k] += 1
            continue

        for k in required_keys:
            if k not in meta:
                missing_key_counts[k] += 1

        did = meta.get("document_id")
        if not (isinstance(did, str) and did.strip()):
            invalid_type_counts["document_id"] += 1

        sg = meta.get("source_gcs")
        if not (isinstance(sg, str) and sg.startswith(required_source_prefix)):
            invalid_type_counts["source_gcs"] += 1

        mm = meta.get("machine_model")
        if not _is_list_of_str(mm):
            invalid_type_counts["machine_model"] += 1

        mmn = meta.get("machine_model_names")
        if not _is_list_of_str(mmn):
            invalid_type_counts["machine_model_names"] += 1

        mmids = meta.get("machine_model_ids")
        if not _is_list_of_int(mmids):
            invalid_type_counts["machine_model_ids"] += 1

        if len(samples) < 5:
            samples.append(
                {
                    "document_id": did,
                    "source_gcs": sg,
                    "machine_model": mm,
                    "machine_model_ids": mmids,
                    "machine_model_names": mmn,
                }
            )

    print("[AUDIT] Ingestion Metadata Audit", flush=True)
    print(f"[AUDIT] workdir={workdir}", flush=True)
    print(f"[AUDIT] index_dir={index_dir}", flush=True)
    print(f"[AUDIT] nodes_total={total}", flush=True)
    print(f"[AUDIT] source_gcs_required_prefix={required_source_prefix}", flush=True)
    print("", flush=True)

    print("[AUDIT] Missing key counts:", flush=True)
    for k, v in missing_key_counts.items():
        print(f"  - {k}: {v}", flush=True)

    print("[AUDIT] Invalid type/value counts:", flush=True)
    for k, v in invalid_type_counts.items():
        print(f"  - {k}: {v}", flush=True)

    print("", flush=True)
    print("[AUDIT] Sample nodes (first 5):", flush=True)
    for s in samples:
        print(json.dumps(s, ensure_ascii=False), flush=True)

    missing_any = any(v > 0 for v in missing_key_counts.values())
    invalid_any = any(v > 0 for v in invalid_type_counts.values())
    if missing_any or invalid_any:
        print("[AUDIT] ❌ FAILED", file=sys.stderr, flush=True)
        return 1

    print("[AUDIT] ✅ PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


