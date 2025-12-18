#!/usr/bin/env python3
"""
One-command smoke check for the local ingestion artifact.

Run:
    python -m backend.scripts.smoke_check_index

What it checks (fast, deterministic):
- doc_manifest.json exists (in INGEST_WORKDIR)
- index artifact directory exists (INGEST_WORKDIR/index_artifact by default)
- required index files exist
- index_manifest.json exists and reports num_chunks > 0 (if present)
- docstore.json has >0 nodes
- node metadata required keys exist and have expected types:
  - document_id: non-empty string
  - source_gcs: string starting with "gs://"
  - machine_model: list[str] (canonical runtime filter key)
  - machine_models: list[str] (alias)
  - machine_model_names: list[str] (alias)
  - machine_model_ids: list[str] (future-proof; may be empty but must exist)

Notes:
- This script does NOT rebuild anything; it only reads local files.
- It is intentionally strict (fail-fast) so CI/ops can trust it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    # backend/scripts/smoke_check_index.py -> backend -> repo root
    return Path(__file__).resolve().parent.parent.parent


def _default_workdir() -> Path:
    # Match backend/ingest.py defaults
    if os.name == "nt":
        return (_repo_root() / "ingest_work").resolve()
    return Path("/workspace/ingest_work").resolve()


def _resolve_workdir() -> Path:
    return Path(os.getenv("INGEST_WORKDIR", str(_default_workdir()))).resolve()


def _resolve_index_dir(workdir: Path) -> Path:
    # Primary expected layout used by the production ingestion flow
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
    # Default even if it doesn't exist (so error message is deterministic)
    return (workdir / "index_artifact").resolve()


def _fail(msg: str) -> "NoReturn":
    print(f"[SMOKE] ❌ {msg}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def _ok(msg: str) -> None:
    print(f"[SMOKE] ✅ {msg}", flush=True)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        _fail(f"Failed to read/parse JSON: {path} ({type(e).__name__}: {e})")


def _is_list_of_str(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(v, str) for v in x)


def main() -> None:
    workdir = _resolve_workdir()
    index_dir = _resolve_index_dir(workdir)

    print("[SMOKE] Index Artifact Smoke Check", flush=True)
    print(f"[SMOKE] workdir={workdir}", flush=True)
    print(f"[SMOKE] index_dir={index_dir}", flush=True)

    # 1) doc_manifest.json
    doc_manifest = workdir / "doc_manifest.json"
    if not doc_manifest.exists():
        _fail(f"Missing doc_manifest.json: {doc_manifest}")
    manifest_obj = _load_json(doc_manifest)
    docs = manifest_obj.get("documents") if isinstance(manifest_obj, dict) else None
    if not isinstance(docs, list):
        _fail("doc_manifest.json invalid: expected top-level {'documents': [...]} list")
    docs_count = len(docs)
    _ok(f"doc_manifest.json exists (documents={docs_count})")

    # 2) index directory + required files
    if not index_dir.exists() or not index_dir.is_dir():
        _fail(f"Index directory missing: {index_dir}")

    # Keep aligned with runtime downloader expectations + ingestion manifest
    required_files = [
        "docstore.json",
        "index_store.json",
        "default__vector_store.json",
        "index_manifest.json",
    ]
    missing = [f for f in required_files if not (index_dir / f).exists()]
    if missing:
        listing = sorted([p.name for p in index_dir.iterdir() if p.is_file()])
        _fail(f"Missing required index files: {missing}. Found files: {listing}")
    _ok(f"Required index files present: {', '.join(required_files)}")

    # 3) index_manifest.json sanity
    index_manifest = _load_json(index_dir / "index_manifest.json")
    num_chunks = None
    if isinstance(index_manifest, dict):
        num_chunks = index_manifest.get("num_chunks")
    try:
        if num_chunks is None or int(num_chunks) <= 0:
            _fail(f"index_manifest.json num_chunks must be > 0 (got {num_chunks})")
    except Exception:
        _fail(f"index_manifest.json num_chunks must be an int-like value (got {num_chunks})")
    _ok(f"index_manifest.json looks valid (num_chunks={int(num_chunks)})")

    # 4) docstore.json node checks
    docstore = _load_json(index_dir / "docstore.json")
    nodes = None
    if isinstance(docstore, dict):
        nodes = docstore.get("docstore/data")
    if not isinstance(nodes, dict):
        _fail("docstore.json invalid: expected top-level key 'docstore/data' mapping")
    node_count = len(nodes)
    if node_count <= 0:
        _fail("docstore.json has 0 nodes")
    _ok(f"docstore.json contains nodes (count={node_count})")

    required_meta_keys = [
        "document_id",
        "source_gcs",
        # canonical + aliases
        "machine_model",
        "machine_models",
        "machine_model_names",
        "machine_model_ids",
    ]

    missing_keys_total = {k: 0 for k in required_meta_keys}
    invalid_total = {k: 0 for k in required_meta_keys}
    unique_doc_ids: set[str] = set()
    with_machine_model_ids = 0

    for wrapped in nodes.values():
        data = wrapped.get("__data__") if isinstance(wrapped, dict) else None
        meta = data.get("metadata") if isinstance(data, dict) else None
        if not isinstance(meta, dict):
            for k in required_meta_keys:
                missing_keys_total[k] += 1
            continue

        for k in required_meta_keys:
            if k not in meta:
                missing_keys_total[k] += 1

        # document_id
        did = meta.get("document_id")
        if not (isinstance(did, str) and did.strip()):
            invalid_total["document_id"] += 1
        else:
            unique_doc_ids.add(did.strip())

        # source_gcs
        sg = meta.get("source_gcs")
        if not (isinstance(sg, str) and sg.startswith("gs://")):
            invalid_total["source_gcs"] += 1

        # machine_model canonical key: list[str]
        mm = meta.get("machine_model")
        if not _is_list_of_str(mm):
            invalid_total["machine_model"] += 1

        # aliases
        if not _is_list_of_str(meta.get("machine_models")):
            invalid_total["machine_models"] += 1
        if not _is_list_of_str(meta.get("machine_model_names")):
            invalid_total["machine_model_names"] += 1
        if not _is_list_of_str(meta.get("machine_model_ids")):
            invalid_total["machine_model_ids"] += 1
        else:
            if len(meta.get("machine_model_ids") or []) > 0:
                with_machine_model_ids += 1

    # Fail if missing keys or invalid types were found anywhere
    missing_any = {k: v for k, v in missing_keys_total.items() if v > 0}
    invalid_any = {k: v for k, v in invalid_total.items() if v > 0}
    if missing_any or invalid_any:
        _fail(f"Node metadata validation failed. missing_keys={missing_any} invalid_values={invalid_any} total_nodes={node_count}")

    pct_with_ids = (with_machine_model_ids / node_count) * 100.0 if node_count else 0.0

    print("", flush=True)
    print("[SMOKE] Summary", flush=True)
    print(f"[SMOKE] documents_in_manifest={docs_count}", flush=True)
    print(f"[SMOKE] nodes_in_docstore={node_count}", flush=True)
    print(f"[SMOKE] unique_document_ids_in_nodes={len(unique_doc_ids)}", flush=True)
    print(f"[SMOKE] nodes_with_machine_model_ids={with_machine_model_ids} ({pct_with_ids:.1f}%)", flush=True)

    _ok("Smoke check passed")


if __name__ == "__main__":
    main()


