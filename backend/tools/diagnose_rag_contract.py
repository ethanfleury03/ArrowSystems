"""
Diagnostic script to validate ingestion/query contract alignment for RAG system.

Checks critical alignment points:
1. Index file naming and presence
2. Filename normalization (DB vs chunk metadata)
3. Machine filtering split-brain (document-level vs chunk-level)
4. Chunk machine_model_ids health
5. Storage directory resolution consistency
6. Docstore ↔ vector store node_id join consistency
7. Embedding dimension match
8. Text presence / non-empty content
9. Required metadata type correctness
10. Customer visibility simulation

Usage:
    python -m backend.tools.diagnose_rag_contract --storage-dir latest_model --role ADMIN
    python -m backend.tools.diagnose_rag_contract --storage-dir latest_model --role CUSTOMER --user-machine "EZCut 330"
    python -m backend.tools.diagnose_rag_contract --download-from-gcs --role ADMIN
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# ---------- Utilities ----------


def normalize_filename(name: str) -> str:
    """Normalize filename to basename only for consistent matching."""
    return os.path.basename((name or "").strip())


def safe_print(s: str) -> None:
    """Print and flush immediately."""
    sys.stdout.write(s + "\n")
    sys.stdout.flush()


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file with UTF-8 encoding."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_docstore_nodes(docstore_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Supports common LlamaIndex docstore formats:
      - {"docstore/data": {node_id: node_data}}
      - {"docstore": {"data": {node_id: node_data}}}
    """
    if "docstore/data" in docstore_json and isinstance(docstore_json["docstore/data"], dict):
        return docstore_json["docstore/data"]
    if "docstore" in docstore_json and isinstance(docstore_json["docstore"], dict):
        ds = docstore_json["docstore"]
        if "data" in ds and isinstance(ds["data"], dict):
            return ds["data"]
    # Some variants may store under "data" directly
    if "data" in docstore_json and isinstance(docstore_json["data"], dict):
        return docstore_json["data"]
    return {}


def get_node_metadata(node_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node payload format varies. LlamaIndex stores nodes wrapped in __data__.
    
    Common formats:
      - {"__data__": {"metadata": {...}, "text": "...", ...}}
      - {"metadata": {...}, "text": "...", ...}  (direct)
      - {"_node_data": {"metadata": {...}, ...}}  (alternative wrapper)
    """
    # LlamaIndex format: nodes are wrapped in __data__
    if "__data__" in node_data:
        inner_data = node_data["__data__"]
        if isinstance(inner_data, dict):
            meta = inner_data.get("metadata")
            if isinstance(meta, dict):
                return meta
    
    # Try direct metadata key
    meta = node_data.get("metadata")
    if isinstance(meta, dict):
        return meta
    
    # Try extra_info
    extra = node_data.get("extra_info")
    if isinstance(extra, dict):
        return extra
    
    # LlamaIndex nodes might have _node_data or node_data structure
    if "_node_data" in node_data:
        node_inner = node_data["_node_data"]
        if isinstance(node_inner, dict):
            meta = node_inner.get("metadata")
            if isinstance(meta, dict):
                return meta
    
    # Some formats store metadata at top level mixed with other fields
    # Extract known metadata keys from top level
    known_metadata_keys = [
        "file_name", "filename", "page_label", "page_number", "content_type",
        "machine_model_ids", "machine_model", "machine_models", "document_id",
        "source_gcs", "gcs_path", "local_path", "file_type"
    ]
    extracted = {}
    for key in known_metadata_keys:
        if key in node_data:
            extracted[key] = node_data[key]
    
    return extracted if extracted else {}


def parse_machine_model_field(value: Any) -> List[str]:
    """
    Document.machine_model can be:
      - None
      - "GENERAL"
      - '["DuraFlex","GENERAL"]'
      - "['DuraFlex']" (sometimes)
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        # Try JSON list
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if x is not None]
            except Exception:
                # fallthrough: treat as single string
                pass
        return [s]
    return [str(value)]


def intersect_case_insensitive(a: Iterable[str], b: Iterable[str]) -> bool:
    """Check if two string iterables have case-insensitive intersection."""
    set_a = {x.strip().lower() for x in a if x}
    set_b = {x.strip().lower() for x in b if x}
    return len(set_a.intersection(set_b)) > 0


def download_index_for_diagnosis(storage_dir: Path) -> Tuple[bool, str]:
    """
    Download RAG index from GCS for diagnostic analysis.
    
    Reads environment variables directly to avoid Settings initialization (which requires DATABASE_URL).
    Tries Python google-cloud-storage library first, falls back to gsutil command-line tool.
    
    Returns:
        (success: bool, error_message: str)
    """
    # Try Python library first
    try:
        from google.cloud import storage
        use_python_lib = True
    except ImportError:
        use_python_lib = False
        # Will try gsutil fallback below
    
    # Read environment variables directly (same defaults as Settings._load_rag_index_config)
    bucket_name = os.getenv("RAG_INDEX_GCS_BUCKET", "arrow-rag-support-prod-rag").strip()
    raw_prefix = os.getenv("RAG_INDEX_GCS_PREFIX", "latest_model/")
    raw_prefix = (raw_prefix or "").strip()
    
    # Normalize prefix (same logic as Settings)
    if raw_prefix:
        normalized = raw_prefix.strip("/")
        index_prefix = f"{normalized}/" if normalized else ""
    else:
        index_prefix = ""
    
    # Ensure directory exists
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[DIAG] Downloading index from GCS...", flush=True)
    print(f"[DIAG]   Bucket: {bucket_name}", flush=True)
    print(f"[DIAG]   Prefix: {index_prefix}", flush=True)
    print(f"[DIAG]   Local: {storage_dir}", flush=True)
    
    # Required files to download
    REQUIRED_FILES = [
        "docstore.json",
        "index_store.json",
        "default__vector_store.json",
    ]
    
    # Try Python library first
    if use_python_lib:
        try:
            # Initialize GCS client
            print(f"[DIAG] Using Python google-cloud-storage library...", flush=True)
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            print(f"[DIAG] ✅ GCS client initialized", flush=True)
            
            # Download required files
            required_success = []
            required_failures = []
            
            for filename in REQUIRED_FILES:
                gcs_obj = f"{index_prefix}{filename}" if index_prefix else filename
                gcs_path = f"gs://{bucket_name}/{gcs_obj}"
                local_file_path = storage_dir / filename
                
                try:
                    blob = bucket.blob(gcs_obj)
                    print(f"[DIAG] Downloading {filename}...", flush=True)
                    blob.download_to_filename(str(local_file_path))
                    
                    if not local_file_path.exists():
                        error = f"Download completed but file not found: {filename}"
                        print(f"[DIAG] ❌ {error}", flush=True)
                        required_failures.append(filename)
                    else:
                        size = local_file_path.stat().st_size
                        print(f"[DIAG] ✅ Downloaded {filename} ({size:,} bytes)", flush=True)
                        required_success.append(filename)
                except Exception as e:
                    error = f"Failed to download {filename}: {type(e).__name__}: {str(e)}"
                    print(f"[DIAG] ❌ {error}", flush=True)
                    required_failures.append(filename)
            
            if required_failures:
                error_msg = f"Failed to download {len(required_failures)} required file(s): {', '.join(required_failures)}"
                print(f"[DIAG] ❌ {error_msg}", flush=True)
                return False, error_msg
            
            print(f"[DIAG] ✅ Index downloaded successfully ({len(required_success)} files)", flush=True)
            return True, ""
            
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            print(f"[DIAG] ❌ Python library download exception: {error}", flush=True)
            print(f"[DIAG] Falling back to gsutil...", flush=True)
            # Fall through to gsutil
    
    # Fallback to gsutil command-line tool
    import subprocess
    import shutil
    
    gsutil_path = shutil.which("gsutil")
    if not gsutil_path:
        error = "Neither google-cloud-storage Python library nor gsutil command-line tool is available.\n" \
                "Install one of:\n" \
                "  - Python library: pip install google-cloud-storage\n" \
                "  - Google Cloud SDK: https://cloud.google.com/sdk/docs/install (includes gsutil)"
        print(f"[DIAG] ❌ {error}", flush=True)
        return False, error
    
    print(f"[DIAG] Using gsutil command-line tool...", flush=True)
    print(f"[DIAG]   gsutil path: {gsutil_path}", flush=True)
    
    try:
        required_success = []
        required_failures = []
        
        for filename in REQUIRED_FILES:
            gcs_obj = f"{index_prefix}{filename}" if index_prefix else filename
            gcs_path = f"gs://{bucket_name}/{gcs_obj}"
            local_file_path = storage_dir / filename
            
            try:
                print(f"[DIAG] Downloading {filename}...", flush=True)
                # Run: gsutil cp gs://bucket/prefix/file.json /local/path/file.json
                result = subprocess.run(
                    [gsutil_path, "cp", gcs_path, str(local_file_path)],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout per file
                )
                
                if result.returncode != 0:
                    error = f"gsutil failed: {result.stderr.strip() or result.stdout.strip()}"
                    print(f"[DIAG] ❌ {error}", flush=True)
                    required_failures.append(filename)
                elif not local_file_path.exists():
                    error = f"Download completed but file not found: {filename}"
                    print(f"[DIAG] ❌ {error}", flush=True)
                    required_failures.append(filename)
                else:
                    size = local_file_path.stat().st_size
                    print(f"[DIAG] ✅ Downloaded {filename} ({size:,} bytes)", flush=True)
                    required_success.append(filename)
            except subprocess.TimeoutExpired:
                error = f"Download timeout for {filename}"
                print(f"[DIAG] ❌ {error}", flush=True)
                required_failures.append(filename)
            except Exception as e:
                error = f"Failed to download {filename}: {type(e).__name__}: {str(e)}"
                print(f"[DIAG] ❌ {error}", flush=True)
                required_failures.append(filename)
        
        if required_failures:
            error_msg = f"Failed to download {len(required_failures)} required file(s): {', '.join(required_failures)}"
            print(f"[DIAG] ❌ {error_msg}", flush=True)
            return False, error_msg
        
        print(f"[DIAG] ✅ Index downloaded successfully via gsutil ({len(required_success)} files)", flush=True)
        return True, ""
        
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)}"
        print(f"[DIAG] ❌ gsutil download exception: {error}", flush=True)
        return False, error


# ---------- Checks ----------


@dataclass
class CheckResult:
    """Result of a diagnostic check."""
    name: str
    status: str  # PASS/WARN/FAIL
    key_numbers: str
    notes: str


def check_index_files(storage_dir: Path) -> Tuple[CheckResult, Dict[str, Any]]:
    """Check (1): Index file naming and presence."""
    required = ["docstore.json", "index_store.json", "default__vector_store.json"]
    
    if not storage_dir.exists():
        return (
            CheckResult(
                name="Index files present",
                status="FAIL",
                key_numbers="storage_dir missing",
                notes=str(storage_dir),
            ),
            {},
        )
    
    present = {p.name for p in storage_dir.glob("*.json")}
    missing = [f for f in required if f not in present]
    vector_like = sorted([p.name for p in storage_dir.glob("*vector_store*.json")])
    
    if missing:
        note = f"Missing: {', '.join(missing)}. Found vector-like: {vector_like or 'none'}"
        status = "FAIL"
    else:
        status = "PASS"
        note = f"Required OK. Vector-like: {vector_like or 'none'}"
        if "default__vector_store.json" in present and "vector_store.json" in present:
            status = "WARN"
            note += " (both default__vector_store.json and vector_store.json exist; confirm orchestrator uses the right dir)"
    
    return (
        CheckResult(
            name="Index files present",
            status=status,
            key_numbers=f"{len(present)} json files",
            notes=note,
        ),
        {"present": sorted(present), "missing": missing, "vector_like": vector_like},
    )


def load_db_documents() -> Tuple[List[Any], Optional[str]]:
    """
    Load all Document records from database.
    
    Returns:
        (documents: List[Any], error_message: Optional[str])
        If error_message is not None, database access failed.
    """
    try:
        from backend.utils.db import SessionLocal, Document  # type: ignore
    except Exception as e:
        return [], f"Failed to import database modules: {type(e).__name__}: {str(e)}"
    
    try:
        session = SessionLocal()
        try:
            return session.query(Document).all(), None
        finally:
            session.close()
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return [], error_msg


def check_filename_alignment(storage_dir: Path, sample: int) -> Tuple[CheckResult, Dict[str, Any]]:
    """Check (2): Filename normalization alignment (DB vs chunk metadata) with canonical comparison."""
    try:
        from backend.utils.filenames import canonicalize_filename
    except ImportError:
        # Fallback if module not available
        def canonicalize_filename(name: str) -> str:
            return normalize_filename(name)
    
    docstore_path = storage_dir / "docstore.json"
    if not docstore_path.exists():
        return (
            CheckResult("Filename alignment", "FAIL", "docstore.json missing", ""),
            {},
        )
    
    docstore = load_json(docstore_path)
    nodes = get_docstore_nodes(docstore)
    if not nodes:
        return (
            CheckResult("Filename alignment", "FAIL", "0 nodes", "docstore has no nodes under expected keys"),
            {},
        )
    
    # Chunk filenames from docstore - track multiple metrics
    chunk_files_raw: Set[str] = set()
    chunk_files_canonical: Set[str] = set()
    chunk_files_base: Set[str] = set()
    missing_file_name_key = 0
    empty_file_name_string = 0
    file_name_counts: Dict[str, int] = defaultdict(int)
    examples: List[Tuple[str, str]] = []
    
    # Breakdown analysis for missing file_name nodes
    missing_nodes_by_content_type: Dict[str, int] = defaultdict(int)
    missing_nodes_with_source_path = 0
    missing_nodes_with_gcs_path = 0
    missing_nodes_with_file_path = 0
    missing_nodes_repairable = 0
    missing_node_examples: List[Dict[str, Any]] = []
    
    for i, (node_id, node_data) in enumerate(nodes.items()):
        meta = get_node_metadata(node_data)
        if "file_name" not in meta:
            missing_file_name_key += 1
            fn = ""
            
            # Analyze missing node for diagnostics
            content_type = meta.get("content_type", "unknown")
            missing_nodes_by_content_type[content_type] += 1
            
            # Check if repairable
            has_source = bool(meta.get("source_path"))
            has_gcs = bool(meta.get("gcs_path"))
            has_file_path = bool(meta.get("file_path"))
            
            if has_source:
                missing_nodes_with_source_path += 1
            if has_gcs:
                missing_nodes_with_gcs_path += 1
            if has_file_path:
                missing_nodes_with_file_path += 1
            
            if has_source or has_gcs or has_file_path:
                missing_nodes_repairable += 1
            
            # Collect examples (first 5)
            if len(missing_node_examples) < 5:
                available_keys = [k for k in meta.keys() if k not in ["text", "content"]][:10]
                missing_node_examples.append({
                    "node_id": node_id[:30],
                    "content_type": content_type,
                    "available_keys": available_keys,
                    "has_source_path": has_source,
                    "has_gcs_path": has_gcs,
                    "has_file_path": has_file_path,
                })
        else:
            fn = str(meta.get("file_name", "") or "")
            if not fn:
                empty_file_name_string += 1
        
        if fn:
            chunk_files_raw.add(fn)
            chunk_files_canonical.add(canonicalize_filename(fn))
            chunk_files_base.add(normalize_filename(fn))
            file_name_counts[fn] += 1
        
        if i < sample:
            examples.append((node_id, fn or "MISSING"))
    
    # DB filenames (optional - skip if DB not available)
    docs, db_error = load_db_documents()
    if db_error:
        return (
            CheckResult(
                "Filename alignment",
                "WARN",
                "DB unavailable",
                f"Cannot compare with DB: {db_error}. Showing chunk-only analysis."
            ),
            {
                "docstore_sample": examples,
                "chunk_files_raw": sorted(list(chunk_files_raw))[:20],
                "chunk_files_canonical": sorted(list(chunk_files_canonical))[:20],
                "missing_file_name_key": missing_file_name_key,
                "empty_file_name_string": empty_file_name_string,
                "db_error": db_error
            },
        )
    
    # DB filenames - use canonical for comparison
    db_files_raw = {getattr(d, "file_name", "") or "" for d in docs}
    db_files_canonical = {canonicalize_filename(x) for x in db_files_raw if x}
    db_files_base = {normalize_filename(x) for x in db_files_raw}
    
    # Also check display_name if present
    db_display_canonical = set()
    for d in docs:
        display = getattr(d, "display_name", None)
        if display:
            db_display_canonical.add(canonicalize_filename(display))
    
    # Combine DB canonical (file_name + display_name)
    db_all_canonical = db_files_canonical | db_display_canonical
    
    # Comparisons
    raw_db_only = sorted(db_files_raw - chunk_files_raw)
    raw_chunk_only = sorted(chunk_files_raw - db_files_raw)
    canonical_db_only = sorted(db_all_canonical - chunk_files_canonical)
    canonical_chunk_only = sorted(chunk_files_canonical - db_all_canonical)
    base_db_only = sorted(db_files_base - chunk_files_base)
    base_chunk_only = sorted(chunk_files_base - db_files_base)
    
    # Intersections
    raw_intersection = len(db_files_raw.intersection(chunk_files_raw))
    canonical_intersection = len(db_all_canonical.intersection(chunk_files_canonical))
    base_intersection = len(db_files_base.intersection(chunk_files_base))
    
    # Calculate match rate
    total_nodes = len(nodes)
    nodes_with_valid_filename = total_nodes - missing_file_name_key - empty_file_name_string
    match_rate = canonical_intersection / max(len(db_all_canonical), 1) if db_all_canonical else 0
    
    # Status determination
    status = "PASS"
    notes_parts = []
    
    if missing_file_name_key > 0:
        notes_parts.append(f"missing_key={missing_file_name_key}")
    if empty_file_name_string > 0:
        notes_parts.append(f"empty_string={empty_file_name_string}")
    
    notes_parts.append(f"canonical_intersection={canonical_intersection}/{len(db_all_canonical)}")
    notes_parts.append(f"match_rate={match_rate:.1%}")
    
    if missing_file_name_key > total_nodes * 0.05:  # >5% missing
        status = "FAIL"
        notes_parts.append("(>5% missing file_name key)")
    elif empty_file_name_string > total_nodes * 0.05:  # >5% empty
        status = "FAIL"
        notes_parts.append("(>5% empty file_name)")
    elif canonical_intersection == 0 and len(db_all_canonical) > 0:
        status = "FAIL"
        notes_parts.append("(zero canonical matches)")
    elif canonical_intersection < len(db_all_canonical) * 0.95:  # <95% match
        status = "WARN"
        notes_parts.append("(<95% canonical match)")
    
    # Top offending filenames (chunks not in DB)
    top_offending = sorted(
        [(fn, count) for fn, count in file_name_counts.items() if canonicalize_filename(fn) not in db_all_canonical],
        key=lambda x: x[1],
        reverse=True
    )[:25]
    
    return (
        CheckResult(
            name="Filename alignment",
            status=status,
            key_numbers=f"DB={len(db_files_raw)} chunks={len(chunk_files_raw)} missing_key={missing_file_name_key} empty={empty_file_name_string}",
            notes=" | ".join(notes_parts),
        ),
        {
            "docstore_sample": examples,
            "raw_db_only": raw_db_only[:20],
            "raw_chunk_only": raw_chunk_only[:20],
            "canonical_db_only": canonical_db_only[:20],
            "canonical_chunk_only": canonical_chunk_only[:20],
            "raw_intersection": raw_intersection,
            "canonical_intersection": canonical_intersection,
            "base_intersection": base_intersection,
            "missing_file_name_key": missing_file_name_key,
            "empty_file_name_string": empty_file_name_string,
            "top_offending_filenames": top_offending,
            "match_rate": match_rate,
            "missing_nodes_by_content_type": dict(missing_nodes_by_content_type),
            "missing_nodes_repairable": missing_nodes_repairable,
            "missing_nodes_with_source_path": missing_nodes_with_source_path,
            "missing_nodes_with_gcs_path": missing_nodes_with_gcs_path,
            "missing_nodes_with_file_path": missing_nodes_with_file_path,
            "missing_node_examples": missing_node_examples,
        },
    )


def compute_allowed_filenames(
    docs: List[Any],
    role: str,
    user_machines: List[str],
) -> Set[str]:
    """
    Implement the same document-level allowlist logic as HybridRetriever._get_allowed_filenames().
    
    Uses canonical filenames for consistent matching.
    
    ADMIN/TECH: include active docs; docs with empty machine_model are still allowed
    CUSTOMER: allow only docs whose Document.machine_model includes GENERAL/Any or intersects user_machine_models
    Always exclude inactive docs (is_active=False)
    """
    try:
        from backend.utils.filenames import canonicalize_filename
    except ImportError:
        def canonicalize_filename(name: str) -> str:
            return normalize_filename(name)
    
    allowed: Set[str] = set()
    role = role.upper()
    
    for d in docs:
        file_name = getattr(d, "file_name", None) or ""
        display_name = getattr(d, "display_name", None) or ""
        is_active = getattr(d, "is_active", True)
        if not is_active:
            continue
        
        # Canonicalize for consistent matching
        canonical_file_name = canonicalize_filename(file_name) if file_name else ""
        canonical_display = canonicalize_filename(display_name) if display_name else ""
        
        # For ADMIN/TECH, allow all active docs.
        if role in ("ADMIN", "TECHNICIAN"):
            if canonical_file_name:
                allowed.add(canonical_file_name)
            if canonical_display:
                allowed.add(canonical_display)
            # Also add original for migration tolerance
            if file_name:
                allowed.add(file_name)
            continue
        
        # CUSTOMER role logic:
        mm_field = getattr(d, "machine_model", None)
        doc_machines = parse_machine_model_field(mm_field)
        
        # Empty machine_model → not visible to customers
        if not doc_machines:
            continue
        
        # GENERAL/Any → visible to all
        if intersect_case_insensitive(doc_machines, ["GENERAL", "Any"]):
            allowed.add(file_name)
            continue
        
        # Otherwise must intersect with user's effective machines
        if intersect_case_insensitive(doc_machines, user_machines):
            allowed.add(file_name)
    
    return allowed


def check_machine_split_brain(storage_dir: Path, role: str, user_machines: List[str]) -> Tuple[CheckResult, Dict[str, Any]]:
    """Check (3): Machine filtering split-brain (document-level vs chunk-level)."""
    docstore_path = storage_dir / "docstore.json"
    if not docstore_path.exists():
        return (CheckResult("Machine filtering (doc vs chunk)", "FAIL", "docstore.json missing", ""), {})
    
    docs, db_error = load_db_documents()
    if db_error:
        return (
            CheckResult(
                "Machine filtering (doc vs chunk)",
                "WARN",
                "DB unavailable",
                f"Cannot check machine filtering: {db_error}. Skipping document-level filter check."
            ),
            {"db_error": db_error},
        )
    
    allowed = compute_allowed_filenames(docs, role=role, user_machines=user_machines)
    
    docstore = load_json(docstore_path)
    nodes = get_docstore_nodes(docstore)
    
    # Use canonical filenames for comparison
    try:
        from backend.utils.filenames import canonicalize_filename
    except ImportError:
        def canonicalize_filename(name: str) -> str:
            return normalize_filename(name)
    
    # How many chunks survive allowed filename filter?
    total = 0
    in_allowed = 0
    empty_filename_count = 0
    
    for _, node_data in nodes.items():
        meta = get_node_metadata(node_data)
        fn = str(meta.get("file_name", "") or "")
        total += 1
        
        if not fn:
            empty_filename_count += 1
            # For ADMIN/TECH: allow but count
            # For CUSTOMER: drop (already filtered)
            if role.upper() in ["ADMIN", "TECHNICIAN"]:
                in_allowed += 1
            continue
        
        # Canonicalize for comparison
        canonical_fn = canonicalize_filename(fn)
        
        # Check if canonical or original is in allowed set
        if canonical_fn in allowed or fn in allowed:
            in_allowed += 1
    
    if role.upper() == "CUSTOMER" and len(allowed) == 0:
        # Show a few docs as evidence
        sample_docs = []
        for d in docs[:10]:
            sample_docs.append(
                {
                    "file_name": getattr(d, "file_name", None),
                    "display_name": getattr(d, "display_name", None),
                    "machine_model": getattr(d, "machine_model", None),
                    "is_active": getattr(d, "is_active", None),
                }
            )
        return (
            CheckResult(
                name="Machine filtering (doc vs chunk)",
                status="FAIL",
                key_numbers=f"allowed_filenames=0 total_chunks={total}",
                notes="CUSTOMER allowed_filenames is empty → everything will be filtered out",
            ),
            {"sample_docs": sample_docs},
        )
    
    # Calculate match rate
    match_rate = in_allowed / max(total, 1)
    
    # Status determination with stricter rules for ADMIN
    status = "PASS"
    notes_parts = [f"allowed_filenames={len(allowed)} chunks_in_allowed={in_allowed}/{total}"]
    
    if empty_filename_count > 0:
        notes_parts.append(f"empty_filename={empty_filename_count}")
    
    if len(allowed) > 0 and in_allowed == 0:
        status = "FAIL"
        notes_parts.append("(allowed filenames exist but match zero chunks → filename mismatch likely)")
    elif role.upper() == "ADMIN" and match_rate < 0.95:  # ADMIN: FAIL if <95% match
        status = "FAIL"
        notes_parts.append(f"(ADMIN match_rate={match_rate:.1%} < 95% threshold)")
    elif role.upper() == "CUSTOMER" and total > 0 and match_rate < 0.05:
        status = "WARN"
        notes_parts.append("(very low chunk match rate for CUSTOMER)")
    
    return (
        CheckResult("Machine filtering (doc vs chunk)", status, f"{in_allowed}/{total} (match_rate={match_rate:.1%})", " | ".join(notes_parts)),
        {"allowed_count": len(allowed), "chunks_in_allowed": in_allowed, "total_chunks": total, "match_rate": match_rate, "empty_filename_count": empty_filename_count},
    )


def check_chunk_machine_model_ids(storage_dir: Path) -> Tuple[CheckResult, Dict[str, Any]]:
    """Check (4): Chunk machine_model_ids health check."""
    docstore_path = storage_dir / "docstore.json"
    if not docstore_path.exists():
        return (CheckResult("Chunk machine_model_ids", "FAIL", "docstore.json missing", ""), {})
    
    docstore = load_json(docstore_path)
    nodes = get_docstore_nodes(docstore)
    if not nodes:
        return (CheckResult("Chunk machine_model_ids", "FAIL", "0 nodes", ""), {})
    
    missing = 0
    empty = 0
    non_list = 0
    lengths = Counter()
    
    for _, node_data in nodes.items():
        meta = get_node_metadata(node_data)
        if "machine_model_ids" not in meta:
            missing += 1
            continue
        ids = meta.get("machine_model_ids")
        if not isinstance(ids, list):
            non_list += 1
            continue
        if len(ids) == 0:
            empty += 1
        lengths[len(ids)] += 1
    
    total = len(nodes)
    status = "PASS"
    notes = f"missing={missing}, empty={empty}, non_list={non_list}"
    if missing > 0 or non_list > 0:
        status = "WARN"
    # Treat massive empties as warn (or fail depending on role; role-specific handled elsewhere)
    if empty / max(total, 1) > 0.5:
        status = "WARN"
        notes += " (>50% empty lists)"
    
    # Compact distribution preview
    dist_preview = ", ".join([f"{k}:{v}" for k, v in sorted(lengths.items())[:10]])
    
    return (
        CheckResult("Chunk machine_model_ids", status, f"total_nodes={total}", notes),
        {"length_distribution_preview": dist_preview},
    )


def get_vector_store_node_ids(vector_store_path: Path) -> Set[str]:
    """
    Parse default__vector_store.json to extract all node_ids.
    
    LlamaIndex SimpleVectorStore format:
      {"embedding_dict": {node_id: [vector], ...}}
      or nested under keys like "data" / "embedding_dict"
    """
    if not vector_store_path.exists():
        return set()
    
    try:
        data = load_json(vector_store_path)
        
        # Try direct embedding_dict
        if "embedding_dict" in data and isinstance(data["embedding_dict"], dict):
            return set(data["embedding_dict"].keys())
        
        # Try nested under data
        if "data" in data and isinstance(data["data"], dict):
            if "embedding_dict" in data["data"]:
                return set(data["data"]["embedding_dict"].keys())
        
        # Try other common keys
        for key in ["embeddings", "vectors", "vector_dict"]:
            if key in data and isinstance(data[key], dict):
                return set(data[key].keys())
        
        # If top-level is a dict of node_id -> vector, use keys
        if isinstance(data, dict) and len(data) > 0:
            # Check if first value looks like a vector (list of numbers)
            first_val = next(iter(data.values()))
            if isinstance(first_val, list) and len(first_val) > 0:
                if isinstance(first_val[0], (int, float)):
                    return set(data.keys())
        
        return set()
    except Exception as e:
        safe_print(f"   ⚠️ Failed to parse vector store: {type(e).__name__}: {e}")
        return set()


def get_node_text(node_data: Dict[str, Any]) -> Optional[str]:
    """Extract text from node data, handling various LlamaIndex formats."""
    # Check __data__ wrapper
    if "__data__" in node_data:
        inner = node_data["__data__"]
        if isinstance(inner, dict):
            text = inner.get("text")
            if isinstance(text, str):
                return text
    
    # Direct text key
    text = node_data.get("text")
    if isinstance(text, str):
        return text
    
    # Try _node_data
    if "_node_data" in node_data:
        inner = node_data["_node_data"]
        if isinstance(inner, dict):
            text = inner.get("text")
            if isinstance(text, str):
                return text
    
    return None


def check_storage_resolution(storage_dir_arg: Optional[str], download_from_gcs: bool = False) -> Tuple[CheckResult, Dict[str, Any], Path]:
    """Check (5): Storage directory resolution consistency."""
    resolved = None
    settings_dir = None
    gcs_bucket = None
    gcs_prefix = None
    
    # Optional imports (don't fail if missing)
    try:
        from backend.utils.storage_path import resolve_storage_path  # type: ignore
        resolved_path = resolve_storage_path()
        resolved = str(resolved_path) if resolved_path else None
    except Exception:
        resolved = None
    
    # Read environment variables directly to avoid Settings initialization
    settings_dir = os.getenv("RAG_INDEX_LOCAL_DIR", None)
    gcs_bucket = os.getenv("RAG_INDEX_GCS_BUCKET", "arrow-rag-support-prod-rag")
    raw_prefix = os.getenv("RAG_INDEX_GCS_PREFIX", "latest_model/")
    raw_prefix = (raw_prefix or "").strip()
    if raw_prefix:
        normalized = raw_prefix.strip("/")
        gcs_prefix = f"{normalized}/" if normalized else ""
    else:
        gcs_prefix = ""
    
    # If downloading from GCS, use a temp directory or configured local dir
    if download_from_gcs:
        if settings_dir:
            chosen = Path(settings_dir)
        else:
            import tempfile
            chosen = Path(tempfile.mkdtemp(prefix="rag_diagnosis_"))
            print(f"[DIAG] Using temporary directory: {chosen}", flush=True)
    else:
        chosen = Path(storage_dir_arg) if storage_dir_arg else Path(resolved or "latest_model")
    
    notes_parts = []
    if settings_dir is not None:
        notes_parts.append(f"settings.RAG_INDEX_LOCAL_DIR={settings_dir}")
    if resolved is not None:
        notes_parts.append(f"resolve_storage_path()={resolved}")
    if download_from_gcs and gcs_bucket:
        notes_parts.append(f"GCS source: gs://{gcs_bucket}/{gcs_prefix or ''}")
    notes_parts.append(f"chosen={str(chosen)}")
    
    status = "PASS"
    if resolved and storage_dir_arg and Path(resolved) != Path(storage_dir_arg):
        status = "WARN"
        notes_parts.append("chosen != resolve_storage_path (verify ingest & query use same dir)")
    
    return (
        CheckResult("Storage path resolution", status, "-", " | ".join(notes_parts)),
        {"settings_dir": settings_dir, "resolved": resolved, "chosen": str(chosen), "gcs_bucket": gcs_bucket, "gcs_prefix": gcs_prefix},
        chosen,
    )


def check_docstore_vector_join(storage_dir: Path) -> Tuple[CheckResult, Dict[str, Any]]:
    """Check (6): Docstore ↔ vector store node_id join consistency."""
    docstore_path = storage_dir / "docstore.json"
    vector_store_path = storage_dir / "default__vector_store.json"
    
    if not docstore_path.exists():
        return (
            CheckResult("Docstore ↔ Vector join", "FAIL", "docstore.json missing", ""),
            {},
        )
    
    if not vector_store_path.exists():
        return (
            CheckResult("Docstore ↔ Vector join", "FAIL", "default__vector_store.json missing", ""),
            {},
        )
    
    try:
        docstore = load_json(docstore_path)
        docstore_nodes = get_docstore_nodes(docstore)
        docstore_ids = set(docstore_nodes.keys())
        
        vector_ids = get_vector_store_node_ids(vector_store_path)
        
        docstore_only = sorted(docstore_ids - vector_ids)
        vector_only = sorted(vector_ids - docstore_ids)
        
        total_docstore = len(docstore_ids)
        total_vector = len(vector_ids)
        intersection = len(docstore_ids & vector_ids)
        
        docstore_diff_rate = len(docstore_only) / max(total_docstore, 1)
        vector_diff_rate = len(vector_only) / max(total_vector, 1)
        
        status = "PASS"
        notes = f"intersection={intersection}, docstore={total_docstore}, vector={total_vector}"
        
        if docstore_diff_rate > 0.005 or vector_diff_rate > 0.005:  # 0.5% threshold
            status = "FAIL"
            notes += f" (drift >0.5%: docstore_only={len(docstore_only)}, vector_only={len(vector_only)})"
        elif docstore_diff_rate > 0.001 or vector_diff_rate > 0.001:  # 0.1% threshold
            status = "WARN"
            notes += f" (small drift: docstore_only={len(docstore_only)}, vector_only={len(vector_only)})"
        
        return (
            CheckResult(
                "Docstore ↔ Vector join",
                status,
                f"docstore={total_docstore} vector={total_vector} intersection={intersection}",
                notes,
            ),
            {
                "docstore_count": total_docstore,
                "vector_count": total_vector,
                "intersection": intersection,
                "docstore_only": docstore_only[:20],
                "vector_only": vector_only[:20],
                "docstore_only_count": len(docstore_only),
                "vector_only_count": len(vector_only),
            },
        )
    except Exception as e:
        return (
            CheckResult("Docstore ↔ Vector join", "FAIL", f"Parse error: {type(e).__name__}", str(e)),
            {},
        )


def check_embedding_dimensions(storage_dir: Path) -> Tuple[CheckResult, Dict[str, Any]]:
    """Check (7): Embedding dimension sanity."""
    vector_store_path = storage_dir / "default__vector_store.json"
    
    if not vector_store_path.exists():
        return (
            CheckResult("Embedding dimensions", "FAIL", "default__vector_store.json missing", ""),
            {},
        )
    
    try:
        data = load_json(vector_store_path)
        
        # Extract embedding_dict
        embedding_dict = None
        if "embedding_dict" in data:
            embedding_dict = data["embedding_dict"]
        elif "data" in data and "embedding_dict" in data["data"]:
            embedding_dict = data["data"]["embedding_dict"]
        elif isinstance(data, dict) and len(data) > 0:
            first_val = next(iter(data.values()))
            if isinstance(first_val, list) and len(first_val) > 0 and isinstance(first_val[0], (int, float)):
                embedding_dict = data
        
        if not embedding_dict or not isinstance(embedding_dict, dict):
            return (
                CheckResult("Embedding dimensions", "FAIL", "Could not find embedding_dict", ""),
                {},
            )
        
        # Sample 20 vectors (or all if fewer)
        sample_size = min(20, len(embedding_dict))
        sample_ids = list(embedding_dict.keys())[:sample_size]
        
        dimensions = []
        non_numeric = 0
        has_nan = 0
        has_inf = 0
        
        for node_id in sample_ids:
            vec = embedding_dict[node_id]
            if not isinstance(vec, list):
                non_numeric += 1
                continue
            
            dim = len(vec)
            dimensions.append(dim)
            
            # Check for non-numeric, NaN, or Inf
            for val in vec:
                if not isinstance(val, (int, float)):
                    non_numeric += 1
                    break
                if math.isnan(val):
                    has_nan += 1
                    break
                if math.isinf(val):
                    has_inf += 1
                    break
        
        if not dimensions:
            return (
                CheckResult("Embedding dimensions", "FAIL", "No valid vectors found", ""),
                {},
            )
        
        unique_dims = set(dimensions)
        expected_dim = dimensions[0] if dimensions else None
        
        # Check expected dimension from config (if available)
        expected_config_dim = None
        try:
            from backend.config.env import settings
            # Common embedding model dimensions
            embed_model = getattr(settings, "EMBEDDING_MODEL", None) or os.getenv("EMBEDDING_MODEL", "")
            if "bge-large" in embed_model.lower() or "bge_large" in embed_model.lower():
                expected_config_dim = 1024
            elif "bge-base" in embed_model.lower() or "bge_base" in embed_model.lower():
                expected_config_dim = 768
            elif "sentence-transformers" in embed_model.lower():
                # Default for most sentence-transformers
                expected_config_dim = 384
        except Exception:
            pass
        
        status = "PASS"
        notes = f"dim={expected_dim}, sampled={sample_size}"
        
        if len(unique_dims) > 1:
            status = "FAIL"
            notes += f" (inconsistent dimensions: {unique_dims})"
        elif expected_config_dim and expected_dim != expected_config_dim:
            status = "WARN"
            notes += f" (expected {expected_config_dim} from config, got {expected_dim})"
        
        if non_numeric > 0:
            status = "FAIL"
            notes += f" (non-numeric values: {non_numeric})"
        
        if has_nan > 0:
            status = "FAIL"
            notes += f" (NaN values: {has_nan})"
        
        if has_inf > 0:
            status = "WARN"
            notes += f" (Inf values: {has_inf})"
        
        return (
            CheckResult("Embedding dimensions", status, f"dim={expected_dim}", notes),
            {
                "dimension": expected_dim,
                "unique_dimensions": sorted(unique_dims),
                "sample_size": sample_size,
                "non_numeric": non_numeric,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "expected_config_dim": expected_config_dim,
            },
        )
    except Exception as e:
        return (
            CheckResult("Embedding dimensions", "FAIL", f"Parse error: {type(e).__name__}", str(e)),
            {},
        )


def check_text_presence(storage_dir: Path) -> Tuple[CheckResult, Dict[str, Any]]:
    """Check (8): Text presence / non-empty content."""
    docstore_path = storage_dir / "docstore.json"
    
    if not docstore_path.exists():
        return (
            CheckResult("Text presence", "FAIL", "docstore.json missing", ""),
            {},
        )
    
    try:
        docstore = load_json(docstore_path)
        nodes = get_docstore_nodes(docstore)
        
        if not nodes:
            return (
                CheckResult("Text presence", "FAIL", "0 nodes", ""),
                {},
            )
        
        missing_text = 0
        empty_text = 0
        very_short_text = 0
        text_lengths = []
        
        for node_id, node_data in nodes.items():
            text = get_node_text(node_data)
            
            if text is None:
                missing_text += 1
            elif not text.strip():
                empty_text += 1
            else:
                text_lengths.append(len(text))
                if len(text.strip()) < 20:
                    very_short_text += 1
        
        total = len(nodes)
        empty_rate = (missing_text + empty_text) / max(total, 1)
        
        status = "PASS"
        notes = f"missing={missing_text}, empty={empty_text}, very_short={very_short_text}"
        
        if empty_rate > 0.05:  # 5% threshold
            status = "FAIL"
            notes += f" ({empty_rate:.1%} empty - exceeds 5%)"
        elif empty_rate > 0.01:  # 1% threshold
            status = "WARN"
            notes += f" ({empty_rate:.1%} empty - exceeds 1%)"
        
        avg_length = sum(text_lengths) / max(len(text_lengths), 1) if text_lengths else 0
        
        return (
            CheckResult("Text presence", status, f"total={total} empty={missing_text+empty_text}", notes),
            {
                "total_nodes": total,
                "missing_text": missing_text,
                "empty_text": empty_text,
                "very_short_text": very_short_text,
                "empty_rate": empty_rate,
                "avg_text_length": avg_length,
                "min_text_length": min(text_lengths) if text_lengths else 0,
                "max_text_length": max(text_lengths) if text_lengths else 0,
            },
        )
    except Exception as e:
        return (
            CheckResult("Text presence", "FAIL", f"Parse error: {type(e).__name__}", str(e)),
            {},
        )


def check_metadata_types(storage_dir: Path) -> Tuple[CheckResult, Dict[str, Any]]:
    """Check (9): Required metadata type correctness."""
    docstore_path = storage_dir / "docstore.json"
    
    if not docstore_path.exists():
        return (
            CheckResult("Metadata types", "FAIL", "docstore.json missing", ""),
            {},
        )
    
    try:
        docstore = load_json(docstore_path)
        nodes = get_docstore_nodes(docstore)
        
        if not nodes:
            return (
                CheckResult("Metadata types", "FAIL", "0 nodes", ""),
                {},
            )
        
        non_list_machine_ids = 0
        non_int_entries = 0
        missing_content_type = 0
        missing_page_label = 0
        
        for node_id, node_data in nodes.items():
            meta = get_node_metadata(node_data)
            
            # Check machine_model_ids
            mm_ids = meta.get("machine_model_ids")
            if mm_ids is not None:
                if not isinstance(mm_ids, list):
                    non_list_machine_ids += 1
                else:
                    for entry in mm_ids:
                        if not isinstance(entry, int):
                            non_int_entries += 1
            
            # Check content_type
            if "content_type" not in meta:
                missing_content_type += 1
            
            # Check page_label (optional but common)
            if "page_label" not in meta:
                missing_page_label += 1
        
        total = len(nodes)
        
        status = "PASS"
        notes = f"non_list_mm_ids={non_list_machine_ids}, non_int_entries={non_int_entries}"
        
        if non_list_machine_ids > 0:
            status = "FAIL"
            notes += " (machine_model_ids must be list)"
        
        if non_int_entries > 0:
            if status == "PASS":
                status = "WARN"
            notes += f" (non-int entries in machine_model_ids: {non_int_entries})"
        
        return (
            CheckResult("Metadata types", status, f"total={total}", notes),
            {
                "total_nodes": total,
                "non_list_machine_ids": non_list_machine_ids,
                "non_int_entries": non_int_entries,
                "missing_content_type": missing_content_type,
                "missing_page_label": missing_page_label,
            },
        )
    except Exception as e:
        return (
            CheckResult("Metadata types", "FAIL", f"Parse error: {type(e).__name__}", str(e)),
            {},
        )


def resolve_machine_names_to_ids(machine_names: List[str]) -> Tuple[Dict[str, int], List[str]]:
    """
    Resolve machine model names to IDs from database.
    
    Returns:
        (resolved: {name -> id}, unmatched: [names])
    """
    resolved = {}
    unmatched = []
    
    if not machine_names:
        return resolved, unmatched
    
    try:
        from backend.utils.db import SessionLocal, MachineModel  # type: ignore
        session = SessionLocal()
        try:
            all_machines = session.query(MachineModel).all()
            
            # Normalize input names for matching
            normalized_input = {}
            for name in machine_names:
                normalized = name.strip().lower().replace(" ", "_").replace("-", "_")
                normalized_input[normalized] = name
            
            # Build lookup: normalized_name -> (id, original_name)
            machine_lookup = {}
            for mm in all_machines:
                db_name = mm.name or ""
                normalized_db = db_name.strip().lower().replace(" ", "_").replace("-", "_")
                machine_lookup[normalized_db] = (mm.id, db_name)
            
            # Match input to DB
            for normalized, original in normalized_input.items():
                matched = False
                # Exact match
                if normalized in machine_lookup:
                    mm_id, db_name = machine_lookup[normalized]
                    resolved[original] = mm_id
                    matched = True
                else:
                    # Fuzzy match: check if normalized contains or is contained
                    for norm_db, (mm_id, db_name) in machine_lookup.items():
                        if normalized in norm_db or norm_db in normalized:
                            resolved[original] = mm_id
                            matched = True
                            break
                
                if not matched:
                    unmatched.append(original)
            
            return resolved, unmatched
        finally:
            session.close()
    except Exception as e:
        safe_print(f"   ⚠️ Failed to resolve machine names: {type(e).__name__}: {e}")
        return {}, machine_names


def get_machine_model_name_map() -> Dict[int, str]:
    """Get mapping of machine_model_id -> name from database."""
    try:
        from backend.utils.db import SessionLocal, MachineModel  # type: ignore
        session = SessionLocal()
        try:
            machines = session.query(MachineModel).all()
            return {mm.id: mm.name or f"Unknown_{mm.id}" for mm in machines}
        finally:
            session.close()
    except Exception:
        return {}


def get_top_machine_model_candidates(unmatched_names: List[str], limit: int = 10) -> List[Tuple[str, int]]:
    """Get top N closest machine model names from DB for unmatched input names."""
    try:
        from backend.utils.db import SessionLocal, MachineModel  # type: ignore
        session = SessionLocal()
        try:
            all_machines = session.query(MachineModel).all()
            candidates = []
            
            for unmatched in unmatched_names:
                normalized_unmatched = unmatched.strip().lower().replace(" ", "_").replace("-", "_")
                for mm in all_machines:
                    db_name = mm.name or ""
                    normalized_db = db_name.strip().lower().replace(" ", "_").replace("-", "_")
                    # Simple similarity: count common substrings
                    similarity = 0
                    if normalized_unmatched in normalized_db or normalized_db in normalized_unmatched:
                        similarity = min(len(normalized_unmatched), len(normalized_db))
                    candidates.append((db_name, mm.id, similarity))
            
            # Sort by similarity (descending) and return top N
            candidates.sort(key=lambda x: x[2], reverse=True)
            return [(name, mm_id) for name, mm_id, _ in candidates[:limit]]
        finally:
            session.close()
    except Exception:
        return []


def check_machine_model_distribution(storage_dir: Path) -> Tuple[CheckResult, Dict[str, Any]]:
    """Check chunk machine_model_ids distribution (top 10)."""
    docstore_path = storage_dir / "docstore.json"
    
    if not docstore_path.exists():
        return (
            CheckResult("Machine model distribution", "FAIL", "docstore.json missing", ""),
            {},
        )
    
    try:
        docstore = load_json(docstore_path)
        nodes = get_docstore_nodes(docstore)
        
        if not nodes:
            return (
                CheckResult("Machine model distribution", "FAIL", "0 nodes", ""),
                {},
            )
        
        # Count machine_model_ids across all nodes
        mm_id_counts: Counter = Counter()
        total_with_ids = 0
        
        for node_id, node_data in nodes.items():
            meta = get_node_metadata(node_data)
            mm_ids = meta.get("machine_model_ids")
            if isinstance(mm_ids, list):
                for mm_id in mm_ids:
                    if isinstance(mm_id, int):
                        mm_id_counts[mm_id] += 1
                        total_with_ids += 1
        
        # Get top 10
        top_10_ids = mm_id_counts.most_common(10)
        
        # Map IDs to names if DB available
        id_to_name = get_machine_model_name_map()
        top_10_with_names = [
            (mm_id, count, id_to_name.get(mm_id, f"Unknown_{mm_id}"))
            for mm_id, count in top_10_ids
        ]
        
        return (
            CheckResult(
                "Machine model distribution",
                "PASS",
                f"total_with_ids={total_with_ids}",
                f"top_10_ids={len(top_10_ids)}",
            ),
            {
                "total_with_ids": total_with_ids,
                "top_10": top_10_with_names,
                "unique_machine_ids": len(mm_id_counts),
            },
        )
    except Exception as e:
        return (
            CheckResult("Machine model distribution", "FAIL", f"Parse error: {type(e).__name__}", str(e)),
            {},
        )


def check_customer_visibility(storage_dir: Path, role: str, user_machines: List[str]) -> Tuple[CheckResult, Dict[str, Any]]:
    """Check (10): Customer visibility simulation."""
    if role.upper() != "CUSTOMER":
        return (
            CheckResult("Customer visibility", "SKIP", f"role={role}", "Only runs for CUSTOMER role"),
            {},
        )
    
    docstore_path = storage_dir / "docstore.json"
    if not docstore_path.exists():
        return (
            CheckResult("Customer visibility", "FAIL", "docstore.json missing", ""),
            {},
        )
    
    try:
        # Load DB documents
        docs, db_error = load_db_documents()
        if not docs:
            return (
                CheckResult("Customer visibility", "WARN", "DB unavailable", f"Cannot check without DB: {db_error or 'no documents'}"),
                {},
            )
        
        # Resolve machine names to IDs
        resolved_machines, unmatched = resolve_machine_names_to_ids(user_machines)
        resolved_ids = list(resolved_machines.values())
        
        # Compute allowed filenames (same logic as orchestrator)
        allowed_filenames = compute_allowed_filenames(docs, role=role, user_machines=user_machines)
        
        # Load docstore
        docstore = load_json(docstore_path)
        nodes = get_docstore_nodes(docstore)
        
        # Count visible chunks
        total_chunks = 0
        visible_chunks = 0
        excluded_inactive = 0
        excluded_machine_mismatch = 0
        excluded_missing_metadata = 0
        
        # Build filename -> is_active map from DB
        filename_to_active = {}
        filename_to_machines = {}
        for d in docs:
            fn = getattr(d, "file_name", None) or ""
            filename_to_active[fn] = getattr(d, "is_active", True)
            mm_field = getattr(d, "machine_model", None)
            filename_to_machines[fn] = parse_machine_model_field(mm_field)
        
        for node_id, node_data in nodes.items():
            total_chunks += 1
            meta = get_node_metadata(node_data)
            fn = str(meta.get("file_name", "") or "")
            
            # Check if excluded due to inactive doc
            if fn in filename_to_active and not filename_to_active[fn]:
                excluded_inactive += 1
                continue
            
            # Check if excluded due to filename not in allowed
            if fn not in allowed_filenames:
                excluded_machine_mismatch += 1
                continue
            
            # Check if excluded due to missing machine_model_ids
            mm_ids = meta.get("machine_model_ids")
            if not isinstance(mm_ids, list) or len(mm_ids) == 0:
                excluded_missing_metadata += 1
                continue
            
            # If we get here, chunk is visible
            visible_chunks += 1
        
        status = "PASS"
        notes = f"allowed_docs={len(allowed_filenames)}, visible_chunks={visible_chunks}/{total_chunks}"
        
        if len(allowed_filenames) > 0 and visible_chunks == 0:
            status = "FAIL"
            notes += " (allowed docs exist but zero visible chunks)"
        elif len(allowed_filenames) == 0:
            status = "FAIL"
            notes += " (no allowed documents for customer)"
        elif visible_chunks / max(total_chunks, 1) < 0.05:
            status = "WARN"
            notes += f" (very low visibility: {visible_chunks/total_chunks*100:.1f}%)"
        
        return (
            CheckResult("Customer visibility", status, f"visible={visible_chunks}/{total_chunks}", notes),
            {
                "allowed_documents": len(allowed_filenames),
                "total_chunks": total_chunks,
                "visible_chunks": visible_chunks,
                "excluded_inactive": excluded_inactive,
                "excluded_machine_mismatch": excluded_machine_mismatch,
                "excluded_missing_metadata": excluded_missing_metadata,
                "resolved_machine_ids": resolved_ids,
                "resolved_machine_names": resolved_machines,
                "unmatched_machine_names": unmatched,
            },
        )
    except Exception as e:
        return (
            CheckResult("Customer visibility", "FAIL", f"Error: {type(e).__name__}", str(e)),
            {},
        )


# ---------- Main ----------


def main() -> int:
    """Main diagnostic entry point."""
    parser = argparse.ArgumentParser(
        description="Diagnose ingestion/query contract mismatches for RAG index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ADMIN check (no customer gating) - using local index
  python -m backend.tools.diagnose_rag_contract --storage-dir latest_model --role ADMIN

  # CUSTOMER check with GCS download (production index)
  python -m backend.tools.diagnose_rag_contract --download-from-gcs --role CUSTOMER --user-machine "EZCut 330"
  
  # ADMIN check with GCS download
  python -m backend.tools.diagnose_rag_contract --download-from-gcs --role ADMIN
        """
    )
    parser.add_argument(
        "--storage-dir",
        default=None,
        help="Index directory (default: resolve_storage_path() or latest_model). Ignored if --download-from-gcs is used."
    )
    parser.add_argument(
        "--download-from-gcs",
        action="store_true",
        help="Download index from GCS before analysis (uses production index)"
    )
    parser.add_argument(
        "--role",
        default="ADMIN",
        choices=["ADMIN", "TECHNICIAN", "CUSTOMER"],
        help="User role for machine filtering checks"
    )
    parser.add_argument(
        "--user-machine",
        action="append",
        default=[],
        help="Repeatable. Used for CUSTOMER allowlist checks."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Number of sample nodes to print from docstore."
    )
    args = parser.parse_args()
    
    results: List[CheckResult] = []
    details: Dict[str, Any] = {}
    
    # (5) storage resolution
    r, d, chosen_dir = check_storage_resolution(args.storage_dir, download_from_gcs=args.download_from_gcs)
    results.append(r)
    details["storage_resolution"] = d
    
    storage_dir = chosen_dir
    if not storage_dir.is_absolute():
        # Keep behavior consistent: relative paths are relative to repo cwd
        storage_dir = (Path.cwd() / storage_dir).resolve()
    
    # Download from GCS if requested
    if args.download_from_gcs:
        print(f"\n{'='*80}")
        print("Downloading index from GCS for analysis...")
        print(f"{'='*80}\n")
        success, error = download_index_for_diagnosis(storage_dir)
        if not success:
            print(f"\n❌ Failed to download index from GCS: {error}")
            print("Cannot proceed with diagnosis without index files.")
            return 1
        print()  # Blank line after download
    
    safe_print(f"\nUsing storage_dir: {storage_dir}\n")
    
    # (1) index files
    r, d = check_index_files(storage_dir)
    results.append(r)
    details["index_files"] = d
    
    # (2) filename alignment
    r, d = check_filename_alignment(storage_dir, sample=args.sample)
    results.append(r)
    details["filename_alignment"] = d
    
    # Print filename mismatch details if any (enhanced diagnostics)
    if "missing_file_name_key" in d and d["missing_file_name_key"] > 0:
        missing_count = d["missing_file_name_key"]
        total_nodes = len(get_docstore_nodes(load_json(storage_dir / "docstore.json"))) if (storage_dir / "docstore.json").exists() else 1
        safe_print(f"\n ⚠️ Missing file_name key: {missing_count} nodes ({missing_count/max(total_nodes,1)*100:.1f}%)")
        
        # Breakdown by content_type
        if "missing_nodes_by_content_type" in d and d["missing_nodes_by_content_type"]:
            safe_print(f"\n   Breakdown by content_type:")
            for ct, count in sorted(d["missing_nodes_by_content_type"].items(), key=lambda x: x[1], reverse=True):
                safe_print(f"     {ct}: {count} nodes")
        
        # Repairability analysis
        if "missing_nodes_repairable" in d:
            repairable = d["missing_nodes_repairable"]
            safe_print(f"\n   Repairability analysis:")
            safe_print(f"     Repairable (has source_path/gcs_path/file_path): {repairable}/{missing_count} ({repairable/max(missing_count,1)*100:.1f}%)")
            if "missing_nodes_with_source_path" in d:
                safe_print(f"     Has source_path: {d['missing_nodes_with_source_path']}")
            if "missing_nodes_with_gcs_path" in d:
                safe_print(f"     Has gcs_path: {d['missing_nodes_with_gcs_path']}")
            if "missing_nodes_with_file_path" in d:
                safe_print(f"     Has file_path: {d['missing_nodes_with_file_path']}")
        
        # Example missing nodes
        if "missing_node_examples" in d and d["missing_node_examples"]:
            safe_print(f"\n   Example missing nodes (first 5):")
            for ex in d["missing_node_examples"]:
                safe_print(f"     node_id={ex.get('node_id')} content_type={ex.get('content_type')}")
                safe_print(f"       available_keys: {ex.get('available_keys', [])[:8]}")
                safe_print(f"       repairable: {ex.get('has_source_path') or ex.get('has_gcs_path') or ex.get('has_file_path')}")
    
    if "empty_file_name_string" in d and d["empty_file_name_string"] > 0:
        safe_print(f"\n ⚠️ Empty file_name string: {d['empty_file_name_string']} nodes")
    if "canonical_db_only" in d and d["canonical_db_only"]:
        safe_print(f"\n DB-only canonical filenames (top 20): {d['canonical_db_only']}")
    if "canonical_chunk_only" in d and d["canonical_chunk_only"]:
        safe_print(f"\n Chunk-only canonical filenames (top 20): {d['canonical_chunk_only']}")
    if "top_offending_filenames" in d and d["top_offending_filenames"]:
        safe_print(f"\n Top 25 offending filenames (chunks not in DB):")
        for fn, count in d["top_offending_filenames"]:
            safe_print(f"   {fn}: {count} chunks")
    if "match_rate" in d:
        safe_print(f"\n Canonical match rate: {d['match_rate']:.1%}")
    
    # (3) machine split-brain
    r, d = check_machine_split_brain(storage_dir, role=args.role, user_machines=args.user_machine)
    results.append(r)
    details["machine_split_brain"] = d
    
    # Print sample docs if CUSTOMER has empty allowed_filenames
    if args.role == "CUSTOMER" and d.get("sample_docs"):
        safe_print(f"\n Sample documents (showing why allowed_filenames is empty):")
        for doc in d["sample_docs"]:
            safe_print(f"   file_name={doc.get('file_name')}, display_name={doc.get('display_name')}, machine_model={doc.get('machine_model')}, is_active={doc.get('is_active')}")
    
    # Print machine filtering match rate
    if "match_rate" in d:
        match_rate = d["match_rate"]
        if args.role == "ADMIN" and match_rate < 0.95:
            safe_print(f"\n ❌ ADMIN match rate {match_rate:.1%} < 95% threshold - filename alignment issue")
        elif match_rate < 0.05:
            safe_print(f"\n ⚠️ Very low match rate {match_rate:.1%} - likely filename mismatch")
    
    # (4) chunk machine_model_ids
    r, d = check_chunk_machine_model_ids(storage_dir)
    results.append(r)
    details["chunk_machine_model_ids"] = d
    
    # (6) docstore ↔ vector store join
    r, d = check_docstore_vector_join(storage_dir)
    results.append(r)
    details["docstore_vector_join"] = d
    
    # (7) embedding dimensions
    r, d = check_embedding_dimensions(storage_dir)
    results.append(r)
    details["embedding_dimensions"] = d
    
    # (8) text presence
    r, d = check_text_presence(storage_dir)
    results.append(r)
    details["text_presence"] = d
    
    # (9) metadata types
    r, d = check_metadata_types(storage_dir)
    results.append(r)
    details["metadata_types"] = d
    
    # (10) customer visibility (if CUSTOMER role)
    r, d = check_customer_visibility(storage_dir, role=args.role, user_machines=args.user_machine)
    results.append(r)
    details["customer_visibility"] = d
    
    # Print samples with better diagnostics
    docstore_path = storage_dir / "docstore.json"
    if docstore_path.exists():
        try:
            docstore = load_json(docstore_path)
            nodes = get_docstore_nodes(docstore)
            safe_print("\nDocstore sample nodes:")
            for i, (node_id, node_data) in enumerate(list(nodes.items())[: args.sample]):
                meta = get_node_metadata(node_data)
                # Show raw keys if metadata is empty
                if not meta:
                    top_keys = [k for k in node_data.keys() if not k.startswith("_")][:10]
                    safe_print(
                        f"  {i+1}. node_id={node_id[:20]}... "
                        f"⚠️ No metadata found. Top-level keys: {top_keys}"
                    )
                    # Show a sample of the actual data structure for debugging
                    if i == 0:  # Only show for first node to avoid spam
                        sample_data = {k: str(v)[:50] for k, v in list(node_data.items())[:5]}
                        safe_print(f"      First node sample data: {sample_data}")
                else:
                    # Show actual values, not just "MISSING"
                    file_name = meta.get('file_name') if meta.get('file_name') is not None else 'MISSING'
                    machine_ids = meta.get('machine_model_ids') if meta.get('machine_model_ids') is not None else 'MISSING'
                    content_type = meta.get('content_type') if meta.get('content_type') is not None else 'MISSING'
                    page_label = meta.get('page_label') if meta.get('page_label') is not None else 'MISSING'
                    safe_print(
                        f"  {i+1}. node_id={node_id[:20]}... "
                        f"file_name={file_name} "
                        f"machine_model_ids={machine_ids} "
                        f"content_type={content_type} "
                        f"page_label={page_label}"
                    )
        except Exception as e:
            safe_print(f"\nDocstore sample nodes: failed to read ({type(e).__name__}: {e})")
    
    # Print detailed results for new checks
    if "docstore_vector_join" in details and details["docstore_vector_join"]:
        d = details["docstore_vector_join"]
        if d.get("docstore_only_count", 0) > 0:
            safe_print(f"\n  ⚠️ Docstore-only node IDs (top 10): {d.get('docstore_only', [])[:10]}")
        if d.get("vector_only_count", 0) > 0:
            safe_print(f"\n  ⚠️ Vector-only node IDs (top 10): {d.get('vector_only', [])[:10]}")
    
    if "embedding_dimensions" in details and details["embedding_dimensions"]:
        d = details["embedding_dimensions"]
        safe_print(f"\n  Embedding dimension: {d.get('dimension', 'unknown')}")
        if d.get("expected_config_dim"):
            safe_print(f"  Expected (from config): {d.get('expected_config_dim')}")
        if d.get("non_numeric", 0) > 0 or d.get("has_nan", 0) > 0:
            safe_print(f"  ⚠️ Vector quality issues: non_numeric={d.get('non_numeric', 0)}, NaN={d.get('has_nan', 0)}, Inf={d.get('has_inf', 0)}")
    
    if "text_presence" in details and details["text_presence"]:
        d = details["text_presence"]
        if d.get("empty_rate", 0) > 0:
            safe_print(f"\n  ⚠️ Text issues: missing={d.get('missing_text', 0)}, empty={d.get('empty_text', 0)}, very_short={d.get('very_short_text', 0)}")
            safe_print(f"  Text stats: avg_len={d.get('avg_text_length', 0):.0f}, min={d.get('min_text_length', 0)}, max={d.get('max_text_length', 0)}")
    
    if "metadata_types" in details and details["metadata_types"]:
        d = details["metadata_types"]
        if d.get("non_list_machine_ids", 0) > 0 or d.get("non_int_entries", 0) > 0:
            safe_print(f"\n  ⚠️ Metadata type issues: non_list_mm_ids={d.get('non_list_machine_ids', 0)}, non_int_entries={d.get('non_int_entries', 0)}")
    
    if "machine_model_distribution" in details and details["machine_model_distribution"]:
        d = details["machine_model_distribution"]
        safe_print(f"\n  Machine model distribution (top 10):")
        for mm_id, count, name in d.get("top_10", [])[:10]:
            safe_print(f"    - {name} (ID {mm_id}): {count} chunks")
    
    if "customer_visibility" in details and details["customer_visibility"] and args.role == "CUSTOMER":
        d = details["customer_visibility"]
        safe_print(f"\n  Customer visibility breakdown:")
        
        # Machine resolution
        resolved = d.get("resolved_machine_names", {})
        unmatched = d.get("unmatched_machine_names", [])
        if resolved:
            safe_print(f"    Machine name resolution:")
            for input_name, mm_id in resolved.items():
                id_to_name = get_machine_model_name_map()
                db_name = id_to_name.get(mm_id, f"Unknown_{mm_id}")
                safe_print(f"      '{input_name}' -> ID {mm_id} ({db_name})")
        if unmatched:
            safe_print(f"    ⚠️ Unmatched machine names: {unmatched}")
            candidates = get_top_machine_model_candidates(unmatched, limit=10)
            if candidates:
                safe_print(f"    Top 10 closest DB matches:")
                for name, mm_id in candidates:
                    safe_print(f"      - {name} (ID {mm_id})")
        
        safe_print(f"    - Allowed documents: {d.get('allowed_documents', 0)}")
        safe_print(f"    - Visible chunks: {d.get('visible_chunks', 0)}/{d.get('total_chunks', 0)}")
        safe_print(f"    - Excluded (inactive): {d.get('excluded_inactive', 0)}")
        safe_print(f"    - Excluded (machine mismatch): {d.get('excluded_machine_mismatch', 0)}")
        safe_print(f"    - Excluded (missing metadata): {d.get('excluded_missing_metadata', 0)}")
    
    # Summary table
    safe_print("\n" + "=" * 110)
    safe_print("Summary:")
    safe_print("=" * 110)
    safe_print(f"{'Check':40} | {'Status':5} | {'Key numbers':20} | Notes")
    safe_print("-" * 110)
    for r in results:
        safe_print(f"{r.name:40} | {r.status:5} | {r.key_numbers:20} | {r.notes}")
    safe_print("=" * 110)
    
    # Exit codes:
    # 1 = critical failures
    # 2 = warnings only
    # 0 = all pass
    any_fail = any(r.status == "FAIL" for r in results)
    any_warn = any(r.status == "WARN" for r in results)
    
    if any_fail:
        safe_print("\n❌ FAIL: Critical issues detected. Review failures above.")
        return 1
    if any_warn:
        safe_print("\n⚠️  WARN: Non-critical issues detected. Review warnings above.")
        return 2
    safe_print("\n✅ PASS: All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

