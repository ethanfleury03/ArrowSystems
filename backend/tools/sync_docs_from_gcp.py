"""
Sync documents and metadata from GCP for offline ingestion.

Downloads all documents from GCS and exports a manifest file with required metadata.
Can fetch metadata from admin endpoint or use a local metadata file.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.utils.filenames import canonicalize_filename
from backend.utils.gcs_client import list_objects, download_to_file


def fetch_metadata_from_endpoint(
    base_url: str,
    auth_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch metadata from admin export endpoint."""
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests library required. Install with: pip install requests")
    
    url = f"{base_url.rstrip('/')}/admin/documents/export-metadata"
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
        # Also support X-User-Token header (used by frontend)
        headers["X-User-Token"] = auth_token
    
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def compute_file_hash(file_path: Path) -> Optional[str]:
    """Compute SHA256 hash of file."""
    try:
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


def sync_docs_from_gcp(
    docs_bucket: str,
    docs_prefix: str,
    output_dir: Path,
    manifest_path: Optional[Path] = None,
    metadata_source: str = "endpoint",  # "endpoint" or "file"
    endpoint_url: Optional[str] = None,
    auth_token: Optional[str] = None,
    metadata_file: Optional[Path] = None,
    on_missing_metadata: str = "warn",  # "fail", "warn", or "skip"
    compute_hash: bool = False,
) -> Path:
    """
    Download all documents from GCS and create manifest.
    
    Args:
        docs_bucket: GCS bucket name
        docs_prefix: GCS prefix (e.g., "ROOT" or "documents/")
        output_dir: Local directory to save documents
        manifest_path: Path to write manifest.json (default: output_dir/manifest.json)
        metadata_source: "endpoint" or "file"
        endpoint_url: Base URL for admin endpoint (if metadata_source="endpoint")
        auth_token: Bearer token or X-User-Token for endpoint (if metadata_source="endpoint")
        metadata_file: Path to existing metadata JSON (if metadata_source="file")
        on_missing_metadata: What to do if metadata not found: "fail", "warn", or "skip"
        compute_hash: Whether to compute SHA256 hashes (slower but more robust)
    
    Returns:
        Path to manifest.json
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if manifest_path is None:
        manifest_path = output_dir / "manifest.json"
    else:
        manifest_path = Path(manifest_path).resolve()
    
    # Normalize prefix
    if docs_prefix and docs_prefix.upper() == "ROOT":
        docs_prefix = ""
    if docs_prefix and not docs_prefix.endswith("/"):
        docs_prefix = f"{docs_prefix}/"
    
    # Fetch metadata
    metadata_by_gcs_path: Dict[str, Dict[str, Any]] = {}
    
    if metadata_source == "endpoint":
        if not endpoint_url:
            raise ValueError("endpoint_url required when metadata_source='endpoint'")
        print(f"📥 Fetching metadata from {endpoint_url}...")
        try:
            manifest_data = fetch_metadata_from_endpoint(endpoint_url, auth_token)
            for doc in manifest_data.get("documents", []):
                gcs_path = doc.get("gcs_path")
                if gcs_path:
                    metadata_by_gcs_path[gcs_path] = doc
            print(f"   ✅ Loaded metadata for {len(metadata_by_gcs_path)} documents")
        except Exception as e:
            if on_missing_metadata == "fail":
                raise
            print(f"   ⚠️  Failed to fetch metadata: {e}")
            if on_missing_metadata == "warn":
                print("   Continuing without metadata...")
    
    elif metadata_source == "file":
        if not metadata_file or not metadata_file.exists():
            raise ValueError(f"metadata_file not found: {metadata_file}")
        print(f"📥 Loading metadata from {metadata_file}...")
        with open(metadata_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        for doc in manifest_data.get("documents", []):
            gcs_path = doc.get("gcs_path")
            if gcs_path:
                metadata_by_gcs_path[gcs_path] = doc
        print(f"   ✅ Loaded metadata for {len(metadata_by_gcs_path)} documents")
    
    # List all objects in GCS
    print(f"📦 Listing documents from gs://{docs_bucket}/{docs_prefix}...")
    objects = list_objects(docs_bucket, docs_prefix)
    
    supported_exts = {".pdf", ".docx", ".md", ".markdown"}
    manifest_entries = []
    downloaded_count = 0
    matched_count = 0
    missing_metadata_count = 0
    skipped_count = 0
    
    # Track canonical filename collisions
    canonical_to_paths: Dict[str, List[str]] = {}
    
    for obj in objects:
        if obj.name.endswith("/"):
            continue
        
        ext = Path(obj.name).suffix.lower()
        if ext not in supported_exts:
            continue
        
        gcs_path = f"gs://{docs_bucket}/{obj.name}"
        
        # Get metadata
        metadata = metadata_by_gcs_path.get(gcs_path, {})
        
        # Determine canonical filename and local path
        canonical_file_name = metadata.get("canonical_file_name")
        document_id = metadata.get("document_id", "unknown")
        
        if not canonical_file_name:
            # Fallback: derive from GCS path
            filename = os.path.basename(obj.name)
            canonical_file_name = canonicalize_filename(filename)
        
        # Prevent collisions: use {canonical_stem}__{document_id[:8]}{ext}
        canonical_stem = Path(canonical_file_name).stem
        canonical_ext = Path(canonical_file_name).suffix
        doc_id_short = document_id[:8] if document_id != "unknown" else "unknown"
        local_filename = f"{canonical_stem}__{doc_id_short}{canonical_ext}"
        
        # Track collisions
        if canonical_file_name not in canonical_to_paths:
            canonical_to_paths[canonical_file_name] = []
        canonical_to_paths[canonical_file_name].append(local_filename)
        
        local_path = output_dir / local_filename
        
        # Check if we should skip (missing metadata)
        if not metadata:
            missing_metadata_count += 1
            if on_missing_metadata == "skip":
                print(f"   ⏭️  Skipping {os.path.basename(obj.name)} (no metadata)")
                skipped_count += 1
                continue
            elif on_missing_metadata == "fail":
                raise ValueError(f"Missing metadata for {gcs_path}. Set --on-missing-metadata warn|skip to continue.")
            else:  # warn
                print(f"   ⚠️  No metadata for {os.path.basename(obj.name)}, using defaults")
        
        # Download file
        if local_path.exists() and local_path.stat().st_size == obj.size:
            print(f"   ⏭️  Skipping {local_filename} (already exists)")
        else:
            print(f"   📥 Downloading {local_filename}...")
            if not download_to_file(gcs_path, str(local_path)):
                print(f"   ❌ Failed to download {gcs_path}")
                continue
            downloaded_count += 1
        
        # Compute hash if requested
        sha256 = None
        if compute_hash:
            sha256 = compute_file_hash(local_path)
        
        # Build manifest entry
        entry = {
            "canonical_file_name": canonical_file_name,
            "display_name": metadata.get("display_name") or os.path.basename(obj.name),
            "gcs_path": gcs_path,
            "local_path": str(local_path),
            "sha256": sha256,
            "file_size_bytes": local_path.stat().st_size,
            "machine_model_ids": metadata.get("machine_model_ids", []),
            "machine_model_names": metadata.get("machine_model_names", []),
            "is_active": metadata.get("is_active", True),
            "category": metadata.get("category"),
            "product_family": metadata.get("product_family"),
            "document_id": document_id,
        }
        
        if metadata:
            matched_count += 1
        
        manifest_entries.append(entry)
    
    # Report collisions
    collisions = {k: v for k, v in canonical_to_paths.items() if len(v) > 1}
    if collisions:
        print(f"\n⚠️  Warning: {len(collisions)} canonical filename collisions detected:")
        for canonical, paths in list(collisions.items())[:5]:
            print(f"   {canonical} -> {len(paths)} files")
    
    # Write manifest
    manifest = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_bucket": docs_bucket,
        "source_prefix": docs_prefix,
        "documents": manifest_entries
    }
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    
    print(f"\n✅ Sync complete:")
    print(f"   - Documents downloaded: {downloaded_count}")
    print(f"   - Total documents: {len(manifest_entries)}")
    print(f"   - Metadata matched: {matched_count}")
    print(f"   - Metadata missing: {missing_metadata_count}")
    print(f"   - Skipped: {skipped_count}")
    print(f"   - Manifest: {manifest_path}")
    
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Sync documents and metadata from GCP for offline ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch metadata from admin endpoint and download docs
  python -m backend.tools.sync_docs_from_gcp \\
    --docs-bucket arrow-rag-support-prod-docs \\
    --docs-prefix ROOT \\
    --output-dir ./docs \\
    --metadata-source endpoint \\
    --endpoint-url https://support.arrsys.com \\
    --auth-token YOUR_JWT_TOKEN

  # Use existing metadata file
  python -m backend.tools.sync_docs_from_gcp \\
    --docs-bucket arrow-rag-support-prod-docs \\
    --docs-prefix ROOT \\
    --output-dir ./docs \\
    --metadata-source file \\
    --metadata-file ./metadata.json
        """
    )
    parser.add_argument("--docs-bucket", required=True, help="GCS bucket name")
    parser.add_argument("--docs-prefix", default="ROOT", help="GCS prefix (default: ROOT)")
    parser.add_argument("--output-dir", default="./docs", help="Output directory (default: ./docs)")
    parser.add_argument("--manifest", help="Manifest output path (default: output_dir/manifest.json)")
    
    parser.add_argument(
        "--metadata-source",
        choices=["endpoint", "file"],
        default="endpoint",
        help="Metadata source: endpoint (admin API) or file (local JSON)"
    )
    parser.add_argument("--endpoint-url", help="Base URL for admin endpoint (if metadata_source=endpoint)")
    parser.add_argument("--auth-token", help="Bearer token or X-User-Token for endpoint (if metadata_source=endpoint)")
    parser.add_argument("--metadata-file", help="Path to metadata JSON file (if metadata_source=file)")
    parser.add_argument(
        "--on-missing-metadata",
        choices=["fail", "warn", "skip"],
        default="warn",
        help="What to do if metadata not found (default: warn)"
    )
    parser.add_argument("--hash", action="store_true", help="Compute SHA256 hashes (slower)")
    
    args = parser.parse_args()
    
    try:
        manifest_path = sync_docs_from_gcp(
            docs_bucket=args.docs_bucket,
            docs_prefix=args.docs_prefix,
            output_dir=Path(args.output_dir),
            manifest_path=Path(args.manifest) if args.manifest else None,
            metadata_source=args.metadata_source,
            endpoint_url=args.endpoint_url,
            auth_token=args.auth_token,
            metadata_file=Path(args.metadata_file) if args.metadata_file else None,
            on_missing_metadata=args.on_missing_metadata,
            compute_hash=args.hash,
        )
        
        print(f"\n📋 Next steps:")
        print(f"   1. Review manifest: {manifest_path}")
        print(f"   2. Run ingestion: python backend/ingest.py --docs-dir {args.output_dir} --manifest {manifest_path} --no-db")
        print(f"   3. Run diagnostics: python -m backend.tools.diagnose_rag_contract --storage-dir latest_model --offline-manifest {manifest_path}")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

