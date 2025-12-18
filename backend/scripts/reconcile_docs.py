#!/usr/bin/env python3
"""
Reconcile DB document records vs GCS objects (no HTTP required).

Goal:
- Detect DB records whose expected GCS object is missing (common cause of count mismatch)
- Optionally delete orphan/failed DB records so the Admin UI matches the bucket

Usage:
  python -m backend.scripts.reconcile_docs --dry-run
  python -m backend.scripts.reconcile_docs --fix-failed-only --apply
  python -m backend.scripts.reconcile_docs --fix-orphans --apply

Inputs:
- DATABASE_URL (required)
- GCS_DOCS_BUCKET (default: arrow-rag-support-prod-docs)
- GCS_DOCS_PREFIX (default: "" (bucket root); supports "ROOT" sentinel)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from backend.config.env import normalize_gcs_prefix
from backend.utils.db import SessionLocal, DocumentIngestionMetadata, Document
from backend.utils.gcs_client import list_objects


def _sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r"[^\w\s.-]", "_", filename)
    sanitized = sanitized.replace(" ", "_")
    return sanitized


def _parse_gs_uri(gs_uri: str) -> tuple[Optional[str], Optional[str]]:
    if not gs_uri or not gs_uri.startswith("gs://"):
        return None, None
    path = gs_uri.replace("gs://", "").strip("/")
    if not path:
        return None, None
    parts = path.split("/", 1)
    bucket = parts[0]
    obj = parts[1] if len(parts) > 1 else ""
    return bucket, obj


@dataclass
class DbRow:
    metadata_id: str
    filename: str
    status: str
    error_message: str | None
    meta_file_path: str | None
    document_id: int | None
    doc_gcs_path: str | None
    doc_is_active: bool | None

    def expected_object_candidates(self, configured_prefix: str) -> list[str]:
        """
        Returns possible object names (bucket-relative keys) that might represent this DB row.
        Order is important: prefer authoritative stored URI paths first.
        """
        candidates: list[str] = []

        for uri in [self.doc_gcs_path, self.meta_file_path]:
            if uri and uri.startswith("gs://"):
                _b, obj = _parse_gs_uri(uri)
                if obj:
                    candidates.append(obj)

        # Fallbacks when DB did not store a gs://... path.
        #
        # New canonical scheme (root/prefix filename-only):
        #   <prefix><sanitized_filename>   OR   <sanitized_filename> (bucket root)
        #
        # Legacy scheme (older uploads):
        #   <prefix><metadata_id>/<sanitized_filename> OR <metadata_id>/<sanitized_filename>
        if self.metadata_id and self.filename:
            sanitized = _sanitize_filename(self.filename)
            if configured_prefix:
                # New canonical
                candidates.append(f"{configured_prefix}{sanitized}")
                candidates.append(f"{configured_prefix}{self.filename}")
                # Legacy
                candidates.append(f"{configured_prefix}{self.metadata_id}/{sanitized}")
                candidates.append(f"{configured_prefix}{self.metadata_id}/{self.filename}")
            else:
                # New canonical (bucket root)
                candidates.append(sanitized)
                candidates.append(self.filename)
                # Legacy
                candidates.append(f"{self.metadata_id}/{sanitized}")
                candidates.append(f"{self.metadata_id}/{self.filename}")

        # Last resort: filename-only
        if self.filename:
            candidates.append(self.filename)
            candidates.append(_sanitize_filename(self.filename))

        # De-dup while preserving order
        seen = set()
        out: list[str] = []
        for c in candidates:
            if not c or c in seen:
                continue
            seen.add(c)
            out.append(c)
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile DB documents vs GCS objects (no HTTP).")
    parser.add_argument("--fix-orphans", action="store_true", help="Delete DB records that reference missing GCS objects.")
    parser.add_argument("--fix-failed-only", action="store_true", help="Only delete FAILED records that are missing in GCS.")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Dry run (default): no destructive actions.")
    parser.add_argument("--apply", action="store_true", help="Actually perform deletes (overrides --dry-run).")
    parser.add_argument("--print-json", action="store_true", help="Print machine-readable JSON output.")
    args = parser.parse_args()

    if args.apply:
        args.dry_run = False

    bucket = (os.getenv("GCS_DOCS_BUCKET") or os.getenv("DOCS_GCS_BUCKET") or "arrow-rag-support-prod-docs").strip()
    raw_prefix = os.environ.get("GCS_DOCS_PREFIX")
    if raw_prefix is None:
        raw_prefix = os.environ.get("DOCS_GCS_PREFIX")
    prefix = normalize_gcs_prefix(raw_prefix)

    if not os.getenv("DATABASE_URL"):
        raise SystemExit("DATABASE_URL is required")

    # List GCS objects under prefix (skip directory markers)
    gcs_objs = list_objects(bucket, prefix)
    gcs_keys = {o.name for o in gcs_objs if o.name and not o.name.endswith("/")}

    # Load DB rows similar to /admin/documents endpoint (metadata is source-of-truth for listing)
    session = SessionLocal()
    try:
        results = (
            session.query(DocumentIngestionMetadata, Document)
            .outerjoin(Document, DocumentIngestionMetadata.filename == Document.file_name)
            .order_by(DocumentIngestionMetadata.created_at.desc())
            .all()
        )

        db_rows: list[DbRow] = []
        for meta, doc in results:
            db_rows.append(
                DbRow(
                    metadata_id=str(meta.id),
                    filename=str(meta.filename),
                    status=str(meta.status or ""),
                    error_message=getattr(meta, "error_message", None),
                    meta_file_path=getattr(meta, "file_path", None),
                    document_id=int(doc.id) if doc is not None and getattr(doc, "id", None) is not None else None,
                    doc_gcs_path=getattr(doc, "gcs_path", None) if doc is not None else None,
                    doc_is_active=getattr(doc, "is_active", None) if doc is not None else None,
                )
            )

        missing_in_gcs: list[dict[str, Any]] = []
        ok_in_gcs: int = 0
        failed_docs: list[dict[str, Any]] = []

        # Map of object keys that are "claimed" by DB rows (best-effort)
        claimed_keys: set[str] = set()

        for row in db_rows:
            candidates = row.expected_object_candidates(prefix)
            found = None
            for c in candidates:
                if c in gcs_keys:
                    found = c
                    break
            if found:
                ok_in_gcs += 1
                claimed_keys.add(found)
            else:
                missing_in_gcs.append(
                    {
                        "metadata_id": row.metadata_id,
                        "document_id": row.document_id,
                        "filename": row.filename,
                        "status": row.status,
                        "error": row.error_message,
                        "doc_gcs_path": row.doc_gcs_path,
                        "meta_file_path": row.meta_file_path,
                        "candidates": candidates[:6],
                    }
                )

            if row.status.upper() == "FAILED":
                failed_docs.append(
                    {
                        "metadata_id": row.metadata_id,
                        "document_id": row.document_id,
                        "filename": row.filename,
                        "error": row.error_message,
                        "doc_gcs_path": row.doc_gcs_path,
                        "meta_file_path": row.meta_file_path,
                    }
                )

        missing_in_db = sorted(list(gcs_keys - claimed_keys))

        summary = {
            "gcs_bucket": bucket,
            "gcs_prefix": prefix,
            "gcs_object_count": len(gcs_keys),
            "db_metadata_rows": len(db_rows),
            "db_rows_with_object_found": ok_in_gcs,
            "missing_in_gcs_count": len(missing_in_gcs),
            "missing_in_db_count": len(missing_in_db),
            "failed_docs_count": len(failed_docs),
            "dry_run": args.dry_run,
            "fix_orphans": bool(args.fix_orphans),
            "fix_failed_only": bool(args.fix_failed_only),
        }

        if args.print_json:
            print(json.dumps({"summary": summary, "missing_in_gcs": missing_in_gcs, "missing_in_db": missing_in_db, "failed_docs": failed_docs}, indent=2))
        else:
            print("=" * 80)
            print("DB ↔ GCS Reconciliation (no HTTP)")
            print("=" * 80)
            print(f"GCS: gs://{bucket}/{prefix}" if prefix else f"GCS: gs://{bucket}/ (bucket root)")
            print(f"GCS objects: {len(gcs_keys)}")
            print(f"DB metadata rows: {len(db_rows)}")
            print(f"DB rows matched to existing GCS objects: {ok_in_gcs}")
            print(f"Missing in GCS (DB rows with no matching object): {len(missing_in_gcs)}")
            print(f"Missing in DB (GCS objects not claimed by DB): {len(missing_in_db)}")
            print(f"FAILED docs in DB: {len(failed_docs)}")
            print("")

            if missing_in_gcs:
                print("Missing in GCS (first 20):")
                for r in missing_in_gcs[:20]:
                    print(f"- metadata_id={r['metadata_id']} filename={r['filename']} status={r['status']}")
                    print(f"  error={r['error']}")
                    print(f"  doc_gcs_path={r['doc_gcs_path']}")
                    print(f"  meta_file_path={r['meta_file_path']}")
                    print(f"  candidates={r['candidates']}")
                print("")

            if failed_docs:
                print("FAILED docs (first 20):")
                for r in failed_docs[:20]:
                    print(f"- metadata_id={r['metadata_id']} filename={r['filename']}")
                    print(f"  error={r['error']}")
                print("")

        # Deletion plan
        to_delete: list[DbRow] = []
        missing_ids = {r["metadata_id"] for r in missing_in_gcs}
        if args.fix_orphans or args.fix_failed_only:
            for row in db_rows:
                if row.metadata_id not in missing_ids:
                    continue
                if args.fix_failed_only and row.status.upper() != "FAILED":
                    continue
                to_delete.append(row)

        if to_delete and args.dry_run:
            print(f"[DRY RUN] Would delete {len(to_delete)} DB record(s) missing in GCS.")
            print("Run again with --apply to perform deletion.")
            return 0

        if to_delete and not args.dry_run:
            deleted = 0
            for row in to_delete:
                # Delete metadata row (UI source of truth)
                meta = session.query(DocumentIngestionMetadata).filter(DocumentIngestionMetadata.id == row.metadata_id).first()
                if meta:
                    session.delete(meta)
                # Also delete Document row by filename (best-effort cleanup)
                doc = session.query(Document).filter(Document.file_name == row.filename).first()
                if doc:
                    session.delete(doc)
                deleted += 1
                print(f"[DELETE] metadata_id={row.metadata_id} filename={row.filename} (missing in GCS)")
            session.commit()
            print(f"Deleted {deleted} DB record(s).")

        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())


