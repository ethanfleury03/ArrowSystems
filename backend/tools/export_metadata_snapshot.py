"""
Export document metadata snapshot from PostgreSQL to GCS.

This tool exports a JSON snapshot of document metadata (document_id, source_gcs,
machine_model_ids, machine_model_names, file_name) from the production database
to a GCS location. This snapshot can be used by ingest.py when DATABASE_URL
is unavailable (e.g., in RunPod environments).

Usage:
    export METADATA_SNAPSHOT_GCS_URI=gs://bucket/path/metadata_snapshot.json
    python backend/tools/export_metadata_snapshot.py
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
script_dir = Path(__file__).resolve().parent
backend_dir = script_dir.parent
repo_root = backend_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.utils.db import SessionLocal, Document, MachineModel
from backend.utils.gcs_client import upload_bytes, parse_gcs_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_metadata_snapshot() -> None:
    """
    Export document metadata from PostgreSQL to GCS as JSON snapshot.
    
    The snapshot includes:
    - source_gcs (or gcs_path) matching what ingest uses
    - document_id (DB UUID/integer as string)
    - file_name
    - machine_model_ids (list of integers as strings)
    - machine_model_names (list of strings)
    
    Format:
    {
        "generated_at": "ISO8601 timestamp",
        "count": N,
        "documents": [
            {
                "source_gcs": "gs://bucket/path",
                "document_id": "123",
                "file_name": "document.pdf",
                "machine_model_ids": ["1", "2"],
                "machine_model_names": ["DuraFlex", "DuraCore"]
            },
            ...
        ]
    }
    """
    snapshot_uri = os.getenv("METADATA_SNAPSHOT_GCS_URI")
    if not snapshot_uri:
        raise RuntimeError(
            "METADATA_SNAPSHOT_GCS_URI environment variable is required. "
            "Set it to a GCS URI like: gs://bucket/path/metadata_snapshot.json"
        )
    
    # Parse GCS URI
    bucket_name, blob_name = parse_gcs_path(snapshot_uri)
    if not bucket_name or not blob_name:
        raise ValueError(f"Invalid GCS URI format: {snapshot_uri}. Expected format: gs://bucket/path.json")
    
    logger.info(f"Exporting metadata snapshot to {snapshot_uri}")
    
    # Query database
    session = SessionLocal()
    documents_data = []
    
    try:
        # Query all documents with their machine models
        # Use eager loading to avoid N+1 queries
        documents = session.query(Document).all()
        
        logger.info(f"Found {len(documents)} documents in database")
        
        for doc in documents:
            # Get source_gcs (prefer gcs_path, fallback to constructing from bucket/prefix)
            source_gcs = doc.gcs_path
            if not source_gcs and doc.file_name:
                # If gcs_path is missing, we can't reliably construct it
                # Skip documents without gcs_path
                logger.warning(f"Document ID {doc.id} ({doc.file_name}) missing gcs_path, skipping")
                continue
            
            # Get machine models from join table
            machine_model_ids = []
            machine_model_names = []
            
            # Check if join table exists
            try:
                from sqlalchemy import text
                exists_row = session.execute(
                    text("SELECT to_regclass('public.document_machine_models')")
                ).scalar()
                has_join_table = bool(exists_row)
            except Exception:
                has_join_table = False
            
            if has_join_table:
                # Query join table for machine models
                try:
                    rows = session.execute(
                        text(
                            """
                            SELECT mm.id AS id, mm.name AS name
                            FROM public.document_machine_models dmm
                            JOIN public.machine_models mm ON mm.id = dmm.machine_model_id
                            WHERE dmm.document_id = :doc_id
                            """
                        ),
                        {"doc_id": int(doc.id)},
                    ).fetchall()
                    machine_model_ids = [str(int(r.id)) for r in rows if getattr(r, "id", None) is not None]
                    machine_model_names = [str(r.name).strip() for r in rows if getattr(r, "name", None) and str(r.name).strip()]
                except Exception as e:
                    logger.warning(f"Failed to query machine models for document_id={doc.id}: {e}")
                    # Fallback to legacy machine_model field
                    if doc.machine_model:
                        # Parse legacy format (could be JSON array string or comma-separated)
                        try:
                            import json as json_lib
                            parsed = json_lib.loads(doc.machine_model)
                            if isinstance(parsed, list):
                                machine_model_names = [str(m).strip() for m in parsed if m]
                        except Exception:
                            # Not JSON, try comma-separated
                            machine_model_names = [m.strip() for m in str(doc.machine_model).split(",") if m.strip()]
            else:
                # Legacy fallback: parse machine_model field
                if doc.machine_model:
                    try:
                        import json as json_lib
                        parsed = json_lib.loads(doc.machine_model)
                        if isinstance(parsed, list):
                            machine_model_names = [str(m).strip() for m in parsed if m]
                    except Exception:
                        # Not JSON, try comma-separated
                        machine_model_names = [m.strip() for m in str(doc.machine_model).split(",") if m.strip()]
                
                # Try to resolve IDs from names
                if machine_model_names:
                    machine_models = session.query(MachineModel).filter(
                        MachineModel.name.in_(machine_model_names)
                    ).all()
                    machine_model_ids = [str(int(m.id)) for m in machine_models if m.id is not None]
            
            # Build document entry
            doc_entry = {
                "source_gcs": source_gcs,
                "document_id": str(doc.id),
                "file_name": doc.file_name or None,
                "machine_model_ids": machine_model_ids,
                "machine_model_names": machine_model_names,
            }
            documents_data.append(doc_entry)
        
        # Build snapshot JSON
        snapshot = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(documents_data),
            "documents": documents_data,
        }
        
        # Serialize to JSON
        json_content = json.dumps(snapshot, indent=2, sort_keys=True)
        json_bytes = json_content.encode("utf-8")
        
        logger.info(f"Generated snapshot: {len(documents_data)} documents, {len(json_bytes)} bytes")
        
        # Upload to GCS
        logger.info(f"Uploading to {snapshot_uri}...")
        uploaded_uri = upload_bytes(
            bucket_name=bucket_name,
            object_name=blob_name,
            content=json_bytes,
            content_type="application/json",
        )
        
        if uploaded_uri:
            logger.info(f"✅ Successfully uploaded metadata snapshot to {uploaded_uri}")
            logger.info(f"   - Documents: {len(documents_data)}")
            logger.info(f"   - Size: {len(json_bytes)} bytes")
        else:
            raise RuntimeError(f"Failed to upload snapshot to {snapshot_uri}")
            
    except Exception as e:
        logger.error(f"Failed to export metadata snapshot: {e}", exc_info=True)
        raise
    finally:
        session.close()


if __name__ == "__main__":
    try:
        export_metadata_snapshot()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        sys.exit(1)

