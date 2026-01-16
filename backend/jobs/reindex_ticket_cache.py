#!/usr/bin/env python3
"""
Cloud Run Job entrypoint for rebuilding ticket cache index.

This job:
1. Acquires a distributed lock in GCS
2. Checks if reindex is needed
3. Exports ticket cache artifacts from Postgres
4. Builds a fresh ticket index
5. Backs up existing index
6. Uploads new index to GCS
7. Writes manifest
8. Releases lock

Usage:
    python -m backend.jobs.reindex_ticket_cache
"""

import json
import os
import sys
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

# Add repo root to path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from backend.config.env import settings
from backend.logging_config import get_logger
from backend.rag.ticket_index_manifest import (
    TicketIndexManifest,
    read_manifest,
    write_manifest,
    get_db_ticket_stats,
    needs_reindex
)
from backend.rag.ticket_index_downloader import check_ticket_index_exists

logger = get_logger(__name__)


def acquire_gcs_lock(bucket_name: str, lock_path: str) -> bool:
    """
    Acquire distributed lock in GCS using generation-match precondition.
    
    Creates lock object only if it doesn't exist (atomic operation).
    
    Args:
        bucket_name: GCS bucket name
        lock_path: GCS object path for lock
        
    Returns:
        True if lock acquired, False if already exists
    """
    try:
        from google.cloud import storage
    except ImportError:
        logger.error("[TICKET_REINDEX] google-cloud-storage not installed")
        return False
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(lock_path)
        
        # Try to create lock with generation=0 (only succeeds if doesn't exist)
        lock_content = json.dumps({
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "job_execution_id": os.getenv("CLOUD_RUN_EXECUTION", "unknown"),
        })
        
        try:
            # Use if_generation_match=0 to ensure atomic creation
            blob.upload_from_string(
                lock_content,
                content_type="application/json",
                if_generation_match=0  # Only create if generation is 0 (doesn't exist)
            )
            
            logger.info("[TICKET_REINDEX] Acquired lock", lock_path=lock_path)
            return True
            
        except Exception as e:
            # Check if error is due to generation mismatch (lock exists)
            if "conditionNotMet" in str(e) or "412" in str(e):
                logger.info("[TICKET_REINDEX] Lock already exists - another job is running", lock_path=lock_path)
                return False
            else:
                raise
        
    except Exception as e:
        logger.error(
            "[TICKET_REINDEX] Failed to acquire lock",
            lock_path=lock_path,
            error=str(e),
            exc_info=True
        )
        return False


def release_gcs_lock(bucket_name: str, lock_path: str) -> bool:
    """
    Release distributed lock by deleting lock object.
    
    Args:
        bucket_name: GCS bucket name
        lock_path: GCS object path for lock
        
    Returns:
        True if released, False otherwise
    """
    try:
        from google.cloud import storage
    except ImportError:
        return False
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(lock_path)
        
        if blob.exists():
            blob.delete()
            logger.info("[TICKET_REINDEX] Released lock", lock_path=lock_path)
            return True
        else:
            logger.warning("[TICKET_REINDEX] Lock does not exist when releasing", lock_path=lock_path)
            return False
            
    except Exception as e:
        logger.error(
            "[TICKET_REINDEX] Failed to release lock",
            lock_path=lock_path,
            error=str(e),
            exc_info=True
        )
        return False


def export_ticket_artifacts_from_postgres(output_path: Path) -> Dict[str, Any]:
    """
    Export ticket cache artifacts from Postgres to JSONL.
    
    Uses the enhanced export logic with conversation_json join and redaction.
    
    Args:
        output_path: Path to write JSONL file
        
    Returns:
        Dict with export stats
    """
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import NullPool
    # Import export function - use backend utils version (supports conversation_json and redaction)
    from backend.utils.ticket_cache_artifacts import build_ticket_cache_artifact
    
    if not settings.DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")
    
    engine = create_engine(settings.DATABASE_URL, poolclass=NullPool, future=True)
    
    # Query for cache-eligible tickets (same as export_cache_artifacts.py)
    query = text("""
        SELECT 
            j.ticket_id,
            j.raw_response_json,
            j.cache_eligible,
            j.confidence,
            j.model,
            j.prompt_version,
            j.judged_at,
            m.manual_status,
            m.reviewer,
            d.conversation_json
        FROM ticket_judgements j
        LEFT JOIN ticket_manual_reviews m ON j.ticket_id = m.ticket_id
        LEFT JOIN tickets_detail d ON j.ticket_id = d.ticket_id
        WHERE (
            (j.review_status = 'approved' OR (j.review_status IS NULL AND j.cache_eligible = true))
            OR (m.manual_status = 'approved')
        )
        AND (m.manual_status IS NULL OR m.manual_status != 'rejected')
        AND j.cache_eligible = true
        ORDER BY j.ticket_id
    """)
    
    artifacts = []
    failed = 0
    errors = []
    
    with engine.connect() as conn:
        result = conn.execute(query)
        rows = [dict(row._mapping) for row in result.fetchall()]
    
    logger.info(f"[TICKET_REINDEX] Found {len(rows)} cache-eligible tickets")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in rows:
            ticket_id_str = row['ticket_id']
            
            try:
                # Parse raw_response_json
                raw_json = row['raw_response_json']
                if isinstance(raw_json, str):
                    raw_json = json.loads(raw_json)
                elif not isinstance(raw_json, dict):
                    raise ValueError(f"raw_response_json is not a dict, got: {type(raw_json)}")
                
                # Parse conversation_json
                conversation_json = None
                if row.get('conversation_json'):
                    conv_json = row['conversation_json']
                    if isinstance(conv_json, str):
                        conversation_json = json.loads(conv_json)
                    elif isinstance(conv_json, dict):
                        conversation_json = conv_json
                
                # Build artifact
                extra_meta = {
                    "prompt_version": row.get("prompt_version"),
                }
                
                artifact = build_ticket_cache_artifact(
                    ticket_id=ticket_id_str,
                    raw_response_json=raw_json,
                    conversation_json=conversation_json,
                    extra_meta=extra_meta
                )
                
                # Write JSONL line
                f.write(json.dumps(artifact, ensure_ascii=False) + '\n')
                artifacts.append(artifact)
                
            except Exception as e:
                failed += 1
                error_msg = f"Failed to process ticket {ticket_id_str}: {e}"
                errors.append(error_msg)
                logger.warning("[TICKET_REINDEX] " + error_msg)
    
    logger.info(
        "[TICKET_REINDEX] Export complete",
        exported=len(artifacts),
        failed=failed
    )
    
    return {
        "total": len(rows),
        "exported": len(artifacts),
        "failed": failed,
        "errors": errors,
    }


def build_ticket_index(jsonl_path: Path, index_dir: Path) -> bool:
    """
    Build ticket index from JSONL artifacts.
    
    Uses ingest_ticket_cache_artifacts logic as library call.
    
    Args:
        jsonl_path: Path to JSONL file with artifacts
        index_dir: Directory to build index in
        
    Returns:
        True if successful, False otherwise
    """
    from backend.scripts.ingest_ticket_cache_artifacts import ingest_artifacts
    
    # Ensure index dir exists
    index_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(
        "[TICKET_REINDEX] Building ticket index",
        jsonl_path=str(jsonl_path),
        index_dir=str(index_dir)
    )
    
    try:
        result = ingest_artifacts(
            jsonl_path=str(jsonl_path),
            index_dir=str(index_dir),
            dry_run=False,
            skip_existing=False  # Overwrite semantics
        )
        
        if result["failed"] > 0:
            logger.error(
                "[TICKET_REINDEX] Index build had failures",
                failed=result["failed"],
                errors=result.get("errors", [])[:5]
            )
            return False
        
        # Verify required files exist
        required_files = ["docstore.json", "index_store.json", "default__vector_store.json"]
        for filename in required_files:
            file_path = index_dir / filename
            if not file_path.exists() or file_path.stat().st_size < 1024:
                logger.error(
                    "[TICKET_REINDEX] Required index file missing or too small",
                    filename=filename,
                    exists=file_path.exists(),
                    size=file_path.stat().st_size if file_path.exists() else 0
                )
                return False
        
        logger.info(
            "[TICKET_REINDEX] Index build complete",
            inserted=result["inserted"],
            overwritten=result.get("overwritten", 0)
        )
        
        return True
        
    except Exception as e:
        logger.error(
            "[TICKET_REINDEX] Index build failed",
            error=str(e),
            exc_info=True
        )
        return False


def backup_existing_index(bucket_name: str, index_prefix: str, backups_prefix: str) -> Optional[str]:
    """
    Backup existing ticket index to backups prefix.
    
    Args:
        bucket_name: GCS bucket name
        index_prefix: Current index prefix
        backups_prefix: Backups prefix
        
    Returns:
        Backup prefix path if successful, None otherwise
    """
    try:
        from google.cloud import storage
    except ImportError:
        logger.error("[TICKET_REINDEX] google-cloud-storage not installed")
        return None
    
    # Check if index exists
    if not check_ticket_index_exists(bucket_name, index_prefix):
        logger.info("[TICKET_REINDEX] No existing index to backup")
        return None
    
    # Create backup prefix with timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_prefix = f"{backups_prefix}{timestamp}/"
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # List all blobs under index_prefix
        source_blobs = list(bucket.list_blobs(prefix=index_prefix))
        
        if not source_blobs:
            logger.info("[TICKET_REINDEX] No files to backup")
            return None
        
        # Copy each blob to backup prefix
        copied = 0
        for source_blob in source_blobs:
            # Skip .keep files
            if source_blob.name.endswith(".keep"):
                continue
            
            # Destination blob name
            relative_name = source_blob.name[len(index_prefix):]
            dest_name = f"{backup_prefix}{relative_name}"
            
            # Copy blob
            dest_blob = bucket.blob(dest_name)
            dest_blob.rewrite(source_blob)
            copied += 1
        
        logger.info(
            "[TICKET_REINDEX] Backup created",
            backup_prefix=backup_prefix,
            files_copied=copied
        )
        
        return backup_prefix
        
    except Exception as e:
        logger.error(
            "[TICKET_REINDEX] Backup failed",
            error=str(e),
            exc_info=True
        )
        return None


def upload_index_to_gcs(index_dir: Path, bucket_name: str, index_prefix: str) -> bool:
    """
    Upload ticket index directory to GCS.
    
    Args:
        index_dir: Local directory containing index files
        bucket_name: GCS bucket name
        index_prefix: GCS prefix to upload to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from google.cloud import storage
    except ImportError:
        logger.error("[TICKET_REINDEX] google-cloud-storage not installed")
        return False
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Required files to upload
        required_files = [
            "docstore.json",
            "index_store.json",
            "default__vector_store.json",
        ]
        
        # Optional files
        optional_files = [
            "index_manifest.json",
        ]
        
        uploaded = 0
        
        # Upload required files
        for filename in required_files:
            local_path = index_dir / filename
            if not local_path.exists():
                logger.error("[TICKET_REINDEX] Required file missing", filename=filename)
                return False
            
            blob_name = f"{index_prefix}{filename}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(local_path))
            uploaded += 1
            
            logger.debug(
                "[TICKET_REINDEX] Uploaded file",
                filename=filename,
                size=local_path.stat().st_size
            )
        
        # Upload optional files
        for filename in optional_files:
            local_path = index_dir / filename
            if local_path.exists():
                blob_name = f"{index_prefix}{filename}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(local_path))
                uploaded += 1
        
        logger.info(
            "[TICKET_REINDEX] Index upload complete",
            files_uploaded=uploaded,
            prefix=index_prefix
        )
        
        return True
        
    except Exception as e:
        logger.error(
            "[TICKET_REINDEX] Index upload failed",
            error=str(e),
            exc_info=True
        )
        return False


def main() -> int:
    """
    Main job entrypoint.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    bucket_name = settings.RAG_BUCKET
    index_prefix = settings.TICKET_PREFIX
    backups_prefix = settings.TICKET_BACKUPS_PREFIX
    manifest_path = settings.TICKET_MANIFEST_PATH
    lock_path = settings.TICKET_LOCK_PATH
    
    logger.info(
        "[TICKET_REINDEX] Starting ticket cache reindex job",
        bucket=bucket_name,
        index_prefix=index_prefix,
        manifest_path=manifest_path,
        lock_path=lock_path
    )
    
    # Step 1: Acquire lock
    if not acquire_gcs_lock(bucket_name, lock_path):
        logger.info("[TICKET_REINDEX] Could not acquire lock - another job is running")
        return 0  # Exit gracefully, not an error
    
    lock_acquired = True
    
    try:
        # Step 2: Check needs_reindex again (defense-in-depth)
        needs_rebuild, reason = needs_reindex(bucket_name, index_prefix, manifest_path)
        
        if not needs_rebuild:
            logger.info(
                "[TICKET_REINDEX] Reindex not needed",
                reason=reason or "up_to_date"
            )
            return 0
        
        logger.info(
            "[TICKET_REINDEX] Reindex needed",
            reason=reason
        )
        
        # Step 3: Export artifacts from Postgres
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            jsonl_path = tmp_path / "cache_artifacts.jsonl"
            
            export_stats = export_ticket_artifacts_from_postgres(jsonl_path)
            
            if export_stats["failed"] > 0:
                logger.warning(
                    "[TICKET_REINDEX] Export had failures",
                    failed=export_stats["failed"]
                )
            
            if export_stats["exported"] == 0:
                logger.error("[TICKET_REINDEX] No artifacts exported")
                return 1
            
            # Step 4: Build index
            index_dir = tmp_path / "ticket_cache_model_build"
            
            if not build_ticket_index(jsonl_path, index_dir):
                logger.error("[TICKET_REINDEX] Index build failed")
                return 1
            
            # Step 5: Backup existing index
            backup_prefix = backup_existing_index(bucket_name, index_prefix, backups_prefix)
            if backup_prefix:
                logger.info("[TICKET_REINDEX] Backup created", backup_prefix=backup_prefix)
            
            # Step 6: Upload new index
            if not upload_index_to_gcs(index_dir, bucket_name, index_prefix):
                logger.error("[TICKET_REINDEX] Index upload failed")
                return 1
            
            # Step 7: Write manifest
            db_stats = get_db_ticket_stats()
            job_execution_id = os.getenv("CLOUD_RUN_EXECUTION", "unknown")
            
            manifest = TicketIndexManifest(
                built_at=datetime.now(timezone.utc).isoformat(),
                max_updated_at_indexed=db_stats["max_updated_at"],
                eligible_count_indexed=db_stats["eligible_count"],
                index_prefix=index_prefix,
                job_execution_id=job_execution_id,
            )
            
            if not write_manifest(bucket_name, manifest_path, manifest):
                logger.error("[TICKET_REINDEX] Manifest write failed")
                return 1
            
            logger.info(
                "[TICKET_REINDEX] Reindex complete",
                built_at=manifest.built_at,
                eligible_count=manifest.eligible_count_indexed,
                backup_prefix=backup_prefix
            )
            
            return 0
            
    finally:
        # Step 8: Release lock
        if lock_acquired:
            release_gcs_lock(bucket_name, lock_path)


if __name__ == "__main__":
    try:
        # Set LLM to None to avoid OpenAI initialization
        from llama_index.core import Settings
        Settings.llm = None
        
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error("[TICKET_REINDEX] Fatal error", error=str(e), exc_info=True)
        sys.exit(1)
