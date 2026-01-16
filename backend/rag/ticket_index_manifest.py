"""
Ticket index manifest utilities for tracking index state and determining reindex needs.

Manifest tracks:
- built_at: When index was last built
- max_updated_at_indexed: Max updated_at from DB at time of build
- eligible_count_indexed: Count of eligible tickets at time of build
- index_prefix: GCS prefix where index lives
- job_execution_id: Cloud Run Job execution ID (best-effort)
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from backend.config.env import settings
from backend.logging_config import get_logger
from backend.rag.ticket_index_downloader import check_ticket_index_exists

logger = get_logger(__name__)


class TicketIndexManifest:
    """Manifest schema for ticket index state."""
    
    def __init__(
        self,
        built_at: str,
        max_updated_at_indexed: Optional[str],
        eligible_count_indexed: int,
        index_prefix: str,
        job_execution_id: Optional[str] = None,
        source_query_hash: Optional[str] = None
    ):
        self.built_at = built_at
        self.max_updated_at_indexed = max_updated_at_indexed
        self.eligible_count_indexed = eligible_count_indexed
        self.index_prefix = index_prefix
        self.job_execution_id = job_execution_id
        self.source_query_hash = source_query_hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "built_at": self.built_at,
            "max_updated_at_indexed": self.max_updated_at_indexed,
            "eligible_count_indexed": self.eligible_count_indexed,
            "index_prefix": self.index_prefix,
            "job_execution_id": self.job_execution_id,
            "source_query_hash": self.source_query_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TicketIndexManifest":
        """Create from dict."""
        return cls(
            built_at=data.get("built_at", ""),
            max_updated_at_indexed=data.get("max_updated_at_indexed"),
            eligible_count_indexed=data.get("eligible_count_indexed", 0),
            index_prefix=data.get("index_prefix", "ticket_cache/latest_model/"),
            job_execution_id=data.get("job_execution_id"),
            source_query_hash=data.get("source_query_hash"),
        )


def read_manifest(bucket_name: str, manifest_path: str) -> Optional[TicketIndexManifest]:
    """
    Read manifest from GCS.
    
    Args:
        bucket_name: GCS bucket name
        manifest_path: GCS object path to manifest
        
    Returns:
        TicketIndexManifest if exists, None otherwise
    """
    try:
        from google.cloud import storage
    except ImportError:
        logger.warning("[TICKET_REINDEX] google-cloud-storage not installed - cannot read manifest")
        return None
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(manifest_path)
        
        if not blob.exists():
            logger.debug("[TICKET_REINDEX] Manifest does not exist", manifest_path=manifest_path)
            return None
        
        content = blob.download_as_text()
        data = json.loads(content)
        manifest = TicketIndexManifest.from_dict(data)
        
        logger.info("[TICKET_REINDEX] Read manifest", manifest_path=manifest_path, built_at=manifest.built_at)
        return manifest
        
    except Exception as e:
        logger.warning(
            "[TICKET_REINDEX] Failed to read manifest",
            manifest_path=manifest_path,
            error=str(e)
        )
        return None


def write_manifest(
    bucket_name: str,
    manifest_path: str,
    manifest: TicketIndexManifest
) -> bool:
    """
    Write manifest to GCS.
    
    Args:
        bucket_name: GCS bucket name
        manifest_path: GCS object path to manifest
        manifest: Manifest to write
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from google.cloud import storage
    except ImportError:
        logger.error("[TICKET_REINDEX] google-cloud-storage not installed - cannot write manifest")
        return False
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(manifest_path)
        
        content = json.dumps(manifest.to_dict(), indent=2)
        blob.upload_from_string(content, content_type="application/json")
        
        logger.info(
            "[TICKET_REINDEX] Wrote manifest",
            manifest_path=manifest_path,
            built_at=manifest.built_at,
            eligible_count=manifest.eligible_count_indexed
        )
        return True
        
    except Exception as e:
        logger.error(
            "[TICKET_REINDEX] Failed to write manifest",
            manifest_path=manifest_path,
            error=str(e),
            exc_info=True
        )
        return False


def get_db_ticket_stats() -> Dict[str, Any]:
    """
    Get current ticket statistics from database.
    
    Returns:
        Dict with:
        - max_updated_at: Max updated_at from ticket_judgements (ISO string or None)
        - eligible_count: Count of cache-eligible tickets
    """
    from backend.utils.db import SessionLocal, TicketJudgement
    from sqlalchemy import func, text
    
    stats = {
        "max_updated_at": None,
        "eligible_count": 0,
    }
    
    try:
        with SessionLocal() as session:
            # Get max updated_at from ticket_judgements (use judged_at as proxy for updated_at)
            # Also check tickets_index.updated_at if available
            query = text("""
                SELECT 
                    MAX(GREATEST(
                        COALESCE(j.judged_at, '1970-01-01'::timestamp),
                        COALESCE(i.updated_at, '1970-01-01'::timestamp)
                    )) as max_updated_at,
                    COUNT(DISTINCT j.ticket_id) FILTER (WHERE j.cache_eligible = true) as eligible_count
                FROM ticket_judgements j
                LEFT JOIN tickets_index i ON j.ticket_id = i.ticket_id
                LEFT JOIN ticket_manual_reviews m ON j.ticket_id = m.ticket_id
                WHERE (
                    (j.review_status = 'approved' OR (j.review_status IS NULL AND j.cache_eligible = true))
                    OR (m.manual_status = 'approved')
                )
                AND (m.manual_status IS NULL OR m.manual_status != 'rejected')
                AND j.cache_eligible = true
            """)
            
            result = session.execute(query)
            row = result.fetchone()
            
            if row:
                max_updated_at = row[0]
                eligible_count = row[1] or 0
                
                if max_updated_at:
                    # Convert to ISO string
                    if isinstance(max_updated_at, datetime):
                        stats["max_updated_at"] = max_updated_at.isoformat()
                    else:
                        stats["max_updated_at"] = str(max_updated_at)
                
                stats["eligible_count"] = int(eligible_count)
                
    except Exception as e:
        logger.warning(
            "[TICKET_REINDEX] Failed to get DB ticket stats",
            error=str(e),
            exc_info=True
        )
    
    return stats


def needs_reindex(
    bucket_name: str,
    index_prefix: str,
    manifest_path: str
) -> tuple[bool, Optional[str]]:
    """
    Determine if ticket index needs to be rebuilt.
    
    Checks:
    1. Index exists in GCS (has real files, not just .keep)
    2. Manifest exists
    3. DB max_updated_at > manifest.max_updated_at_indexed
    4. DB eligible_count != manifest.eligible_count_indexed
    
    Args:
        bucket_name: GCS bucket name
        index_prefix: GCS prefix for ticket index
        manifest_path: GCS path to manifest
        
    Returns:
        Tuple of (needs_reindex: bool, reason: Optional[str])
    """
    # Check if index exists
    index_exists = check_ticket_index_exists(bucket_name, index_prefix)
    
    if not index_exists:
        logger.info("[TICKET_REINDEX] Index does not exist - needs reindex")
        return True, "index_missing"
    
    # Read manifest
    manifest = read_manifest(bucket_name, manifest_path)
    
    if not manifest:
        logger.info("[TICKET_REINDEX] Manifest does not exist - needs reindex")
        return True, "manifest_missing"
    
    # Get current DB stats
    db_stats = get_db_ticket_stats()
    
    # Check if DB has newer tickets
    if db_stats["max_updated_at"] and manifest.max_updated_at_indexed:
        try:
            db_max = datetime.fromisoformat(db_stats["max_updated_at"].replace("Z", "+00:00"))
            manifest_max = datetime.fromisoformat(manifest.max_updated_at_indexed.replace("Z", "+00:00"))
            
            if db_max > manifest_max:
                logger.info(
                    "[TICKET_REINDEX] DB has newer tickets - needs reindex",
                    db_max=db_stats["max_updated_at"],
                    manifest_max=manifest.max_updated_at_indexed
                )
                return True, "new_tickets_exist"
        except Exception as e:
            logger.warning(
                "[TICKET_REINDEX] Failed to compare timestamps",
                error=str(e),
                db_max=db_stats["max_updated_at"],
                manifest_max=manifest.max_updated_at_indexed
            )
    
    # Check if count changed
    if db_stats["eligible_count"] != manifest.eligible_count_indexed:
        logger.info(
            "[TICKET_REINDEX] Eligible count changed - needs reindex",
            db_count=db_stats["eligible_count"],
            manifest_count=manifest.eligible_count_indexed
        )
        return True, "count_mismatch"
    
    logger.debug(
        "[TICKET_REINDEX] Index is up to date",
        db_max=db_stats["max_updated_at"],
        manifest_max=manifest.max_updated_at_indexed,
        db_count=db_stats["eligible_count"],
        manifest_count=manifest.eligible_count_indexed
    )
    
    return False, None
