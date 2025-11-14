"""
Audit Logging Module

Provides lightweight audit logging for admin-facing events.
Stores only essential events in the database (not raw logs).
Full structlog logs remain in Cloud Logging.
"""

import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from fastapi import Request

from .db import SessionLocal, AuditLog, run_sync
from ..logging_context import get_request_id, get_user_id, get_user_role
from ..logging_config import get_logger

logger = get_logger(__name__)


async def audit_log(
    event: str,
    level: str = "info",
    user_id: Optional[str] = None,
    role: Optional[str] = None,
    ip_address: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    request: Optional[Request] = None,
) -> None:
    """
    Log an audit event to the database.
    
    Args:
        event: Event name (e.g., "user_login", "manual_upload_start")
        level: Log level ("info", "warning", "error")
        user_id: User ID or email (auto-filled from context if not provided)
        role: User role (auto-filled from context if not provided)
        ip_address: IP address (auto-filled from request if not provided)
        metadata: Additional structured data (will be stored as JSON)
        request: FastAPI request object (for extracting IP address)
    
    This function is non-blocking and should not raise exceptions.
    """
    try:
        # Get values from context if not provided
        if user_id is None:
            user_id = get_user_id()
        if role is None:
            role = get_user_role()
        
        # Get request ID from context
        request_id = get_request_id()
        
        # Get IP address from request if not provided
        if ip_address is None and request:
            # Try to get real IP from headers (for proxies)
            ip_address = (
                request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or
                request.headers.get("X-Real-IP", "").strip() or
                request.client.host if request.client else None
            )
        
        # Prepare metadata (convert to JSON-serializable dict)
        metadata_dict = {}
        if metadata:
            try:
                # Ensure metadata is JSON-serializable
                metadata_dict = json.loads(json.dumps(metadata))
            except (TypeError, ValueError):
                # If metadata is not serializable, convert to string
                metadata_dict = {"raw": str(metadata)}
        
        # Insert audit log asynchronously
        def _insert_log():
            with SessionLocal() as session:
                try:
                    audit_entry = AuditLog(
                        timestamp=datetime.utcnow(),
                        level=level.lower(),
                        event=event,
                        user_id=str(user_id) if user_id else None,
                        role=role,
                        ip_address=ip_address,
                        event_metadata=metadata_dict,  # Use event_metadata (maps to metadata column in DB)
                        request_id=request_id,
                    )
                    session.add(audit_entry)
                    session.flush()  # Ensure it's written before commit
                    session.commit()
                    # Verify it was saved
                    saved_id = audit_entry.id
                    print(f"DEBUG: Audit log saved successfully: {event} for user {user_id}, id={saved_id}", file=sys.stderr)
                except Exception as e:
                    session.rollback()
                    # Log error but don't raise (audit logging should never break the app)
                    logger.warning("audit_log_failed", error=str(e), original_event=event, exc_info=True)
                    # Always print error for debugging
                    print(f"DEBUG: Audit log failed: {e}", file=sys.stderr)
        
        # Run in thread pool to avoid blocking
        await run_sync(_insert_log)
        
        # Also log to structlog for Cloud Logging (with audit prefix)
        logger.info(
            "audit_event",
            audit_event=event,
            level=level,
            user_id=user_id,
            role=role,
            ip_address=ip_address,
            request_id=request_id,
            metadata=metadata_dict,
        )
        
    except Exception as e:
        # Never raise exceptions from audit logging
        logger.error("audit_log_exception", error=str(e), original_event=event, exc_info=True)

