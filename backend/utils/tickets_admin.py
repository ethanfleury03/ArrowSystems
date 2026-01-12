"""
Admin utilities for reading tickets from Scraper SQLite database.

This module provides read-only access to ticket data stored in the Scraper SQLite DB.
Handles graceful degradation when the DB is not available.
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..logging_config import get_logger

logger = get_logger(__name__)


def get_tickets_db_path() -> Optional[Path]:
    """
    Get the path to the tickets SQLite database.
    
    Checks:
    1. TICKETS_DB_PATH environment variable (absolute path)
    2. Scraper/data/tickets.db (relative to project root)
    
    Returns:
        Path to tickets.db if found, None otherwise
    """
    # Check environment variable first
    env_path = os.getenv("TICKETS_DB_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        logger.warning(f"TICKETS_DB_PATH set but file not found: {env_path}")
    
    # Try relative path: Scraper/data/tickets.db from project root
    # Project root is typically 2 levels up from backend/utils/
    project_root = Path(__file__).parent.parent.parent
    relative_path = project_root / "Scraper" / "data" / "tickets.db"
    if relative_path.exists():
        return relative_path
    
    logger.debug(f"Tickets DB not found at {relative_path}")
    return None


def get_tickets_connection() -> Optional[sqlite3.Connection]:
    """
    Get a SQLite connection to the tickets database.
    
    Returns:
        sqlite3.Connection if DB exists, None otherwise
    """
    db_path = get_tickets_db_path()
    if not db_path:
        return None
    
    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to tickets DB at {db_path}: {e}")
        return None


def extract_machine_models(raw_response_json: Optional[str]) -> List[str]:
    """
    Extract machine model names from raw_response_json.
    
    Args:
        raw_response_json: JSON string from ticket_judgements.raw_response_json
        
    Returns:
        List of machine model names (empty list if none found)
    """
    if not raw_response_json:
        return []
    
    try:
        data = json.loads(raw_response_json)
        # Check various possible fields
        for key in ["machine_models", "machine_model_names", "machine_model"]:
            value = data.get(key)
            if isinstance(value, list):
                return [str(m) for m in value if m]
            elif isinstance(value, str) and value:
                return [value]
    except Exception:
        pass
    
    return []


def extract_confirmation_status(raw_response_json: Optional[str]) -> bool:
    """
    Extract confirmation status from raw_response_json.
    
    Args:
        raw_response_json: JSON string from ticket_judgements.raw_response_json
        
    Returns:
        True if confirmed, False otherwise
    """
    if not raw_response_json:
        return False
    
    try:
        data = json.loads(raw_response_json)
        confirmation = data.get("confirmation")
        if isinstance(confirmation, dict):
            return bool(confirmation.get("confirmed", False))
        elif isinstance(confirmation, bool):
            return confirmation
    except Exception:
        pass
    
    return False


def get_tickets_page(
    page: int = 1,
    page_size: int = 50,
    q: Optional[str] = None,
    sort: str = "judged_at DESC"
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Get a paginated list of tickets with their judgment data.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page (max 200)
        q: Optional search query (searches ticket_id, subject, outcome)
        sort: Sort order (default: "judged_at DESC")
        
    Returns:
        Tuple of (items list, total count)
        
    Raises:
        FileNotFoundError: If tickets DB is not available
    """
    # Validate inputs
    page = max(1, page)
    page_size = min(max(1, page_size), 200)
    offset = (page - 1) * page_size
    
    conn = get_tickets_connection()
    if not conn:
        raise FileNotFoundError(
            "Tickets database not found. Ensure Scraper/data/tickets.db exists "
            "or set TICKETS_DB_PATH environment variable."
        )
    
    try:
        cursor = conn.cursor()
        
        # Build WHERE clause for search
        where_clauses = []
        params = []
        
        if q:
            search_term = f"%{q}%"
            # Search in ticket_id, subject, and problem
            # Note: outcome is in raw_response_json (JSON), so we search the JSON string
            where_clauses.append(
                "(j.ticket_id LIKE ? OR i.subject LIKE ? OR j.problem LIKE ? OR j.raw_response_json LIKE ?)"
            )
            params.extend([search_term, search_term, search_term, search_term])
        
        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        # Build sort clause (sanitize to prevent SQL injection)
        allowed_sort_fields = {
            "ticket_id", "judged_at", "created_at", "updated_at", 
            "cache_eligible", "confidence", "outcome"
        }
        sort_parts = sort.split()
        if len(sort_parts) >= 2:
            field = sort_parts[0]
            direction = sort_parts[1].upper()
            if field in allowed_sort_fields and direction in ("ASC", "DESC"):
                order_by = f"ORDER BY {field} {direction}"
            else:
                order_by = "ORDER BY judged_at DESC"
        else:
            order_by = "ORDER BY judged_at DESC"
        
        # Query to get tickets with judgment and manual review data
        # Join tickets_index for subject/status, ticket_judgements for judgment data,
        # and ticket_manual_reviews for manual review status
        count_query = f"""
            SELECT COUNT(DISTINCT j.ticket_id) as total
            FROM ticket_judgements j
            LEFT JOIN tickets_index i ON j.ticket_id = i.ticket_id
            LEFT JOIN ticket_manual_reviews m ON j.ticket_id = m.ticket_id
            {where_sql}
        """
        
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]
        
        # Main query
        query = f"""
            SELECT 
                j.ticket_id,
                i.subject,
                i.status,
                i.created_at,
                i.updated_at,
                j.cache_eligible,
                j.confidence,
                j.review_status,
                j.judged_at,
                j.raw_response_json,
                m.manual_status,
                m.reviewer as manual_reviewer,
                m.reviewed_at as manual_reviewed_at
            FROM ticket_judgements j
            LEFT JOIN tickets_index i ON j.ticket_id = i.ticket_id
            LEFT JOIN ticket_manual_reviews m ON j.ticket_id = m.ticket_id
            {where_sql}
            {order_by}
            LIMIT ? OFFSET ?
        """
        
        params.extend([page_size, offset])
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Transform rows to response format
        items = []
        for row in rows:
            # Extract outcome from raw_response_json
            outcome = None
            raw_json = row["raw_response_json"]
            if raw_json:
                try:
                    data = json.loads(raw_json)
                    outcome = data.get("outcome")
                except Exception:
                    pass
            
            # Determine effective review status
            review_status = None
            if row["manual_status"]:
                review_status = "approved" if row["manual_status"] == "approved" else "rejected"
            elif row["review_status"]:
                review_status = row["review_status"]
            
            item = {
                "ticket_id": row["ticket_id"],
                "subject": row["subject"] or "",
                "status": row["status"] or "",
                "created_at": row["created_at"],
                "updated_at": row["updated_at"] or row["judged_at"],
                "cache_eligible": bool(row["cache_eligible"]),
                "review_status": review_status,
                "manual_status": row["manual_status"],
                "outcome": outcome,
                "confidence": float(row["confidence"]) if row["confidence"] is not None else 0.0,
                "has_confirmation": extract_confirmation_status(raw_json),
                "machine_models": extract_machine_models(raw_json),
            }
            items.append(item)
        
        return items, total
        
    finally:
        conn.close()
