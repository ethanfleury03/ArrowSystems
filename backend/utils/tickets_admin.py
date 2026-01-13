"""
Admin utilities for reading tickets from Scraper database (SQLite or Postgres).

This module provides read-only access to ticket data stored in either
Scraper SQLite DB (local dev) or Postgres (production).
Handles graceful degradation when the DB is not available.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from sqlalchemy.engine import Engine

from ..logging_config import get_logger

logger = get_logger(__name__)


def _get_tickets_backend() -> str:
    """
    Get the tickets storage backend type.
    
    Returns:
        'postgres' or 'sqlite'
    """
    backend = os.getenv("TICKETS_STORAGE_BACKEND", "sqlite").lower()
    
    # In Cloud Run/prod, enforce postgres
    if os.getenv("K_SERVICE") or os.getenv("GAE_ENV"):
        if backend != "postgres":
            raise RuntimeError(
                "TICKETS_STORAGE_BACKEND must be 'postgres' in Cloud Run/production. "
                "SQLite is not available in containerized environments."
            )
        return "postgres"
    
    return backend


def get_tickets_db_path() -> Optional[Path]:
    """
    Get the path to the tickets SQLite database (only for sqlite backend).
    
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


def _get_sqlite_connection() -> Optional[sqlite3.Connection]:
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


def _get_postgres_engine() -> Engine:
    """
    Get a Postgres engine for ticket queries.
    
    Returns:
        SQLAlchemy Engine
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required for Postgres backend")
    
    return create_engine(database_url, poolclass=NullPool, future=True)


def extract_machine_models(raw_response_json: Optional[str], assigned_machine_model_ids: Optional[str] = None) -> List[str]:
    """
    Extract machine model names from assigned_machine_model_ids (from backfill) or raw_response_json.
    
    Args:
        raw_response_json: JSON string from ticket_judgements.raw_response_json (legacy)
        assigned_machine_model_ids: JSON array string from ticket_machine_model_assignment.machine_model_ids
        
    Returns:
        List of machine model names (empty list if none found)
    """
    # First, try to get from assigned machine model IDs (from backfill script)
    if assigned_machine_model_ids:
        try:
            model_ids = json.loads(assigned_machine_model_ids)
            if isinstance(model_ids, list) and model_ids:
                # Get model names from matches table
                # For now, return IDs - we could join with machine_models table later
                # But for display, IDs are fine since we can look them up
                return [str(mid) for mid in model_ids if mid]
        except Exception:
            pass
    
    # Fallback to raw_response_json (legacy)
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


def get_machine_model_names_from_ids_postgres(model_ids: List[int]) -> List[str]:
    """
    Get machine model names from IDs by querying PostgreSQL machine_models table.
    
    Args:
        model_ids: List of machine model IDs
        
    Returns:
        List of unique machine model names
    """
    if not model_ids:
        return []
    
    try:
        from ..utils.db import SessionLocal, MachineModel
        
        with SessionLocal() as pg_session:
            machine_models = pg_session.query(MachineModel).filter(
                MachineModel.id.in_(model_ids)
            ).all()
            
            if machine_models:
                # Return names in the order of IDs provided
                id_to_name = {model.id: model.name for model in machine_models}
                return [id_to_name[mid] for mid in model_ids if mid in id_to_name]
    except Exception as e:
        logger.debug(f"Failed to query PostgreSQL for machine model names: {e}")
    
    return []


def get_machine_model_names_from_ids(conn: sqlite3.Connection, model_ids: List[int]) -> List[str]:
    """
    Get machine model names from IDs by querying PostgreSQL machine_models table.
    Falls back to matches table if PostgreSQL query fails.
    
    Args:
        conn: SQLite connection (unused, kept for compatibility)
        model_ids: List of machine model IDs
        
    Returns:
        List of unique machine model names
    """
    if not model_ids:
        return []
    
    # Try PostgreSQL first (authoritative source)
    try:
        from ..utils.db import SessionLocal, MachineModel
        
        with SessionLocal() as pg_session:
            machine_models = pg_session.query(MachineModel).filter(
                MachineModel.id.in_(model_ids)
            ).all()
            
            if machine_models:
                # Return names in the order of IDs provided
                id_to_name = {model.id: model.name for model in machine_models}
                return [id_to_name[mid] for mid in model_ids if mid in id_to_name]
    except Exception as e:
        logger.debug(f"Failed to query PostgreSQL for machine model names: {e}")
    
    # Fallback: query matches table (may not have all models) - only for SQLite
    backend = _get_tickets_backend()
    if backend == "sqlite":
        try:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(model_ids))
            query = f"""
                SELECT DISTINCT machine_model_id, machine_model_name
                FROM ticket_machine_model_matches
                WHERE machine_model_id IN ({placeholders})
                ORDER BY machine_model_name
            """
            cursor.execute(query, model_ids)
            rows = cursor.fetchall()
            # Return names in the order of IDs provided
            id_to_name = {row[0]: row[1] for row in rows if row[1]}
            return [id_to_name[mid] for mid in model_ids if mid in id_to_name]
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
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Get a paginated list of tickets with their judgment data.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page (max 200)
        q: Optional search query (searches ticket_id, subject, outcome)
        sort: Sort order (default: "judged_at DESC")
        
    Returns:
        Tuple of (items list, total count, cache_eligible_total)
        
    Raises:
        FileNotFoundError: If tickets DB is not available
    """
    # Validate inputs
    page = max(1, page)
    page_size = min(max(1, page_size), 200)
    offset = (page - 1) * page_size
    
    backend = _get_tickets_backend()
    
    if backend == "postgres":
        return _get_tickets_page_postgres(page, page_size, offset, q, sort)
    else:
        return _get_tickets_page_sqlite(page, page_size, offset, q, sort)


def _get_tickets_page_postgres(
    page: int,
    page_size: int,
    offset: int,
    q: Optional[str],
    sort: str
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Postgres implementation of get_tickets_page."""
    engine = _get_postgres_engine()
    
    # Build WHERE clause for search
    where_clauses = []
    params: Dict[str, Any] = {}
    
    # Only show solved/completed tickets
    where_clauses.append("(i.is_solved = true OR i.status IN ('solved', 'closed'))")
    
    if q:
        search_term = f"%{q}%"
        # Search in ticket_id, subject, and problem
        # Note: outcome is in raw_response_json (JSONB), so we search the JSON string
        where_clauses.append(
            "(j.ticket_id LIKE :search_term OR i.subject LIKE :search_term OR j.problem LIKE :search_term OR j.raw_response_json::text LIKE :search_term)"
        )
        params["search_term"] = search_term
    
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
    
    with engine.connect() as conn:
        # Count query
        count_query = f"""
            SELECT COUNT(DISTINCT j.ticket_id) as total
            FROM ticket_judgements j
            LEFT JOIN tickets_index i ON j.ticket_id = i.ticket_id
            LEFT JOIN ticket_manual_reviews m ON j.ticket_id = m.ticket_id
            {where_sql}
        """
        result = conn.execute(text(count_query), params)
        total = result.scalar() or 0
        
        # Cache eligible count query
        cache_eligible_count_query = """
            SELECT COUNT(DISTINCT j.ticket_id) as cache_eligible_total
            FROM ticket_judgements j
            LEFT JOIN tickets_index i ON j.ticket_id = i.ticket_id
            WHERE j.cache_eligible = true
            AND (i.is_solved = true OR i.status IN ('solved', 'closed'))
        """
        result = conn.execute(text(cache_eligible_count_query))
        cache_eligible_total = result.scalar() or 0
        
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
                m.reviewed_at as manual_reviewed_at,
                a.machine_model_ids as assigned_machine_model_ids,
                a.status as assignment_status
            FROM ticket_judgements j
            LEFT JOIN tickets_index i ON j.ticket_id = i.ticket_id
            LEFT JOIN ticket_manual_reviews m ON j.ticket_id = m.ticket_id
            LEFT JOIN ticket_machine_model_assignment a ON j.ticket_id = a.ticket_id
            {where_sql}
            {order_by}
            LIMIT :page_size OFFSET :offset
        """
        params["page_size"] = page_size
        params["offset"] = offset
        
        result = conn.execute(text(query), params)
        rows = [dict(row._mapping) for row in result.fetchall()]
    
    # Transform rows to response format
    items = []
    for row in rows:
        # Extract outcome from raw_response_json (may be dict from JSONB or string)
        outcome = None
        raw_json = row["raw_response_json"]
        if raw_json:
            try:
                if isinstance(raw_json, str):
                    data = json.loads(raw_json)
                else:
                    data = raw_json
                outcome = data.get("outcome")
            except Exception:
                pass
        
        # Determine effective review status
        review_status = None
        if row["manual_status"]:
            review_status = "approved" if row["manual_status"] == "approved" else "rejected"
        elif row["review_status"]:
            review_status = row["review_status"]
        
        # Extract machine models from assignment table (preferred) or raw_response_json (fallback)
        assigned_model_ids = row["assigned_machine_model_ids"]
        machine_model_ids = []
        if assigned_model_ids:
            if isinstance(assigned_model_ids, list):
                machine_model_ids = assigned_model_ids
            elif isinstance(assigned_model_ids, str):
                try:
                    machine_model_ids = json.loads(assigned_model_ids)
                except Exception:
                    pass
        
        # Get machine model names from IDs (using Postgres connection)
        machine_model_names = get_machine_model_names_from_ids_postgres(machine_model_ids) if machine_model_ids else []
        
        # Fallback to extracting from raw_response_json if no assigned models
        if not machine_model_names:
            raw_json_str = json.dumps(raw_json) if isinstance(raw_json, dict) else (raw_json or "")
            machine_model_names = extract_machine_models(raw_json_str, None)
        
        item = {
            "ticket_id": row["ticket_id"],
            "subject": row["subject"] or "",
            "status": row["status"] or "",
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": (row["updated_at"] or row["judged_at"]).isoformat() if (row["updated_at"] or row["judged_at"]) else None,
            "cache_eligible": bool(row["cache_eligible"]),
            "review_status": review_status,
            "manual_status": row["manual_status"],
            "outcome": outcome,
            "confidence": float(row["confidence"]) if row["confidence"] is not None else 0.0,
            "has_confirmation": extract_confirmation_status(json.dumps(raw_json) if isinstance(raw_json, dict) else (raw_json or "")),
            "machine_models": machine_model_names,
        }
        items.append(item)
    
    return items, total, cache_eligible_total


def _get_tickets_page_sqlite(
    page: int,
    page_size: int,
    offset: int,
    q: Optional[str],
    sort: str
) -> Tuple[List[Dict[str, Any]], int, int]:
    """SQLite implementation of get_tickets_page."""
    conn = _get_sqlite_connection()
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
        
        # Only show solved/completed tickets
        where_clauses.append("(i.is_solved = 1 OR i.status IN ('solved', 'closed'))")
        
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
        
        # Query to get total cache-eligible count (across ALL solved tickets, not filtered by search)
        # This gives users a sense of the overall dataset
        cache_eligible_count_query = """
            SELECT COUNT(DISTINCT j.ticket_id) as cache_eligible_total
            FROM ticket_judgements j
            LEFT JOIN tickets_index i ON j.ticket_id = i.ticket_id
            WHERE j.cache_eligible = 1
            AND (i.is_solved = 1 OR i.status IN ('solved', 'closed'))
        """
        
        cursor.execute(cache_eligible_count_query)
        cache_eligible_total = cursor.fetchone()[0]
        
        # Main query - join with machine model assignment table
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
                m.reviewed_at as manual_reviewed_at,
                a.machine_model_ids as assigned_machine_model_ids,
                a.status as assignment_status
            FROM ticket_judgements j
            LEFT JOIN tickets_index i ON j.ticket_id = i.ticket_id
            LEFT JOIN ticket_manual_reviews m ON j.ticket_id = m.ticket_id
            LEFT JOIN ticket_machine_model_assignment a ON j.ticket_id = a.ticket_id
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
            
            # Extract machine models from assignment table (preferred) or raw_response_json (fallback)
            assigned_model_ids = row["assigned_machine_model_ids"]
            machine_model_ids = []
            if assigned_model_ids:
                try:
                    machine_model_ids = json.loads(assigned_model_ids)
                except Exception:
                    pass
            
            # Get machine model names from IDs
            machine_model_names = get_machine_model_names_from_ids(conn, machine_model_ids) if machine_model_ids else []
            
            # Fallback to extracting from raw_response_json if no assigned models
            if not machine_model_names:
                machine_model_names = extract_machine_models(raw_json, None)
            
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
                "machine_models": machine_model_names,
            }
            items.append(item)
        
        return items, total, cache_eligible_total
        
    finally:
        conn.close()


def update_ticket(
    ticket_id: str,
    subject: Optional[str] = None,
    status: Optional[str] = None,
    cache_eligible: Optional[bool] = None,
    confidence: Optional[float] = None,
    review_status: Optional[str] = None,
    outcome: Optional[str] = None,
    machine_model_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Update ticket fields in the SQLite database.
    
    Args:
        ticket_id: Ticket ID to update
        subject: Optional subject to update in tickets_index
        status: Optional status to update in tickets_index
        cache_eligible: Optional cache_eligible flag to update in ticket_judgements
        confidence: Optional confidence score to update in ticket_judgements
        review_status: Optional review_status to update in ticket_judgements
        outcome: Optional outcome to update in raw_response_json
        machine_model_names: Optional list of machine model names to update in assignment table
        
    Returns:
        Updated ticket data
        
    Raises:
        FileNotFoundError: If tickets DB is not available
        ValueError: If ticket_id not found
    """
    conn = get_tickets_connection()
    if not conn:
        raise FileNotFoundError(
            "Tickets database not found. Ensure Scraper/data/tickets.db exists "
            "or set TICKETS_DB_PATH environment variable."
        )
    
    try:
        cursor = conn.cursor()
        
        # Check if ticket exists
        cursor.execute("SELECT ticket_id FROM ticket_judgements WHERE ticket_id = ?", (ticket_id,))
        if not cursor.fetchone():
            raise ValueError(f"Ticket {ticket_id} not found")
        
        # Update tickets_index if subject or status provided
        if subject is not None or status is not None:
            updates = []
            params = []
            if subject is not None:
                updates.append("subject = ?")
                params.append(subject)
            if status is not None:
                updates.append("status = ?")
                params.append(status)
            if updates:
                updates.append("updated_at = ?")
                params.append(datetime.now(timezone.utc).isoformat())
                params.append(ticket_id)
                cursor.execute(
                    f"UPDATE tickets_index SET {', '.join(updates)} WHERE ticket_id = ?",
                    params
                )
        
        # Update ticket_judgements if cache_eligible, confidence, or review_status provided
        if cache_eligible is not None or confidence is not None or review_status is not None or outcome is not None:
            updates = []
            params = []
            
            if cache_eligible is not None:
                updates.append("cache_eligible = ?")
                params.append(1 if cache_eligible else 0)
            
            if confidence is not None:
                updates.append("confidence = ?")
                params.append(confidence)
            
            if review_status is not None:
                updates.append("review_status = ?")
                params.append(review_status)
            
            if outcome is not None:
                # Update outcome in raw_response_json
                cursor.execute("SELECT raw_response_json FROM ticket_judgements WHERE ticket_id = ?", (ticket_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    try:
                        raw_json = json.loads(row[0])
                        raw_json["outcome"] = outcome
                        updates.append("raw_response_json = ?")
                        params.append(json.dumps(raw_json))
                    except Exception:
                        # If JSON parsing fails, create new structure
                        updates.append("raw_response_json = ?")
                        params.append(json.dumps({"outcome": outcome}))
                else:
                    updates.append("raw_response_json = ?")
                    params.append(json.dumps({"outcome": outcome}))
            
            if updates:
                params.append(ticket_id)
                cursor.execute(
                    f"UPDATE ticket_judgements SET {', '.join(updates)} WHERE ticket_id = ?",
                    params
                )
        
        # Update machine model assignment if provided
        if machine_model_names is not None:
            # Get machine model IDs from names (query PostgreSQL machine_models table)
            model_ids = []
            if machine_model_names:
                try:
                    # Import here to avoid circular dependencies
                    from ..utils.db import SessionLocal, MachineModel
                    from sqlalchemy import func
                    
                    with SessionLocal() as pg_session:
                        # Query PostgreSQL for machine model IDs by name (case-insensitive)
                        machine_models = pg_session.query(MachineModel).filter(
                            func.upper(MachineModel.name).in_([name.upper() for name in machine_model_names])
                        ).all()
                        
                        # Create a mapping of upper name to ID
                        name_to_id = {model.name.upper(): model.id for model in machine_models}
                        
                        # Match input names to IDs (preserving order)
                        for name in machine_model_names:
                            matched_id = name_to_id.get(name.upper())
                            if matched_id and matched_id not in model_ids:
                                model_ids.append(matched_id)
                        
                        # Log warning if some names weren't found
                        found_names = {name.upper() for name in machine_model_names if name.upper() in name_to_id}
                        missing_names = [name for name in machine_model_names if name.upper() not in found_names]
                        if missing_names:
                            logger.warning(f"Some machine model names not found in database: {missing_names}")
                except Exception as e:
                    logger.error(f"Failed to lookup machine model IDs from PostgreSQL: {e}", exc_info=True)
                    # Fallback: try to find IDs from matches table (across all tickets)
                    if machine_model_names:
                        placeholders = ",".join("?" * len(machine_model_names))
                        cursor.execute(f"""
                            SELECT DISTINCT machine_model_id, machine_model_name
                            FROM ticket_machine_model_matches
                            WHERE machine_model_name IN ({placeholders})
                        """, machine_model_names)
                        matches = cursor.fetchall()
                        model_ids = [row[0] for row in matches]
            else:
                model_ids = []
            
            # Update assignment table
            if model_ids or not machine_model_names:  # Update even if clearing (empty list)
                assignment_status = "assigned" if len(model_ids) == 1 else ("ambiguous" if len(model_ids) > 1 else "unassigned")
                confidence_score = 1.0 if len(model_ids) == 1 else (0.8 if len(model_ids) > 1 else 0.0)
                
                cursor.execute("""
                    INSERT INTO ticket_machine_model_assignment (
                        ticket_id, machine_model_ids, status, confidence, method, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticket_id) DO UPDATE SET
                        machine_model_ids = excluded.machine_model_ids,
                        status = excluded.status,
                        confidence = excluded.confidence,
                        updated_at = excluded.updated_at
                """, (
                    ticket_id,
                    json.dumps(model_ids),
                    assignment_status,
                    confidence_score,
                    "manual_edit",
                    datetime.now(timezone.utc).isoformat()
                ))
        
        conn.commit()
        
        # Return updated ticket data
        items, _, _ = get_tickets_page(page=1, page_size=1, q=ticket_id)
        if items:
            return items[0]
        else:
            raise ValueError(f"Failed to retrieve updated ticket {ticket_id}")
        
    finally:
        conn.close()


def get_ticket_details(ticket_id: str) -> Dict[str, Any]:
    """
    Get full ticket details including conversation.
    
    Args:
        ticket_id: Ticket ID to fetch
        
    Returns:
        Dictionary with ticket details including conversation, judgment data, etc.
        
    Raises:
        FileNotFoundError: If tickets DB is not available
        ValueError: If ticket_id not found
    """
    conn = get_tickets_connection()
    if not conn:
        raise FileNotFoundError(
            "Tickets database not found. Ensure Scraper/data/tickets.db exists "
            "or set TICKETS_DB_PATH environment variable."
        )
    
    try:
        cursor = conn.cursor()
        
        # Get ticket index data
        cursor.execute("""
            SELECT ticket_id, subject, status, requester_id, created_at, updated_at, is_solved
            FROM tickets_index
            WHERE ticket_id = ?
        """, (ticket_id,))
        index_row = cursor.fetchone()
        if not index_row:
            raise ValueError(f"Ticket {ticket_id} not found")
        
        # Get conversation JSON from tickets_detail
        cursor.execute("""
            SELECT conversation_json
            FROM tickets_detail
            WHERE ticket_id = ?
        """, (ticket_id,))
        detail_row = cursor.fetchone()
        conversation_json = None
        if detail_row and detail_row[0]:
            try:
                conversation_json = json.loads(detail_row[0])
            except Exception:
                pass
        
        # Get judgment data
        cursor.execute("""
            SELECT cache_eligible, confidence, review_status, judged_at, raw_response_json
            FROM ticket_judgements
            WHERE ticket_id = ?
        """, (ticket_id,))
        judgment_row = cursor.fetchone()
        
        # Extract outcome and other fields from raw_response_json
        outcome = None
        problem = None
        resolution_steps = None
        confirmation = None
        rationale = None
        blockers = None
        raw_json = None
        
        if judgment_row and judgment_row["raw_response_json"]:
            try:
                raw_json = json.loads(judgment_row["raw_response_json"])
                outcome = raw_json.get("outcome")
                problem = raw_json.get("problem")
                resolution = raw_json.get("resolution", {})
                if isinstance(resolution, dict):
                    resolution_steps = resolution.get("steps")
                confirmation = raw_json.get("confirmation")
                rationale = raw_json.get("rationale")
                blockers = raw_json.get("blockers")
            except Exception:
                pass
        
        # Get machine models
        cursor.execute("""
            SELECT machine_model_ids
            FROM ticket_machine_model_assignment
            WHERE ticket_id = ?
        """, (ticket_id,))
        assignment_row = cursor.fetchone()
        machine_model_ids = []
        if assignment_row and assignment_row[0]:
            try:
                machine_model_ids = json.loads(assignment_row[0])
            except Exception:
                pass
        
        machine_model_names = get_machine_model_names_from_ids(conn, machine_model_ids) if machine_model_ids else []
        
        # Get manual review status
        cursor.execute("""
            SELECT manual_status, reviewer, reviewed_at
            FROM ticket_manual_reviews
            WHERE ticket_id = ?
        """, (ticket_id,))
        review_row = cursor.fetchone()
        
        return {
            "ticket_id": index_row["ticket_id"],
            "subject": index_row["subject"] or "",
            "status": index_row["status"] or "",
            "requester_id": index_row["requester_id"],
            "created_at": index_row["created_at"],
            "updated_at": index_row["updated_at"],
            "is_solved": bool(index_row["is_solved"]),
            "conversation": conversation_json,
            "cache_eligible": bool(judgment_row["cache_eligible"]) if judgment_row else False,
            "confidence": float(judgment_row["confidence"]) if judgment_row and judgment_row["confidence"] is not None else 0.0,
            "review_status": judgment_row["review_status"] if judgment_row else None,
            "judged_at": judgment_row["judged_at"] if judgment_row else None,
            "outcome": outcome,
            "problem": problem,
            "resolution_steps": resolution_steps,
            "confirmation": confirmation,
            "rationale": rationale,
            "blockers": blockers,
            "machine_models": machine_model_names,
            "manual_status": review_row["manual_status"] if review_row else None,
            "manual_reviewer": review_row["reviewer"] if review_row else None,
            "manual_reviewed_at": review_row["reviewed_at"] if review_row else None,
        }
        
    finally:
        conn.close()
