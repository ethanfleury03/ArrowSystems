"""
SQLite database layer for storing scraped Zendesk tickets.

This module provides a simple storage layer for ticket data without using an ORM.
All operations use Python's stdlib sqlite3 module.

Two-stage pipeline:
- Stage 1: Index all requests (cheap) into tickets_index
- Stage 2: Build detailed conversations (expensive) only for solved tickets into tickets_detail
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default database path
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "tickets.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Get a SQLite connection with proper settings.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        sqlite3.Connection configured with row_factory
    """
    # Ensure parent directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initialize the database by creating tables if they don't exist.
    
    This function is idempotent and safe to call multiple times.
    
    Args:
        db_path: Path to the SQLite database file
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        
        # Create tickets_index table (Stage 1: cheap indexing)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tickets_index (
                ticket_id TEXT PRIMARY KEY,
                status TEXT,
                subject TEXT,
                requester_id TEXT,
                created_at TEXT,
                updated_at TEXT,
                is_solved INTEGER NOT NULL DEFAULT 0,
                indexed_at TEXT NOT NULL
            )
        """)
        
        # Create tickets_detail table (Stage 2: expensive conversation building)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tickets_detail (
                ticket_id TEXT PRIMARY KEY,
                conversation_json TEXT NOT NULL,
                built_at TEXT NOT NULL
            )
        """)
        
        # Create ticket_summaries table (Stage 3: structured problem/solution extraction)
        # Check if table exists with old schema and migrate if needed
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='ticket_summaries'
        """)
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            # Check if table has old schema (has problem_summary instead of problem_text)
            cursor.execute("PRAGMA table_info(ticket_summaries)")
            columns = [row[1] for row in cursor.fetchall()]
            if "problem_summary" in columns:
                # Old schema detected - drop and recreate
                cursor.execute("DROP TABLE IF EXISTS ticket_summaries")
                cursor.execute("DROP INDEX IF EXISTS idx_ticket_summaries_resolution_signal")
                table_exists = False
        
        if not table_exists:
            cursor.execute("""
                CREATE TABLE ticket_summaries (
                    ticket_id TEXT PRIMARY KEY,
                    subject TEXT,
                    status TEXT,
                    problem_text TEXT,
                    solution_text TEXT,
                    key_quotes TEXT,
                    resolution_confirmed INTEGER NOT NULL DEFAULT 0,
                    message_count INTEGER,
                    attachments_count INTEGER,
                    onsite_required INTEGER NOT NULL DEFAULT 0,
                    resolution_mode TEXT NOT NULL DEFAULT 'unknown',
                    resolution_mode_confidence REAL NOT NULL DEFAULT 0.0,
                    onsite_signals TEXT,
                    embedding_text TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    built_at TEXT NOT NULL
                )
            """)
        else:
            # Migrate existing table: add new columns if they don't exist
            cursor.execute("PRAGMA table_info(ticket_summaries)")
            existing_columns = [row[1] for row in cursor.fetchall()]
            
            # Add onsite_required column
            if "onsite_required" not in existing_columns:
                try:
                    cursor.execute("""
                        ALTER TABLE ticket_summaries 
                        ADD COLUMN onsite_required INTEGER NOT NULL DEFAULT 0
                    """)
                except sqlite3.OperationalError:
                    pass  # Column might already exist
            
            # Add resolution_mode column
            if "resolution_mode" not in existing_columns:
                try:
                    cursor.execute("""
                        ALTER TABLE ticket_summaries 
                        ADD COLUMN resolution_mode TEXT NOT NULL DEFAULT 'unknown'
                    """)
                except sqlite3.OperationalError:
                    pass
            
            # Add resolution_mode_confidence column
            if "resolution_mode_confidence" not in existing_columns:
                try:
                    cursor.execute("""
                        ALTER TABLE ticket_summaries 
                        ADD COLUMN resolution_mode_confidence REAL NOT NULL DEFAULT 0.0
                    """)
                except sqlite3.OperationalError:
                    pass
            
            # Add onsite_signals column
            if "onsite_signals" not in existing_columns:
                try:
                    cursor.execute("""
                        ALTER TABLE ticket_summaries 
                        ADD COLUMN onsite_signals TEXT
                    """)
                except sqlite3.OperationalError:
                    pass
            
            # Add embedding_text column
            if "embedding_text" not in existing_columns:
                try:
                    cursor.execute("""
                        ALTER TABLE ticket_summaries 
                        ADD COLUMN embedding_text TEXT
                    """)
                except sqlite3.OperationalError:
                    pass
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tickets_index_is_solved 
            ON tickets_index(is_solved)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tickets_index_status 
            ON tickets_index(status)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticket_summaries_resolution_confirmed 
            ON ticket_summaries(resolution_confirmed)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticket_summaries_status 
            ON ticket_summaries(status)
        """)
        
        # Create ticket_judgements table (LLM-based cache eligibility classification)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ticket_judgements (
                ticket_id TEXT PRIMARY KEY,
                cache_eligible INTEGER NOT NULL,
                confidence REAL NOT NULL,
                problem TEXT,
                resolution_steps_json TEXT,
                confirmation TEXT,
                evidence_json TEXT,
                blockers_json TEXT,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                judged_at TEXT NOT NULL,
                raw_response_json TEXT NOT NULL,
                review_status TEXT,
                review_reason TEXT,
                review_reasons_json TEXT,
                reviewed_at TEXT
            )
        """)
        
        # Migrate existing tables: add review_status columns if they don't exist
        cursor.execute("PRAGMA table_info(ticket_judgements)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "review_status" not in columns:
            cursor.execute("ALTER TABLE ticket_judgements ADD COLUMN review_status TEXT")
        if "review_reason" not in columns:
            cursor.execute("ALTER TABLE ticket_judgements ADD COLUMN review_reason TEXT")
        if "review_reasons_json" not in columns:
            cursor.execute("ALTER TABLE ticket_judgements ADD COLUMN review_reasons_json TEXT")
        if "reviewed_at" not in columns:
            cursor.execute("ALTER TABLE ticket_judgements ADD COLUMN reviewed_at TEXT")
        
        # Create indexes for ticket_judgements
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticket_judgements_cache_eligible 
            ON ticket_judgements(cache_eligible)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticket_judgements_review_status 
            ON ticket_judgements(review_status)
        """)
        
        # Create ticket_triage table (cheap model triage stage)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ticket_triage (
                ticket_id TEXT PRIMARY KEY,
                triage_label TEXT NOT NULL,
                triage_confidence REAL NOT NULL,
                triage_reason TEXT,
                triaged_at TEXT NOT NULL,
                triage_model TEXT NOT NULL,
                triage_prompt_version TEXT NOT NULL,
                triage_raw_response_json TEXT NOT NULL,
                FOREIGN KEY (ticket_id) REFERENCES tickets_detail(ticket_id)
            )
        """)
        
        # Create indexes for ticket_triage
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticket_triage_label 
            ON ticket_triage(triage_label)
        """)
        
        # Create ticket_manual_reviews table (manual override layer)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ticket_manual_reviews (
                ticket_id TEXT PRIMARY KEY,
                manual_status TEXT NOT NULL CHECK(manual_status IN ('approved', 'rejected')),
                manual_reason TEXT,
                manual_confirmation_quote TEXT,
                reviewer TEXT,
                reviewed_at TEXT NOT NULL,
                FOREIGN KEY (ticket_id) REFERENCES ticket_judgements(ticket_id)
            )
        """)
        
        # Create index for manual reviews
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticket_manual_reviews_status 
            ON ticket_manual_reviews(manual_status)
        """)
        
        conn.commit()
    finally:
        conn.close()


def upsert_ticket_index(conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
    """
    Insert or update a ticket in the tickets_index table.
    
    Args:
        conn: SQLite connection
        row: Dict with keys: ticket_id, status, subject, requester_id, created_at, updated_at, is_solved
    """
    indexed_at = datetime.now(timezone.utc).isoformat()
    
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tickets_index (
            ticket_id, status, subject, requester_id, 
            created_at, updated_at, is_solved, indexed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticket_id) DO UPDATE SET
            status = excluded.status,
            subject = excluded.subject,
            requester_id = excluded.requester_id,
            created_at = excluded.created_at,
            updated_at = excluded.updated_at,
            is_solved = excluded.is_solved,
            indexed_at = excluded.indexed_at
    """, (
        row["ticket_id"],
        row.get("status"),
        row.get("subject"),
        row.get("requester_id"),
        row.get("created_at"),
        row.get("updated_at"),
        row.get("is_solved", 0),
        indexed_at
    ))
    
    conn.commit()


def set_ticket_detail(conn: sqlite3.Connection, ticket_id: str, conversation: Dict[str, Any]) -> None:
    """
    Insert or update a ticket's detailed conversation JSON.
    
    Args:
        conn: SQLite connection
        ticket_id: Ticket ID
        conversation: Conversation dict to store as JSON
    """
    built_at = datetime.now(timezone.utc).isoformat()
    conversation_json = json.dumps(conversation, ensure_ascii=False)
    
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tickets_detail (ticket_id, conversation_json, built_at)
        VALUES (?, ?, ?)
        ON CONFLICT(ticket_id) DO UPDATE SET
            conversation_json = excluded.conversation_json,
            built_at = excluded.built_at
    """, (ticket_id, conversation_json, built_at))
    
    conn.commit()


def get_solved_ticket_ids(conn: sqlite3.Connection, statuses: tuple = ("solved", "closed")) -> List[str]:
    """
    Get list of solved ticket IDs from tickets_index.
    
    Args:
        conn: SQLite connection
        statuses: Tuple of status strings to consider solved (default: ("solved", "closed"))
        
    Returns:
        List of ticket IDs where is_solved=1
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ticket_id 
        FROM tickets_index 
        WHERE is_solved = 1
        ORDER BY ticket_id
    """)
    return [row["ticket_id"] for row in cursor.fetchall()]


def get_ticket_ids_without_detail(conn: sqlite3.Connection) -> List[str]:
    """
    Get list of solved ticket IDs that don't have detail records yet.
    
    Args:
        conn: SQLite connection
        
    Returns:
        List of ticket IDs that are solved but missing detail
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT i.ticket_id
        FROM tickets_index i
        LEFT JOIN tickets_detail d ON i.ticket_id = d.ticket_id
        WHERE i.is_solved = 1 AND d.ticket_id IS NULL
        ORDER BY i.ticket_id
    """)
    return [row["ticket_id"] for row in cursor.fetchall()]


def count_index(conn: sqlite3.Connection) -> int:
    """
    Count total tickets in tickets_index.
    
    Args:
        conn: SQLite connection
        
    Returns:
        Total count
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM tickets_index")
    row = cursor.fetchone()
    return row["count"] if row else 0


def count_detail(conn: sqlite3.Connection) -> int:
    """
    Count total tickets in tickets_detail.
    
    Args:
        conn: SQLite connection
        
    Returns:
        Total count
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM tickets_detail")
    row = cursor.fetchone()
    return row["count"] if row else 0


def count_solved(conn: sqlite3.Connection) -> int:
    """
    Count solved tickets in tickets_index.
    
    Args:
        conn: SQLite connection
        
    Returns:
        Count of solved tickets
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM tickets_index WHERE is_solved = 1")
    row = cursor.fetchone()
    return row["count"] if row else 0


def count_open(conn: sqlite3.Connection) -> int:
    """
    Count open tickets in tickets_index.
    
    Args:
        conn: SQLite connection
        
    Returns:
        Count of open tickets
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM tickets_index WHERE is_solved = 0")
    row = cursor.fetchone()
    return row["count"] if row else 0


def upsert_ticket_summary(
    conn: sqlite3.Connection,
    summary_dict: Dict[str, Any],
    rebuild: bool = False
) -> None:
    """
    Insert or update a ticket summary from a summary dictionary.
    
    Args:
        conn: SQLite connection
        summary_dict: Dictionary with summary fields:
            - ticket_id (required)
            - subject, status, problem_text, solution_text, key_quotes
            - resolution_confirmed (int, 0 or 1)
            - message_count, attachments_count (int)
            - created_at, updated_at
        rebuild: If True, overwrite existing; if False, skip if exists
    """
    built_at = datetime.now(timezone.utc).isoformat()
    ticket_id = summary_dict["ticket_id"]
    
    cursor = conn.cursor()
    
    if rebuild:
        # Upsert (insert or update)
        cursor.execute("""
            INSERT INTO ticket_summaries (
                ticket_id, subject, status, problem_text, solution_text,
                key_quotes, resolution_confirmed, message_count, attachments_count,
                onsite_required, resolution_mode, resolution_mode_confidence, onsite_signals,
                embedding_text, created_at, updated_at, built_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticket_id) DO UPDATE SET
                subject = excluded.subject,
                status = excluded.status,
                problem_text = excluded.problem_text,
                solution_text = excluded.solution_text,
                key_quotes = excluded.key_quotes,
                resolution_confirmed = excluded.resolution_confirmed,
                message_count = excluded.message_count,
                attachments_count = excluded.attachments_count,
                onsite_required = excluded.onsite_required,
                resolution_mode = excluded.resolution_mode,
                resolution_mode_confidence = excluded.resolution_mode_confidence,
                onsite_signals = excluded.onsite_signals,
                embedding_text = excluded.embedding_text,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                built_at = excluded.built_at
        """, (
            ticket_id,
            summary_dict.get("subject"),
            summary_dict.get("status"),
            summary_dict.get("problem_text"),
            summary_dict.get("solution_text"),
            summary_dict.get("key_quotes"),
            summary_dict.get("resolution_confirmed", 0),
            summary_dict.get("message_count"),
            summary_dict.get("attachments_count"),
            summary_dict.get("onsite_required", 0),
            summary_dict.get("resolution_mode", "unknown"),
            summary_dict.get("resolution_mode_confidence", 0.0),
            summary_dict.get("onsite_signals"),
            summary_dict.get("embedding_text"),
            summary_dict.get("created_at"),
            summary_dict.get("updated_at"),
            built_at
        ))
    else:
        # Insert only if not exists
        cursor.execute("""
            INSERT OR IGNORE INTO ticket_summaries (
                ticket_id, subject, status, problem_text, solution_text,
                key_quotes, resolution_confirmed, message_count, attachments_count,
                onsite_required, resolution_mode, resolution_mode_confidence, onsite_signals,
                embedding_text, created_at, updated_at, built_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticket_id,
            summary_dict.get("subject"),
            summary_dict.get("status"),
            summary_dict.get("problem_text"),
            summary_dict.get("solution_text"),
            summary_dict.get("key_quotes"),
            summary_dict.get("resolution_confirmed", 0),
            summary_dict.get("message_count"),
            summary_dict.get("attachments_count"),
            summary_dict.get("onsite_required", 0),
            summary_dict.get("resolution_mode", "unknown"),
            summary_dict.get("resolution_mode_confidence", 0.0),
            summary_dict.get("onsite_signals"),
            summary_dict.get("embedding_text"),
            summary_dict.get("created_at"),
            summary_dict.get("updated_at"),
            built_at
        ))
    
    conn.commit()


def get_ticket_detail_json(conn: sqlite3.Connection, ticket_id: str) -> Optional[Dict[str, Any]]:
    """
    Get ticket detail conversation JSON.
    
    Args:
        conn: SQLite connection
        ticket_id: Ticket ID
        
    Returns:
        Parsed conversation dict or None
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT conversation_json
        FROM tickets_detail
        WHERE ticket_id = ?
    """, (ticket_id,))
    row = cursor.fetchone()
    if row is None:
        return None
    
    try:
        return json.loads(row["conversation_json"])
    except (json.JSONDecodeError, TypeError):
        return None


def get_ticket_index(conn: sqlite3.Connection, ticket_id: str) -> Optional[Dict[str, Any]]:
    """
    Get ticket index record.
    
    Args:
        conn: SQLite connection
        ticket_id: Ticket ID
        
    Returns:
        Dict with ticket index data or None
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ticket_id, status, subject, requester_id, created_at, updated_at, is_solved
        FROM tickets_index
        WHERE ticket_id = ?
    """, (ticket_id,))
    row = cursor.fetchone()
    if row is None:
        return None
    return {
        "ticket_id": row["ticket_id"],
        "status": row["status"],
        "subject": row["subject"],
        "requester_id": row["requester_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "is_solved": bool(row["is_solved"])
    }


def get_ticket_detail(conn: sqlite3.Connection, ticket_id: str) -> Optional[Dict[str, Any]]:
    """
    Get ticket detail conversation JSON.
    
    Args:
        conn: SQLite connection
        ticket_id: Ticket ID
        
    Returns:
        Parsed conversation dict or None
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT conversation_json
        FROM tickets_detail
        WHERE ticket_id = ?
    """, (ticket_id,))
    row = cursor.fetchone()
    if row is None:
        return None
    
    try:
        return json.loads(row["conversation_json"])
    except (json.JSONDecodeError, TypeError):
        return None


def count_onsite_required(conn: sqlite3.Connection) -> int:
    """
    Count tickets that require onsite visits.
    
    Args:
        conn: SQLite connection
        
    Returns:
        Count of tickets with onsite_required=1
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM ticket_summaries WHERE onsite_required = 1")
    row = cursor.fetchone()
    return row["count"] if row else 0


def get_onsite_ticket_ids(conn: sqlite3.Connection, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get ticket IDs and basic info for tickets that require onsite visits.
    
    Args:
        conn: SQLite connection
        limit: Maximum number of tickets to return
        
    Returns:
        List of dicts with ticket_id, subject, resolution_mode, onsite_signals
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ticket_id, subject, resolution_mode, onsite_signals
        FROM ticket_summaries
        WHERE onsite_required = 1
        ORDER BY ticket_id
        LIMIT ?
    """, (limit,))
    return [
        {
            "ticket_id": row["ticket_id"],
            "subject": row["subject"],
            "resolution_mode": row["resolution_mode"],
            "onsite_signals": row["onsite_signals"]
        }
        for row in cursor.fetchall()
    ]


def upsert_ticket_judgement(conn: sqlite3.Connection, judgement_dict: Dict[str, Any]) -> None:
    """
    Insert or update a ticket judgement.
    
    Args:
        conn: SQLite connection
        judgement_dict: Dictionary with judgement fields:
            - ticket_id (required)
            - cache_eligible (int, 0 or 1)
            - confidence (float, 0.0-1.0)
            - problem, confirmation (str)
            - resolution_steps_json, evidence_json, blockers_json (JSON strings)
            - model, prompt_version (str)
            - raw_response_json (str, final verifier JSON)
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ticket_judgements (
            ticket_id, cache_eligible, confidence, problem, resolution_steps_json,
            confirmation, evidence_json, blockers_json, model, prompt_version,
            judged_at, raw_response_json, review_status, review_reason, review_reasons_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticket_id) DO UPDATE SET
            cache_eligible = excluded.cache_eligible,
            confidence = excluded.confidence,
            problem = excluded.problem,
            resolution_steps_json = excluded.resolution_steps_json,
            confirmation = excluded.confirmation,
            evidence_json = excluded.evidence_json,
            blockers_json = excluded.blockers_json,
            model = excluded.model,
            prompt_version = excluded.prompt_version,
            judged_at = excluded.judged_at,
            raw_response_json = excluded.raw_response_json,
            review_status = excluded.review_status,
            review_reason = excluded.review_reason,
            review_reasons_json = excluded.review_reasons_json
    """, (
        judgement_dict["ticket_id"],
        judgement_dict.get("cache_eligible", 0),
        judgement_dict.get("confidence", 0.0),
        judgement_dict.get("problem"),
        judgement_dict.get("resolution_steps_json"),
        judgement_dict.get("confirmation"),
        judgement_dict.get("evidence_json"),
        judgement_dict.get("blockers_json"),
        judgement_dict.get("model", ""),
        judgement_dict.get("prompt_version", ""),
        judgement_dict.get("judged_at", datetime.now(timezone.utc).isoformat()),
        judgement_dict.get("raw_response_json", "{}"),
        judgement_dict.get("review_status"),
        judgement_dict.get("review_reason"),
        judgement_dict.get("review_reasons_json")
    ))
    conn.commit()


def get_ticket_ids_needing_judgement(conn: sqlite3.Connection, only_solved: bool = True, limit: Optional[int] = None) -> List[str]:
    """
    Get ticket IDs that need judgement (don't have a judgement yet).
    
    Args:
        conn: SQLite connection
        only_solved: If True, only return solved tickets
        limit: Optional limit on number of tickets
        
    Returns:
        List of ticket IDs
    """
    cursor = conn.cursor()
    
    if only_solved:
        query = """
            SELECT d.ticket_id
            FROM tickets_detail d
            INNER JOIN tickets_index i ON d.ticket_id = i.ticket_id
            LEFT JOIN ticket_judgements j ON d.ticket_id = j.ticket_id
            WHERE i.is_solved = 1 AND j.ticket_id IS NULL
            ORDER BY d.ticket_id
        """
    else:
        query = """
            SELECT d.ticket_id
            FROM tickets_detail d
            LEFT JOIN ticket_judgements j ON d.ticket_id = j.ticket_id
            WHERE j.ticket_id IS NULL
            ORDER BY d.ticket_id
        """
    
    if limit:
        query += " LIMIT ?"
        cursor.execute(query, (limit,))
    else:
        cursor.execute(query)
    
    return [row["ticket_id"] for row in cursor.fetchall()]


def count_judged(conn: sqlite3.Connection) -> int:
    """
    Count tickets that have been judged.
    
    Args:
        conn: SQLite connection
        
    Returns:
        Count of judged tickets
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM ticket_judgements")
    row = cursor.fetchone()
    return row["count"] if row else 0


def count_cache_eligible(conn: sqlite3.Connection) -> int:
    """
    Count tickets that are cache eligible.
    
    Args:
        conn: SQLite connection
        
    Returns:
        Count of tickets with cache_eligible=1
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM ticket_judgements WHERE cache_eligible = 1")
    row = cursor.fetchone()
    return row["count"] if row else 0


def get_all_solved_ticket_ids(conn: sqlite3.Connection) -> List[str]:
    """
    Get all solved ticket IDs (alias for get_solved_ticket_ids).
    
    Args:
        conn: SQLite connection
        
    Returns:
        List of solved ticket IDs
    """
    return get_solved_ticket_ids(conn)


def upsert_ticket_triage(conn: sqlite3.Connection, triage_dict: Dict[str, Any]) -> None:
    """
    Insert or update a ticket triage result.
    
    Args:
        conn: SQLite connection
        triage_dict: Dictionary with triage fields:
            - ticket_id (required)
            - triage_label (deny/candidate/uncertain)
            - triage_confidence (float, 0.0-1.0)
            - triage_reason (str, optional)
            - triage_model (str)
            - triage_prompt_version (str)
            - triage_raw_response_json (str)
    """
    triaged_at = datetime.now(timezone.utc).isoformat()
    ticket_id = triage_dict["ticket_id"]
    
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ticket_triage (
            ticket_id, triage_label, triage_confidence, triage_reason,
            triaged_at, triage_model, triage_prompt_version, triage_raw_response_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticket_id) DO UPDATE SET
            triage_label = excluded.triage_label,
            triage_confidence = excluded.triage_confidence,
            triage_reason = excluded.triage_reason,
            triaged_at = excluded.triaged_at,
            triage_model = excluded.triage_model,
            triage_prompt_version = excluded.triage_prompt_version,
            triage_raw_response_json = excluded.triage_raw_response_json
    """, (
        ticket_id,
        triage_dict["triage_label"],
        triage_dict.get("triage_confidence", 0.0),
        triage_dict.get("triage_reason"),
        triaged_at,
        triage_dict["triage_model"],
        triage_dict["triage_prompt_version"],
        triage_dict["triage_raw_response_json"]
    ))
    conn.commit()


def get_ticket_triage(conn: sqlite3.Connection, ticket_id: str) -> Optional[Dict[str, Any]]:
    """
    Get triage result for a ticket.
    
    Args:
        conn: SQLite connection
        ticket_id: Ticket ID
        
    Returns:
        Triage dict or None if not found
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ticket_id, triage_label, triage_confidence, triage_reason,
               triaged_at, triage_model, triage_prompt_version, triage_raw_response_json
        FROM ticket_triage
        WHERE ticket_id = ?
    """, (ticket_id,))
    row = cursor.fetchone()
    if row:
        return {
            "ticket_id": row["ticket_id"],
            "triage_label": row["triage_label"],
            "triage_confidence": row["triage_confidence"],
            "triage_reason": row["triage_reason"],
            "triaged_at": row["triaged_at"],
            "triage_model": row["triage_model"],
            "triage_prompt_version": row["triage_prompt_version"],
            "triage_raw_response_json": row["triage_raw_response_json"]
        }
    return None


def upsert_manual_review(
    conn: sqlite3.Connection,
    ticket_id: str,
    manual_status: str,
    manual_reason: Optional[str] = None,
    manual_confirmation_quote: Optional[str] = None,
    reviewer: Optional[str] = None
) -> None:
    """
    Insert or update a manual review override.
    
    Args:
        conn: SQLite connection
        ticket_id: Ticket ID
        manual_status: 'approved' or 'rejected'
        manual_reason: Optional reason for manual decision
        manual_confirmation_quote: Optional confirmation quote if approved
        reviewer: Optional reviewer name/identifier
    """
    reviewed_at = datetime.now(timezone.utc).isoformat()
    
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ticket_manual_reviews (
            ticket_id, manual_status, manual_reason, manual_confirmation_quote,
            reviewer, reviewed_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticket_id) DO UPDATE SET
            manual_status = excluded.manual_status,
            manual_reason = excluded.manual_reason,
            manual_confirmation_quote = excluded.manual_confirmation_quote,
            reviewer = excluded.reviewer,
            reviewed_at = excluded.reviewed_at
    """, (
        ticket_id,
        manual_status,
        manual_reason,
        manual_confirmation_quote,
        reviewer,
        reviewed_at
    ))
    
    conn.commit()


def get_manual_review(conn: sqlite3.Connection, ticket_id: str) -> Optional[Dict[str, Any]]:
    """
    Get manual review override for a ticket.
    
    Args:
        conn: SQLite connection
        ticket_id: Ticket ID
        
    Returns:
        Manual review dict or None if not found
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ticket_id, manual_status, manual_reason, manual_confirmation_quote,
               reviewer, reviewed_at
        FROM ticket_manual_reviews
        WHERE ticket_id = ?
    """, (ticket_id,))
    row = cursor.fetchone()
    if row:
        return {
            "ticket_id": row["ticket_id"],
            "manual_status": row["manual_status"],
            "manual_reason": row["manual_reason"],
            "manual_confirmation_quote": row["manual_confirmation_quote"],
            "reviewer": row["reviewer"],
            "reviewed_at": row["reviewed_at"]
        }
    return None


if __name__ == "__main__":
    """
    CLI entrypoint: Initialize database and print counts.
    """
    print("Initializing database...")
    init_db()
    
    db_path = DEFAULT_DB_PATH
    print(f"\nDatabase path: {Path(db_path).absolute()}")
    
    conn = get_connection()
    try:
        index_count = count_index(conn)
        detail_count = count_detail(conn)
        solved_count = count_solved(conn)
        open_count = count_open(conn)
        
        print(f"\nIndexed tickets: {index_count}")
        print(f"  - Solved: {solved_count}")
        print(f"  - Open: {open_count}")
        print(f"\nDetailed conversations: {detail_count}")
        
    finally:
        conn.close()
