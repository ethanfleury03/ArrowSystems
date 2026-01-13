#!/usr/bin/env python3
"""
One-time migration script: SQLite tickets -> Postgres.

Reads tickets from Scraper SQLite database and migrates them to Postgres.
Idempotent: safe to rerun (uses UPSERT).

Usage:
    # Dry-run (rollback transaction)
    python -m backend.scripts.migrate_tickets_sqlite_to_postgres --dry-run --sqlite-path Scraper/data/tickets.db
    
    # Dry-run with log file
    python -m backend.scripts.migrate_tickets_sqlite_to_postgres --dry-run --sqlite-path Scraper/data/tickets.db --log-file out/migrate_dryrun.log
    
    # Real migration
    python -m backend.scripts.migrate_tickets_sqlite_to_postgres --sqlite-path Scraper/data/tickets.db
    
    # Limit to specific tables
    python -m backend.scripts.migrate_tickets_sqlite_to_postgres --tables tickets_index,ticket_judgements
    
    # Limit number of rows per table
    python -m backend.scripts.migrate_tickets_sqlite_to_postgres --limit 100
"""

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load .env file from backend directory BEFORE importing backend modules
try:
    from dotenv import load_dotenv
    backend_dir = project_root / "backend"
    env_path = backend_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)  # override=True ensures .env values take precedence
        print(f"[OK] Loaded environment variables from {env_path}", file=sys.stderr)
    else:
        print(f"[WARNING] .env file not found at {env_path}", file=sys.stderr)
        print(f"   DATABASE_URL must be set as environment variable", file=sys.stderr)
except ImportError:
    print("[WARNING] python-dotenv not installed.", file=sys.stderr)
    print("   Install with: pip install python-dotenv", file=sys.stderr)
    print("   Or set DATABASE_URL as environment variable", file=sys.stderr)

# Verify DATABASE_URL is available before importing backend modules
if not os.getenv("DATABASE_URL"):
    print("[ERROR] DATABASE_URL not found after loading .env", file=sys.stderr)
    print("   Ensure backend/.env contains DATABASE_URL=...", file=sys.stderr)
    sys.exit(1)

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import IntegrityError

# Note: We don't import backend.utils.db models here because they trigger
# settings initialization which requires DATABASE_URL. The migration script
# uses raw SQLAlchemy Core queries instead.


@dataclass
class MigrationContext:
    """Context object passed to all migrator functions."""
    sqlite_conn: sqlite3.Connection
    session: Session
    dry_run: bool
    orphan_policy: str  # "skip" | "backfill" | "error"
    parent_ids: set[str] = field(default_factory=set)  # ticket_ids present in Postgres tickets_index
    stats: Dict[str, Any] = field(default_factory=dict)  # per-table counters + error samples
    args: Optional[Any] = None  # argparse.Namespace for verbose/debug caps
    table_name: str = ""  # current table being migrated
    rows: List[sqlite3.Row] = field(default_factory=list)  # current table rows


def parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except Exception:
        return None


class TeeOutput:
    """Write to both stdout/stderr and a log file."""
    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file
        self.log_handle: Optional[TextIO] = None
        if log_file:
            # Ensure directory exists
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self.log_handle = open(log_file, 'w', encoding='utf-8', buffering=1)  # Line buffered
    
    def write(self, text: str, file: Optional[TextIO] = None):
        """Write text to both stdout/stderr and log file."""
        target = file or sys.stdout
        target.write(text)
        target.flush()
        if self.log_handle:
            self.log_handle.write(text)
            self.log_handle.flush()
    
    def print(self, *args, file: Optional[TextIO] = None, **kwargs):
        """Print to both stdout/stderr and log file."""
        # Convert args to string
        text = ' '.join(str(arg) for arg in args)
        if kwargs.get('end', '\n') != '\n':
            text += kwargs.get('end', '\n')
        else:
            text += '\n'
        self.write(text, file=file)
    
    def close(self):
        """Close the log file handle."""
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def to_jsonb(value: Any, default: Any = None) -> Optional[str]:
    """
    Convert value to JSONB-compatible JSON string.
    
    Args:
        value: Value to convert (can be dict, list, str, None)
        default: Default value if value is None (e.g., '{}' or '[]')
    
    Returns:
        JSON string ready for CAST(:param AS jsonb), or None
    """
    if value is None:
        if default is not None:
            return json.dumps(default, ensure_ascii=False)
        return None
    
    # If already a JSON string, validate and return
    if isinstance(value, str):
        try:
            # Validate it's valid JSON
            json.loads(value)
            return value
        except json.JSONDecodeError:
            # Not valid JSON, try to parse as Python object
            try:
                parsed = json.loads(value)
                return json.dumps(parsed, ensure_ascii=False)
            except:
                # If still fails, wrap as string value
                return json.dumps(value, ensure_ascii=False)
    
    # Convert Python object to JSON string
    return json.dumps(value, ensure_ascii=False)


def _truncate(s: str, n: int = 300) -> str:
    """Truncate string to n characters with truncation indicator."""
    if len(s) <= n:
        return s
    return s[:n] + f"...({len(s)-n} chars truncated)"


def _safe_param_view(params: Dict[str, Any]) -> Dict[str, str]:
    """Convert params dict to safe string representation for logging."""
    safe = {}
    for k, v in params.items():
        if isinstance(v, str):
            if len(v) > 300:
                safe[k] = _truncate(v, 300)
            else:
                safe[k] = v
        elif isinstance(v, (dict, list)):
            json_str = json.dumps(v, ensure_ascii=False)
            safe[k] = f"<json {len(json_str)} chars>"
        elif isinstance(v, datetime):
            safe[k] = v.isoformat()
        elif v is None:
            safe[k] = "None"
        else:
            safe[k] = repr(v)[:300]
    return safe


def compact_exc(e: Exception) -> str:
    """Get compact exception representation."""
    exc_class = e.__class__.__name__
    exc_msg = str(e).splitlines()[0][:200]
    return f"{exc_class}: {exc_msg}"


def _log_error(context: str, ticket_id: str, exc: Exception, sql: Optional[str] = None, 
               params: Optional[Dict[str, Any]] = None, debug_sql: bool = False):
    """Log error in concise format without huge SQL dumps."""
    exc_compact = compact_exc(exc)
    print(f"ERROR {context} ticket_id={ticket_id} {exc_compact}")
    
    if debug_sql:
        if sql:
            # Show only first line of SQL
            sql_first_line = sql.strip().split('\n')[0]
            print(f"  SQL: {_truncate(sql_first_line, 200)}")
        
        if params:
            safe_params = _safe_param_view(params)
            print(f"  params: {_truncate(str(safe_params), 300)}")


def _get_error_signature(exc: Exception) -> str:
    """Get error signature for deduplication."""
    exc_class = exc.__class__.__name__
    exc_msg = str(exc)[:120]
    return f"{exc_class}: {exc_msg}"


def _is_test_ticket_id(ticket_id: str) -> bool:
    """Check if ticket_id looks like a test ID."""
    ticket_id_upper = ticket_id.upper()
    return (
        ticket_id_upper.startswith("TEST") or
        ticket_id_upper.startswith("DEMO") or
        (not ticket_id.isdigit() and len(ticket_id) < 10)
    )


def _check_orphan(ctx: MigrationContext, ticket_id: str) -> tuple[bool, Optional[str]]:
    """
    Check if ticket_id is orphan and handle according to policy.
    Orphan gate: check BEFORE touching Postgres.
    
    Returns:
        (should_skip: bool, action_taken: Optional[str])
    """
    verbose = ctx.args.verbose if ctx.args else False
    max_error_lines = ctx.args.max_error_lines if ctx.args else 25
    
    # Track printed skips for this table
    if ctx.table_name not in ctx.stats:
        ctx.stats[ctx.table_name] = {"printed_skips": 0}
    printed_skips = ctx.stats[ctx.table_name].get("printed_skips", 0)
    
    # Check if test ticket
    is_test = _is_test_ticket_id(ticket_id)
    
    # Check if orphan (missing parent)
    is_orphan = ticket_id not in ctx.parent_ids
    
    if is_test:
        if verbose and printed_skips < max_error_lines:
            print(f"SKIP test ticket_id={ticket_id}")
            ctx.stats[ctx.table_name]["printed_skips"] += 1
        return True, "test"
    
    if is_orphan:
        if ctx.orphan_policy == "skip":
            if verbose and printed_skips < max_error_lines:
                print(f"SKIP orphan ticket_id={ticket_id} (not in tickets_index)")
                ctx.stats[ctx.table_name]["printed_skips"] += 1
            return True, "orphan"
        elif ctx.orphan_policy == "backfill":
            # Insert minimal parent row
            try:
                with ctx.session.begin_nested():
                    ctx.session.execute(
                        text("""
                            INSERT INTO tickets_index (ticket_id, status, is_solved, indexed_at)
                            VALUES (:ticket_id, 'unknown', false, NOW())
                            ON CONFLICT (ticket_id) DO NOTHING
                        """),
                        {"ticket_id": ticket_id}
                    )
                if verbose and printed_skips < max_error_lines:
                    print(f"BACKFILL parent ticket_id={ticket_id}")
                    ctx.stats[ctx.table_name]["printed_skips"] += 1
                # Update parent set
                ctx.parent_ids.add(ticket_id)
                return False, "backfilled"
            except Exception as e:
                if verbose and printed_skips < max_error_lines:
                    print(f"BACKFILL FAILED ticket_id={ticket_id}: {compact_exc(e)}")
                    ctx.stats[ctx.table_name]["printed_skips"] += 1
                return True, "backfill_failed"
        elif ctx.orphan_policy == "error":
            # Will be handled as error in main loop (don't skip, let it fail)
            return False, None
    
    return False, None


def _execute_with_savepoint(ctx: MigrationContext, sql: str, params: Dict[str, Any], 
                           ticket_id: str) -> bool:
    """
    Execute SQL with savepoint to prevent cascade failures.
    Ensures per-row isolation by rolling back on any error.
    
    Returns:
        success: bool
    """
    max_error_lines = ctx.args.max_error_lines if ctx.args else 25
    debug_sql = ctx.args.debug_sql if ctx.args else False
    
    # Track printed errors for this table
    if ctx.table_name not in ctx.stats:
        ctx.stats[ctx.table_name] = {"printed_errors": 0, "error_tracker": {}}
    
    printed_errors = ctx.stats[ctx.table_name]["printed_errors"]
    error_tracker = ctx.stats[ctx.table_name]["error_tracker"]
    
    try:
        # Use savepoint so we can rollback just this row
        with ctx.session.begin_nested():
            ctx.session.execute(text(sql), params)
        return True
    except Exception as e:
        # CRITICAL: Rollback the session to ensure it's usable for next row
        # Even with begin_nested(), SQLAlchemy sessions can end up "failed" after DBAPI errors
        try:
            ctx.session.rollback()
        except:
            pass
        
        # Log the real error (only if under limit)
        if printed_errors < max_error_lines:
            _log_error(ctx.table_name, ticket_id, e, sql=sql, params=params, debug_sql=debug_sql)
            ctx.stats[ctx.table_name]["printed_errors"] += 1
        
        error_sig = _get_error_signature(e)
        if error_sig not in error_tracker:
            error_tracker[error_sig] = []
        if len(error_tracker[error_sig]) < 3:
            error_tracker[error_sig].append(ticket_id)
        
        return False


def _get_required_parent_ids(sqlite_conn: sqlite3.Connection) -> set[str]:
    """
    Preflight: Get union of all ticket_ids that appear in ANY ticket table in SQLite.
    This ensures we know which parent IDs are needed before migrating child tables.
    """
    cursor = sqlite_conn.cursor()
    required_ids = set()
    
    # Get ticket_ids from all tables
    tables = [
        "tickets_index",
        "tickets_detail",
        "ticket_summaries",
        "ticket_judgements",
        "ticket_triage",
        "ticket_manual_reviews",
        "ticket_machine_model_matches",
        "ticket_machine_model_assignment"
    ]
    
    for table in tables:
        try:
            cursor.execute(f"SELECT DISTINCT ticket_id FROM {table}")
            for row in cursor.fetchall():
                ticket_id = str(row[0])
                if ticket_id and not _is_test_ticket_id(ticket_id):
                    required_ids.add(ticket_id)
        except sqlite3.OperationalError:
            # Table doesn't exist, skip
            pass
    
    return required_ids


def _backfill_missing_parents(ctx: MigrationContext, missing_ids: set[str]) -> int:
    """
    Backfill missing parent ticket_ids into tickets_index.
    Returns number of backfilled IDs.
    """
    if ctx.orphan_policy != "backfill":
        return 0
    
    backfilled = 0
    for ticket_id in missing_ids:
        try:
            with ctx.session.begin_nested():
                ctx.session.execute(
                    text("""
                        INSERT INTO tickets_index (ticket_id, status, is_solved, indexed_at)
                        VALUES (:ticket_id, 'unknown', false, NOW())
                        ON CONFLICT (ticket_id) DO NOTHING
                    """),
                    {"ticket_id": ticket_id}
                )
            ctx.parent_ids.add(ticket_id)
            backfilled += 1
        except Exception as e:
            # Log but continue
            if ctx.args and ctx.args.verbose:
                print(f"  BACKFILL FAILED ticket_id={ticket_id}: {compact_exc(e)}")
    
    return backfilled


def migrate_table(ctx: MigrationContext, table_name: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Migrate a single table from SQLite to Postgres using MigrationContext.
    
    Returns:
        Dict with counts: total, inserted, updated, errors, skipped_orphan, skipped_test
    """
    print(f"\nMigrating {table_name}...")
    
    # Get all rows from SQLite
    cursor = ctx.sqlite_conn.cursor()
    
    if limit:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT ?", (limit,))
    else:
        cursor.execute(f"SELECT * FROM {table_name}")
    
    rows = cursor.fetchall()
    total = len(rows)
    
    if total == 0:
        print(f"  No rows to migrate")
        return {"total": 0, "inserted": 0, "updated": 0, "errors": 0, "skipped_orphan": 0, "skipped_test": 0}
    
    print(f"  Found {total} rows")
    
    # Set context for this table
    ctx.table_name = table_name
    ctx.rows = rows
    
    # Initialize stats for this table
    if table_name not in ctx.stats:
        ctx.stats[table_name] = {
            "inserted": 0,
            "errors": 0,
            "skipped_orphan": 0,
            "skipped_test": 0,
            "printed_errors": 0,
            "printed_skips": 0,
            "error_tracker": {}
        }
    
    # Table-specific migration logic (all use unified MigrationContext signature)
    if table_name == "tickets_index":
        _migrate_tickets_index(ctx)
    elif table_name == "tickets_detail":
        _migrate_tickets_detail(ctx)
    elif table_name == "ticket_summaries":
        _migrate_ticket_summaries(ctx)
    elif table_name == "ticket_judgements":
        _migrate_ticket_judgements(ctx)
    elif table_name == "ticket_triage":
        _migrate_ticket_triage(ctx)
    elif table_name == "ticket_manual_reviews":
        _migrate_ticket_manual_reviews(ctx)
    elif table_name == "ticket_machine_model_matches":
        _migrate_ticket_machine_model_matches(ctx)
    elif table_name == "ticket_machine_model_assignment":
        _migrate_ticket_machine_model_assignment(ctx)
    elif table_name == "scrape_runs":
        _migrate_scrape_runs(ctx)
    else:
        print(f"  WARNING: Unknown table {table_name}, skipping")
        return {"total": total, "inserted": 0, "updated": 0, "errors": total, "skipped_orphan": 0, "skipped_test": 0}
    
    # Extract results from stats
    stats = ctx.stats[table_name]
    result = {
        "total": total,
        "inserted": stats.get("inserted", 0),
        "updated": 0,
        "errors": stats.get("errors", 0),
        "skipped_orphan": stats.get("skipped_orphan", 0),
        "skipped_test": stats.get("skipped_test", 0)
    }
    
    # Print summary
    cascade_count = len(stats.get("error_tracker", {}).get("_cascade_count", []))
    if cascade_count > 0:
        print(f"  [WARN] {cascade_count} rows failed due to aborted transaction; see earlier root error(s).")
    
    print(f"  [TABLE SUMMARY] {table_name}: processed={total} migrated={result['inserted']} errors={result['errors']} skipped_orphan={result['skipped_orphan']} skipped_test={result['skipped_test']}")
    
    error_tracker = stats.get("error_tracker", {})
    if error_tracker and len(error_tracker) > (1 if "_cascade_count" in error_tracker else 0):
        print(f"  [TOP ERRORS]")
        for i, (error_sig, example_ids) in enumerate(error_tracker.items()):
            if error_sig != "_cascade_count" and i < 5:
                print(f"    - {error_sig} (example ticket_id={example_ids[0] if example_ids else 'unknown'})")
    
    return result


def _migrate_tickets_index(ctx: MigrationContext) -> None:
    """Migrate tickets_index table."""
    stats = ctx.stats[ctx.table_name]
    
    sql = """
        INSERT INTO tickets_index (
            ticket_id, status, subject, requester_id,
            created_at, updated_at, is_solved, indexed_at
        )
        VALUES (
            :ticket_id, :status, :subject, :requester_id,
            :created_at, :updated_at, :is_solved, :indexed_at
        )
        ON CONFLICT (ticket_id) DO UPDATE SET
            status = EXCLUDED.status,
            subject = EXCLUDED.subject,
            requester_id = EXCLUDED.requester_id,
            created_at = EXCLUDED.created_at,
            updated_at = EXCLUDED.updated_at,
            is_solved = EXCLUDED.is_solved,
            indexed_at = EXCLUDED.indexed_at
    """
    
    for row in ctx.rows:
        # Convert sqlite3.Row to dict for safe access
        row_dict = dict(row)
        ticket_id = str(row_dict["ticket_id"])
        
        # Skip test tickets
        if _is_test_ticket_id(ticket_id):
            stats["skipped_test"] += 1
            continue
        
        params = {
            "ticket_id": ticket_id,
            "status": row_dict.get("status"),
            "subject": row_dict.get("subject"),
            "requester_id": row_dict.get("requester_id"),
            "created_at": parse_timestamp(row_dict.get("created_at")),
            "updated_at": parse_timestamp(row_dict.get("updated_at")),
            "is_solved": bool(row_dict.get("is_solved", False)),
            "indexed_at": parse_timestamp(row_dict.get("indexed_at"))
        }
        
        if _execute_with_savepoint(ctx, sql, params, ticket_id):
            stats["inserted"] += 1
            ctx.parent_ids.add(ticket_id)
        else:
            stats["errors"] += 1


def _migrate_tickets_detail(ctx: MigrationContext) -> None:
    """Migrate tickets_detail table."""
    stats = ctx.stats[ctx.table_name]
    
    sql = """
        INSERT INTO tickets_detail (ticket_id, conversation_json, built_at)
        VALUES (:ticket_id, CAST(:conversation_json AS jsonb), :built_at)
        ON CONFLICT (ticket_id) DO UPDATE SET
            conversation_json = EXCLUDED.conversation_json,
            built_at = EXCLUDED.built_at
    """
    
    for row in ctx.rows:
        # Convert sqlite3.Row to dict for safe access
        row_dict = dict(row)
        ticket_id = str(row_dict["ticket_id"])
        
        # Orphan gate: check BEFORE touching Postgres
        should_skip, action = _check_orphan(ctx, ticket_id)
        if should_skip:
            if action == "test":
                stats["skipped_test"] += 1
            elif action == "orphan":
                stats["skipped_orphan"] += 1
            continue
        
        # Parse JSON string from SQLite
        conversation_json = row_dict.get("conversation_json")
        if conversation_json and isinstance(conversation_json, str):
            try:
                conversation_json = json.loads(conversation_json)
            except json.JSONDecodeError:
                conversation_json = None
        
        params = {
            "ticket_id": ticket_id,
            "conversation_json": to_jsonb(conversation_json, default={}),
            "built_at": parse_timestamp(row_dict.get("built_at"))
        }
        
        if _execute_with_savepoint(ctx, sql, params, ticket_id):
            stats["inserted"] += 1
        else:
            stats["errors"] += 1


def _migrate_ticket_summaries(ctx: MigrationContext) -> None:
    """Migrate ticket_summaries table."""
    stats = ctx.stats[ctx.table_name]
    
    sql = """
        INSERT INTO ticket_summaries (
            ticket_id, subject, status, problem_text, solution_text,
            key_quotes, resolution_confirmed, message_count, attachments_count,
            onsite_required, resolution_mode, resolution_mode_confidence, onsite_signals,
            embedding_text, created_at, updated_at, built_at
        )
        VALUES (
            :ticket_id, :subject, :status, :problem_text, :solution_text,
            :key_quotes, :resolution_confirmed, :message_count, :attachments_count,
            :onsite_required, :resolution_mode, :resolution_mode_confidence, :onsite_signals,
            :embedding_text, :created_at, :updated_at, :built_at
        )
        ON CONFLICT (ticket_id) DO UPDATE SET
            subject = EXCLUDED.subject,
            status = EXCLUDED.status,
            problem_text = EXCLUDED.problem_text,
            solution_text = EXCLUDED.solution_text,
            key_quotes = EXCLUDED.key_quotes,
            resolution_confirmed = EXCLUDED.resolution_confirmed,
            message_count = EXCLUDED.message_count,
            attachments_count = EXCLUDED.attachments_count,
            onsite_required = EXCLUDED.onsite_required,
            resolution_mode = EXCLUDED.resolution_mode,
            resolution_mode_confidence = EXCLUDED.resolution_mode_confidence,
            onsite_signals = EXCLUDED.onsite_signals,
            embedding_text = EXCLUDED.embedding_text,
            created_at = EXCLUDED.created_at,
            updated_at = EXCLUDED.updated_at,
            built_at = EXCLUDED.built_at
    """
    
    for row in ctx.rows:
        row_dict = dict(row)
        ticket_id = str(row_dict["ticket_id"])
        
        # Orphan gate
        should_skip, action = _check_orphan(ctx, ticket_id)
        if should_skip:
            if action == "test":
                stats["skipped_test"] += 1
            elif action == "orphan":
                stats["skipped_orphan"] += 1
            continue
        
        params = {
            "ticket_id": ticket_id,
            "subject": row_dict.get("subject"),
            "status": row_dict.get("status"),
            "problem_text": row_dict.get("problem_text"),
            "solution_text": row_dict.get("solution_text"),
            "key_quotes": row_dict.get("key_quotes"),
            "resolution_confirmed": bool(row_dict.get("resolution_confirmed", 0)),
            "message_count": row_dict.get("message_count"),
            "attachments_count": row_dict.get("attachments_count"),
            "onsite_required": bool(row_dict.get("onsite_required", 0)),
            "resolution_mode": row_dict.get("resolution_mode", "unknown"),
            "resolution_mode_confidence": float(row_dict.get("resolution_mode_confidence", 0.0)),
            "onsite_signals": row_dict.get("onsite_signals"),
            "embedding_text": row_dict.get("embedding_text"),
            "created_at": parse_timestamp(row_dict.get("created_at")),
            "updated_at": parse_timestamp(row_dict.get("updated_at")),
            "built_at": parse_timestamp(row_dict.get("built_at"))
        }
        
        if _execute_with_savepoint(ctx, sql, params, ticket_id):
            stats["inserted"] += 1
        else:
            stats["errors"] += 1


def _migrate_ticket_judgements(ctx: MigrationContext) -> None:
    """Migrate ticket_judgements table."""
    stats = ctx.stats[ctx.table_name]
    
    # Parse JSON fields helper
    def parse_json_field(value):
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except:
                return value
        return value
    
    sql = """
        INSERT INTO ticket_judgements (
            ticket_id, cache_eligible, confidence, problem, resolution_steps_json,
            confirmation, evidence_json, blockers_json, model, prompt_version,
            judged_at, raw_response_json, review_status, review_reason, review_reasons_json,
            reviewed_at
        )
        VALUES (
            :ticket_id, :cache_eligible, :confidence, :problem, CAST(:resolution_steps_json AS jsonb),
            :confirmation, CAST(:evidence_json AS jsonb), CAST(:blockers_json AS jsonb), :model, :prompt_version,
            :judged_at, CAST(:raw_response_json AS jsonb), :review_status, :review_reason, CAST(:review_reasons_json AS jsonb),
            :reviewed_at
        )
        ON CONFLICT (ticket_id) DO UPDATE SET
            cache_eligible = EXCLUDED.cache_eligible,
            confidence = EXCLUDED.confidence,
            problem = EXCLUDED.problem,
            resolution_steps_json = EXCLUDED.resolution_steps_json,
            confirmation = EXCLUDED.confirmation,
            evidence_json = EXCLUDED.evidence_json,
            blockers_json = EXCLUDED.blockers_json,
            model = EXCLUDED.model,
            prompt_version = EXCLUDED.prompt_version,
            judged_at = EXCLUDED.judged_at,
            raw_response_json = EXCLUDED.raw_response_json,
            review_status = EXCLUDED.review_status,
            review_reason = EXCLUDED.review_reason,
            review_reasons_json = EXCLUDED.review_reasons_json,
            reviewed_at = EXCLUDED.reviewed_at
    """
    
    for row in ctx.rows:
        row_dict = dict(row)
        ticket_id = str(row_dict["ticket_id"])
        
        # Orphan gate
        should_skip, action = _check_orphan(ctx, ticket_id)
        if should_skip:
            if action == "test":
                stats["skipped_test"] += 1
            elif action == "orphan":
                stats["skipped_orphan"] += 1
            continue
        
        params = {
            "ticket_id": ticket_id,
            "cache_eligible": bool(row_dict.get("cache_eligible", False)),
            "confidence": float(row_dict.get("confidence", 0.0)),
            "problem": row_dict.get("problem"),
            "resolution_steps_json": to_jsonb(parse_json_field(row_dict.get("resolution_steps_json"))),
            "confirmation": row_dict.get("confirmation"),
            "evidence_json": to_jsonb(parse_json_field(row_dict.get("evidence_json"))),
            "blockers_json": to_jsonb(parse_json_field(row_dict.get("blockers_json"))),
            "model": row_dict.get("model"),
            "prompt_version": row_dict.get("prompt_version"),
            "judged_at": parse_timestamp(row_dict.get("judged_at")),
            "raw_response_json": to_jsonb(parse_json_field(row_dict.get("raw_response_json")), default={}),
            "review_status": row_dict.get("review_status"),
            "review_reason": row_dict.get("review_reason"),
            "review_reasons_json": to_jsonb(parse_json_field(row_dict.get("review_reasons_json"))),
            "reviewed_at": parse_timestamp(row_dict.get("reviewed_at"))
        }
        
        if _execute_with_savepoint(ctx, sql, params, ticket_id):
            stats["inserted"] += 1
        else:
            stats["errors"] += 1


def _migrate_ticket_triage(ctx: MigrationContext) -> None:
    """Migrate ticket_triage table."""
    stats = ctx.stats[ctx.table_name]
    
    sql = """
        INSERT INTO ticket_triage (
            ticket_id, triage_label, triage_confidence, triage_reason,
            triaged_at, triage_model, triage_prompt_version, triage_raw_response_json
        )
        VALUES (
            :ticket_id, :triage_label, :triage_confidence, :triage_reason,
            :triaged_at, :triage_model, :triage_prompt_version, CAST(:triage_raw_response_json AS jsonb)
        )
        ON CONFLICT (ticket_id) DO UPDATE SET
            triage_label = EXCLUDED.triage_label,
            triage_confidence = EXCLUDED.triage_confidence,
            triage_reason = EXCLUDED.triage_reason,
            triaged_at = EXCLUDED.triaged_at,
            triage_model = EXCLUDED.triage_model,
            triage_prompt_version = EXCLUDED.triage_prompt_version,
            triage_raw_response_json = EXCLUDED.triage_raw_response_json
    """
    
    for row in ctx.rows:
        row_dict = dict(row)
        ticket_id = str(row_dict["ticket_id"])
        
        # Orphan gate
        should_skip, action = _check_orphan(ctx, ticket_id)
        if should_skip:
            if action == "test":
                stats["skipped_test"] += 1
            elif action == "orphan":
                stats["skipped_orphan"] += 1
            continue
        
        triage_raw_json = row_dict.get("triage_raw_response_json")
        if triage_raw_json and isinstance(triage_raw_json, str):
            try:
                triage_raw_json = json.loads(triage_raw_json)
            except json.JSONDecodeError:
                triage_raw_json = None
        
        params = {
            "ticket_id": ticket_id,
            "triage_label": row_dict.get("triage_label"),
            "triage_confidence": float(row_dict.get("triage_confidence", 0.0)),
            "triage_reason": row_dict.get("triage_reason"),
            "triaged_at": parse_timestamp(row_dict.get("triaged_at")),
            "triage_model": row_dict.get("triage_model"),
            "triage_prompt_version": row_dict.get("triage_prompt_version"),
            "triage_raw_response_json": to_jsonb(triage_raw_json, default={})
        }
        
        if _execute_with_savepoint(ctx, sql, params, ticket_id):
            stats["inserted"] += 1
        else:
            stats["errors"] += 1


def _migrate_ticket_manual_reviews(ctx: MigrationContext) -> None:
    """Migrate ticket_manual_reviews table."""
    stats = ctx.stats[ctx.table_name]
    
    sql = """
        INSERT INTO ticket_manual_reviews (
            ticket_id, manual_status, manual_reason, manual_confirmation_quote,
            reviewer, reviewed_at
        )
        VALUES (
            :ticket_id, :manual_status, :manual_reason, :manual_confirmation_quote,
            :reviewer, :reviewed_at
        )
        ON CONFLICT (ticket_id) DO UPDATE SET
            manual_status = EXCLUDED.manual_status,
            manual_reason = EXCLUDED.manual_reason,
            manual_confirmation_quote = EXCLUDED.manual_confirmation_quote,
            reviewer = EXCLUDED.reviewer,
            reviewed_at = EXCLUDED.reviewed_at
    """
    
    for row in ctx.rows:
        row_dict = dict(row)
        ticket_id = str(row_dict["ticket_id"])
        
        # Orphan gate
        should_skip, action = _check_orphan(ctx, ticket_id)
        if should_skip:
            if action == "test":
                stats["skipped_test"] += 1
            elif action == "orphan":
                stats["skipped_orphan"] += 1
            continue
        
        params = {
            "ticket_id": ticket_id,
            "manual_status": row_dict.get("manual_status"),
            "manual_reason": row_dict.get("manual_reason"),
            "manual_confirmation_quote": row_dict.get("manual_confirmation_quote"),
            "reviewer": row_dict.get("reviewer"),
            "reviewed_at": parse_timestamp(row_dict.get("reviewed_at"))
        }
        
        if _execute_with_savepoint(ctx, sql, params, ticket_id):
            stats["inserted"] += 1
        else:
            stats["errors"] += 1


def _migrate_ticket_machine_model_matches(ctx: MigrationContext) -> None:
    """Migrate ticket_machine_model_matches table."""
    stats = ctx.stats[ctx.table_name]
    
    sql = """
        INSERT INTO ticket_machine_model_matches (
            ticket_id, machine_model_id, machine_model_name, match_source,
            score, evidence_snippet, created_at
        )
        VALUES (
            :ticket_id, :machine_model_id, :machine_model_name, :match_source,
            :score, :evidence_snippet, :created_at
        )
        ON CONFLICT (ticket_id, machine_model_id, match_source) DO UPDATE SET
            machine_model_name = EXCLUDED.machine_model_name,
            score = EXCLUDED.score,
            evidence_snippet = EXCLUDED.evidence_snippet,
            created_at = EXCLUDED.created_at
    """
    
    for row in ctx.rows:
        row_dict = dict(row)
        ticket_id = str(row_dict["ticket_id"])
        
        # Orphan gate
        should_skip, action = _check_orphan(ctx, ticket_id)
        if should_skip:
            if action == "test":
                stats["skipped_test"] += 1
            elif action == "orphan":
                stats["skipped_orphan"] += 1
            continue
        
        params = {
            "ticket_id": ticket_id,
            "machine_model_id": int(row_dict.get("machine_model_id", 0)),
            "machine_model_name": row_dict.get("machine_model_name"),
            "match_source": row_dict.get("match_source"),
            "score": int(row_dict.get("score", 0)),
            "evidence_snippet": row_dict.get("evidence_snippet"),
            "created_at": parse_timestamp(row_dict.get("created_at"))
        }
        
        if _execute_with_savepoint(ctx, sql, params, ticket_id):
            stats["inserted"] += 1
        else:
            stats["errors"] += 1


def _migrate_ticket_machine_model_assignment(ctx: MigrationContext) -> None:
    """Migrate ticket_machine_model_assignment table."""
    stats = ctx.stats[ctx.table_name]
    
    sql = """
        INSERT INTO ticket_machine_model_assignment (
            ticket_id, machine_model_ids, status, confidence, method, updated_at
        )
        VALUES (
            :ticket_id, CAST(:machine_model_ids AS jsonb), :status, :confidence, :method, :updated_at
        )
        ON CONFLICT (ticket_id) DO UPDATE SET
            machine_model_ids = EXCLUDED.machine_model_ids,
            status = EXCLUDED.status,
            confidence = EXCLUDED.confidence,
            method = EXCLUDED.method,
            updated_at = EXCLUDED.updated_at
    """
    
    for row in ctx.rows:
        row_dict = dict(row)
        ticket_id = str(row_dict["ticket_id"])
        
        # Orphan gate
        should_skip, action = _check_orphan(ctx, ticket_id)
        if should_skip:
            if action == "test":
                stats["skipped_test"] += 1
            elif action == "orphan":
                stats["skipped_orphan"] += 1
            continue
        
        # Parse machine_model_ids JSON string
        machine_model_ids = row_dict.get("machine_model_ids")
        if machine_model_ids is None:
            machine_model_ids = []
        elif isinstance(machine_model_ids, str):
            try:
                machine_model_ids = json.loads(machine_model_ids)
            except json.JSONDecodeError:
                machine_model_ids = []
        elif not isinstance(machine_model_ids, list):
            machine_model_ids = []
        
        # Convert to JSON string for Postgres JSONB
        machine_model_ids_json = json.dumps(machine_model_ids, ensure_ascii=False)
        
        params = {
            "ticket_id": ticket_id,
            "machine_model_ids": machine_model_ids_json,
            "status": row_dict.get("status"),
            "confidence": float(row_dict.get("confidence", 0.0)) if row_dict.get("confidence") is not None else 0.0,
            "method": row_dict.get("method"),
            "updated_at": parse_timestamp(row_dict.get("updated_at"))
        }
        
        if _execute_with_savepoint(ctx, sql, params, ticket_id):
            stats["inserted"] += 1
        else:
            stats["errors"] += 1


def _migrate_scrape_runs(ctx: MigrationContext) -> None:
    """Migrate scrape_runs table."""
    stats = ctx.stats[ctx.table_name]
    
    sql = """
        INSERT INTO scrape_runs (
            run_id, status, stage, started_at, completed_at,
            error, summary_json, created_by
        )
        VALUES (
            :run_id, :status, :stage, :started_at, :completed_at,
            :error, CAST(:summary_json AS jsonb), :created_by
        )
        ON CONFLICT (run_id) DO UPDATE SET
            status = EXCLUDED.status,
            stage = EXCLUDED.stage,
            started_at = EXCLUDED.started_at,
            completed_at = EXCLUDED.completed_at,
            error = EXCLUDED.error,
            summary_json = EXCLUDED.summary_json,
            created_by = EXCLUDED.created_by
    """
    
    for row in ctx.rows:
        row_dict = dict(row)
        run_id = str(row_dict.get("run_id", "unknown"))
        
        # Parse summary_json if present
        summary_json = row_dict.get("summary_json")
        if summary_json and isinstance(summary_json, str):
            try:
                summary_json = json.loads(summary_json)
            except json.JSONDecodeError:
                summary_json = None
        
        # Convert to JSON string for Postgres JSONB
        summary_json_str = json.dumps(summary_json, ensure_ascii=False) if summary_json else None
        
        params = {
            "run_id": run_id,
            "status": row_dict.get("status"),
            "stage": row_dict.get("stage"),
            "started_at": parse_timestamp(row_dict.get("started_at")),
            "completed_at": parse_timestamp(row_dict.get("completed_at")),
            "error": row_dict.get("error"),
            "summary_json": summary_json_str,
            "created_by": row_dict.get("created_by")
        }
        
        if _execute_with_savepoint(ctx, sql, params, run_id):
            stats["inserted"] += 1
        else:
            stats["errors"] += 1


def main():
    # Create out/ directory if it doesn't exist (for log files, etc.)
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    
    parser = argparse.ArgumentParser(
        description="Migrate tickets from SQLite to Postgres",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--sqlite-path",
        default="Scraper/data/tickets.db",
        help="Path to SQLite tickets database (default: Scraper/data/tickets.db)"
    )
    
    parser.add_argument(
        "--database-url",
        help="Postgres connection string (default: DATABASE_URL env var)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run mode: rollback all changes"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of rows per table (for testing)"
    )
    
    parser.add_argument(
        "--tables",
        help="Comma-separated list of tables to migrate (default: all)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing (default: 1000)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (output will be written to both console and log file)"
    )
    
    parser.add_argument(
        "--orphan-policy",
        choices=["skip", "backfill", "error"],
        default="skip",
        help="How to handle orphan rows (ticket_id not in tickets_index): skip (default), backfill (create parent), or error"
    )
    
    parser.add_argument(
        "--max-error-lines",
        type=int,
        default=25,
        help="Maximum error lines to print per table (default: 25)"
    )
    
    parser.add_argument(
        "--debug-sql",
        action="store_true",
        help="Print full SQL and parameters for errors (default: False, compact logging)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed skip messages for orphan/test tickets"
    )
    
    args = parser.parse_args()
    
    # Set up logging to file if specified
    log_file_path = None
    log_file_handle = None
    if args.log_file:
        log_file_path = Path(args.log_file)
        # Ensure directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Open log file for writing
        log_file_handle = open(log_file_path, 'w', encoding='utf-8', buffering=1)
        
        # Create a class that writes to both stdout/stderr and log file
        class Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, text):
                for f in self.files:
                    f.write(text)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()
        
        # Redirect stdout and stderr to both console and log file
        sys.stdout = Tee(sys.stdout, log_file_handle)
        sys.stderr = Tee(sys.stderr, log_file_handle)
        
        # Write header to log file
        print(f"Migration log started at {datetime.now().isoformat()}", file=sys.stderr)
        print(f"Log file: {log_file_path}", file=sys.stderr)
    
    # Validate SQLite path
    sqlite_path = Path(args.sqlite_path)
    if not sqlite_path.exists():
        print(f"ERROR: SQLite database not found: {sqlite_path}")
        sys.exit(1)
    
    # Get Postgres connection
    database_url = args.database_url or os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable or --database-url required")
        sys.exit(1)
    
    # Connect to SQLite
    sqlite_conn = sqlite3.connect(str(sqlite_path))
    sqlite_conn.row_factory = sqlite3.Row
    
    # Connect to Postgres
    pg_engine = create_engine(database_url, poolclass=NullPool, future=True)
    
    # Determine tables to migrate
    if args.tables:
        tables_to_migrate = [t.strip() for t in args.tables.split(",")]
    else:
        # Default: all ticket tables
        tables_to_migrate = [
            "tickets_index",
            "tickets_detail",
            "ticket_summaries",
            "ticket_judgements",
            "ticket_triage",
            "ticket_manual_reviews",
            "ticket_machine_model_matches",
            "ticket_machine_model_assignment",
            "scrape_runs"
        ]
    
    print("=" * 70)
    print("TICKET MIGRATION: SQLite -> Postgres")
    print("=" * 70)
    print(f"SQLite DB: {sqlite_path}")
    print(f"Postgres: {database_url.split('@')[1] if '@' in database_url else '***'}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")
    print(f"Orphan policy: {args.orphan_policy}")
    print(f"Tables: {', '.join(tables_to_migrate)}")
    if args.limit:
        print(f"Limit: {args.limit} rows per table")
    print("=" * 70)
    
    if args.dry_run:
        print("\n[DRY-RUN MODE] All changes will be rolled back")
    
    # Preflight: Get required parent IDs from SQLite
    print("\n[PREFLIGHT] Computing required parent ticket_ids...")
    required_parent_ids = _get_required_parent_ids(sqlite_conn)
    print(f"  Found {len(required_parent_ids)} unique ticket_ids across all tables")
    
    # Create migration context with Session
    SessionLocal = sessionmaker(bind=pg_engine)
    session = SessionLocal()
    ctx = MigrationContext(
        sqlite_conn=sqlite_conn,
        session=session,
        dry_run=args.dry_run,
        orphan_policy=args.orphan_policy,
        parent_ids=set(),
        stats={},
        args=args
    )
    
    # Wrap entire migration in transaction for dry-run
    outer_txn = session.begin()
    results = {}
    
    try:
        # Migrate tickets_index first
        if "tickets_index" in tables_to_migrate:
            result = migrate_table(ctx, "tickets_index", limit=args.limit)
            results["tickets_index"] = result
            print(f"  [INFO] Tracking {len(ctx.parent_ids)} parent ticket_ids for FK validation")
        
        # Backfill missing parents if policy is backfill
        missing_ids = required_parent_ids - ctx.parent_ids
        if missing_ids and ctx.orphan_policy == "backfill":
            print(f"\n[BACKFILL] Inserting {len(missing_ids)} missing parent ticket_ids...")
            backfilled = _backfill_missing_parents(ctx, missing_ids)
            print(f"  Backfilled {backfilled} parent ticket_ids")
        
        # Migrate remaining tables
        for table_name in tables_to_migrate:
            if table_name == "tickets_index":
                continue  # Already migrated
            
            try:
                result = migrate_table(ctx, table_name, limit=args.limit)
                results[table_name] = result
            except Exception as e:
                print(f"\nERROR migrating {table_name}: {e}")
                import traceback
                if args.debug_sql:
                    traceback.print_exc()
                results[table_name] = {
                    "total": 0, "inserted": 0, "updated": 0, "errors": 1,
                    "skipped_orphan": 0, "skipped_test": 0
                }
        
        # Handle dry-run rollback
        if ctx.dry_run:
            outer_txn.rollback()
            print("\n[DRY-RUN] All changes rolled back")
        else:
            outer_txn.commit()
            print("\n[LIVE] Changes committed")
    except Exception as e:
        outer_txn.rollback()
        print(f"\n[ERROR] Migration failed, rolling back: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        session.close()
        pg_engine.dispose()
    
    # Summary
    print("\n" + "=" * 70)
    print("MIGRATION SUMMARY")
    print("=" * 70)
    total_rows = sum(r.get("total", 0) for r in results.values())
    total_inserted = sum(r.get("inserted", 0) for r in results.values())
    total_errors = sum(r.get("errors", 0) for r in results.values())
    total_skipped_orphan = sum(r.get("skipped_orphan", 0) for r in results.values())
    total_skipped_test = sum(r.get("skipped_test", 0) for r in results.values())
    
    print(f"Total rows processed: {total_rows}")
    print(f"Total rows migrated: {total_inserted}")
    print(f"Total errors: {total_errors}")
    if total_skipped_orphan > 0:
        print(f"Total skipped (orphan): {total_skipped_orphan}")
    if total_skipped_test > 0:
        print(f"Total skipped (test): {total_skipped_test}")
    
    print("\nPer-table breakdown:")
    for table_name, result in results.items():
        parts = [f"{result.get('inserted', 0)} migrated", f"{result.get('errors', 0)} errors"]
        if result.get("skipped_orphan", 0) > 0:
            parts.append(f"{result['skipped_orphan']} skipped_orphan")
        if result.get("skipped_test", 0) > 0:
            parts.append(f"{result['skipped_test']} skipped_test")
        print(f"  {table_name}: {', '.join(parts)}")
    
    if args.dry_run:
        print("\n[DRY-RUN] No changes were committed to Postgres")
    else:
        print("\nMigration completed successfully!")
    
    # Exit code: non-zero if hard errors OR if orphan_policy=error and orphans found
    exit_code = 0
    if total_errors > 0:
        exit_code = 1
    elif args.orphan_policy == "error" and total_skipped_orphan > 0:
        exit_code = 1
        print(f"\n[ERROR] Found {total_skipped_orphan} orphan rows with --orphan-policy=error")
    
    sqlite_conn.close()
    pg_engine.dispose()
    
    # Close log file if opened
    if log_file_handle:
        log_file_handle.close()
        # Restore original stdout/stderr for final message
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        print(f"\nLog file written to: {log_file_path}", file=sys.stderr)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
