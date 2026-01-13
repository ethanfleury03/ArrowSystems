"""
Postgres implementation of TicketStore.

Uses SQLAlchemy Core (not ORM) to avoid backend dependencies.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text, JSON as SQLJSON
from sqlalchemy.engine import Engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.pool import NullPool

from .ticket_store import TicketStore


class PostgresTicketStore(TicketStore):
    """
    Postgres-backed ticket store.
    
    Uses SQLAlchemy Core for direct SQL execution without ORM dependencies.
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize Postgres store.
        
        Args:
            database_url: Optional PostgreSQL connection string.
                         Defaults to DATABASE_URL environment variable.
        """
        if database_url is None:
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                raise ValueError(
                    "DATABASE_URL environment variable is required for PostgresTicketStore. "
                    "Set it or pass database_url parameter."
                )
        
        # Create engine with connection pooling
        self.engine = create_engine(
            database_url,
            poolclass=NullPool,  # Simple connection per operation
            future=True
        )
    
    def _parse_timestamp(self, ts: Optional[str]) -> Optional[datetime]:
        """Parse ISO timestamp string to datetime."""
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except Exception:
            return None
    
    def _format_timestamp(self, dt: Optional[datetime]) -> Optional[str]:
        """Format datetime to ISO string."""
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    
    def _json_to_jsonb(self, value: Any) -> Any:
        """Convert Python object to JSONB-compatible value."""
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value
    
    def upsert_ticket_index(self, row: Dict[str, Any]) -> None:
        indexed_at = datetime.now(timezone.utc)
        
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO tickets_index (
                        ticket_id, status, subject, requester_id,
                        created_at, updated_at, is_solved, indexed_at
                    )
                    VALUES (:ticket_id, :status, :subject, :requester_id,
                            :created_at, :updated_at, :is_solved, :indexed_at)
                    ON CONFLICT (ticket_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        subject = EXCLUDED.subject,
                        requester_id = EXCLUDED.requester_id,
                        created_at = EXCLUDED.created_at,
                        updated_at = EXCLUDED.updated_at,
                        is_solved = EXCLUDED.is_solved,
                        indexed_at = EXCLUDED.indexed_at
                """),
                {
                    "ticket_id": row["ticket_id"],
                    "status": row.get("status"),
                    "subject": row.get("subject"),
                    "requester_id": row.get("requester_id"),
                    "created_at": self._parse_timestamp(row.get("created_at")),
                    "updated_at": self._parse_timestamp(row.get("updated_at")),
                    "is_solved": bool(row.get("is_solved", 0)),
                    "indexed_at": indexed_at
                }
            )
    
    def set_ticket_detail(self, ticket_id: str, conversation: Dict[str, Any]) -> None:
        built_at = datetime.now(timezone.utc)
        conversation_json = json.dumps(conversation, ensure_ascii=False)
        
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO tickets_detail (ticket_id, conversation_json, built_at)
                    VALUES (:ticket_id, :conversation_json::jsonb, :built_at)
                    ON CONFLICT (ticket_id) DO UPDATE SET
                        conversation_json = EXCLUDED.conversation_json,
                        built_at = EXCLUDED.built_at
                """),
                {
                    "ticket_id": ticket_id,
                    "conversation_json": conversation_json,
                    "built_at": built_at
                }
            )
    
    def upsert_ticket_summary(self, summary_dict: Dict[str, Any], rebuild: bool = False) -> None:
        built_at = datetime.now(timezone.utc)
        ticket_id = summary_dict["ticket_id"]
        
        with self.engine.begin() as conn:
            if rebuild:
                conn.execute(
                    text("""
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
                    """),
                    {
                        "ticket_id": ticket_id,
                        "subject": summary_dict.get("subject"),
                        "status": summary_dict.get("status"),
                        "problem_text": summary_dict.get("problem_text"),
                        "solution_text": summary_dict.get("solution_text"),
                        "key_quotes": summary_dict.get("key_quotes"),
                        "resolution_confirmed": bool(summary_dict.get("resolution_confirmed", 0)),
                        "message_count": summary_dict.get("message_count"),
                        "attachments_count": summary_dict.get("attachments_count"),
                        "onsite_required": bool(summary_dict.get("onsite_required", 0)),
                        "resolution_mode": summary_dict.get("resolution_mode", "unknown"),
                        "resolution_mode_confidence": float(summary_dict.get("resolution_mode_confidence", 0.0)),
                        "onsite_signals": summary_dict.get("onsite_signals"),
                        "embedding_text": summary_dict.get("embedding_text"),
                        "created_at": self._parse_timestamp(summary_dict.get("created_at")),
                        "updated_at": self._parse_timestamp(summary_dict.get("updated_at")),
                        "built_at": built_at
                    }
                )
            else:
                conn.execute(
                    text("""
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
                        ON CONFLICT (ticket_id) DO NOTHING
                    """),
                    {
                        "ticket_id": ticket_id,
                        "subject": summary_dict.get("subject"),
                        "status": summary_dict.get("status"),
                        "problem_text": summary_dict.get("problem_text"),
                        "solution_text": summary_dict.get("solution_text"),
                        "key_quotes": summary_dict.get("key_quotes"),
                        "resolution_confirmed": bool(summary_dict.get("resolution_confirmed", 0)),
                        "message_count": summary_dict.get("message_count"),
                        "attachments_count": summary_dict.get("attachments_count"),
                        "onsite_required": bool(summary_dict.get("onsite_required", 0)),
                        "resolution_mode": summary_dict.get("resolution_mode", "unknown"),
                        "resolution_mode_confidence": float(summary_dict.get("resolution_mode_confidence", 0.0)),
                        "onsite_signals": summary_dict.get("onsite_signals"),
                        "embedding_text": summary_dict.get("embedding_text"),
                        "created_at": self._parse_timestamp(summary_dict.get("created_at")),
                        "updated_at": self._parse_timestamp(summary_dict.get("updated_at")),
                        "built_at": built_at
                    }
                )
    
    def upsert_ticket_judgement(self, judgement_dict: Dict[str, Any]) -> None:
        judged_at = self._parse_timestamp(judgement_dict.get("judged_at")) or datetime.now(timezone.utc)
        
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO ticket_judgements (
                        ticket_id, cache_eligible, confidence, problem, resolution_steps_json,
                        confirmation, evidence_json, blockers_json, model, prompt_version,
                        judged_at, raw_response_json, review_status, review_reason, review_reasons_json
                    )
                    VALUES (
                        :ticket_id, :cache_eligible, :confidence, :problem, :resolution_steps_json::jsonb,
                        :confirmation, :evidence_json::jsonb, :blockers_json::jsonb, :model, :prompt_version,
                        :judged_at, :raw_response_json::jsonb, :review_status, :review_reason, :review_reasons_json::jsonb
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
                        review_reasons_json = EXCLUDED.review_reasons_json
                """),
                {
                    "ticket_id": judgement_dict["ticket_id"],
                    "cache_eligible": bool(judgement_dict.get("cache_eligible", 0)),
                    "confidence": float(judgement_dict.get("confidence", 0.0)),
                    "problem": judgement_dict.get("problem"),
                    "resolution_steps_json": judgement_dict.get("resolution_steps_json"),
                    "confirmation": judgement_dict.get("confirmation"),
                    "evidence_json": judgement_dict.get("evidence_json"),
                    "blockers_json": judgement_dict.get("blockers_json"),
                    "model": judgement_dict.get("model", ""),
                    "prompt_version": judgement_dict.get("prompt_version", ""),
                    "judged_at": judged_at,
                    "raw_response_json": judgement_dict.get("raw_response_json", "{}"),
                    "review_status": judgement_dict.get("review_status"),
                    "review_reason": judgement_dict.get("review_reason"),
                    "review_reasons_json": judgement_dict.get("review_reasons_json")
                }
            )
    
    def upsert_ticket_triage(self, triage_dict: Dict[str, Any]) -> None:
        triaged_at = datetime.now(timezone.utc)
        ticket_id = triage_dict["ticket_id"]
        
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO ticket_triage (
                        ticket_id, triage_label, triage_confidence, triage_reason,
                        triaged_at, triage_model, triage_prompt_version, triage_raw_response_json
                    )
                    VALUES (
                        :ticket_id, :triage_label, :triage_confidence, :triage_reason,
                        :triaged_at, :triage_model, :triage_prompt_version, :triage_raw_response_json::jsonb
                    )
                    ON CONFLICT (ticket_id) DO UPDATE SET
                        triage_label = EXCLUDED.triage_label,
                        triage_confidence = EXCLUDED.triage_confidence,
                        triage_reason = EXCLUDED.triage_reason,
                        triaged_at = EXCLUDED.triaged_at,
                        triage_model = EXCLUDED.triage_model,
                        triage_prompt_version = EXCLUDED.triage_prompt_version,
                        triage_raw_response_json = EXCLUDED.triage_raw_response_json
                """),
                {
                    "ticket_id": ticket_id,
                    "triage_label": triage_dict["triage_label"],
                    "triage_confidence": float(triage_dict.get("triage_confidence", 0.0)),
                    "triage_reason": triage_dict.get("triage_reason"),
                    "triaged_at": triaged_at,
                    "triage_model": triage_dict["triage_model"],
                    "triage_prompt_version": triage_dict["triage_prompt_version"],
                    "triage_raw_response_json": triage_dict["triage_raw_response_json"]
                }
            )
    
    def upsert_manual_review(
        self,
        ticket_id: str,
        manual_status: str,
        manual_reason: Optional[str] = None,
        manual_confirmation_quote: Optional[str] = None,
        reviewer: Optional[str] = None
    ) -> None:
        reviewed_at = datetime.now(timezone.utc)
        
        with self.engine.begin() as conn:
            conn.execute(
                text("""
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
                """),
                {
                    "ticket_id": ticket_id,
                    "manual_status": manual_status,
                    "manual_reason": manual_reason,
                    "manual_confirmation_quote": manual_confirmation_quote,
                    "reviewer": reviewer,
                    "reviewed_at": reviewed_at
                }
            )
    
    def get_ticket_detail_json(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT conversation_json FROM tickets_detail WHERE ticket_id = :ticket_id"),
                {"ticket_id": ticket_id}
            )
            row = result.fetchone()
            if row is None:
                return None
            return row[0] if isinstance(row[0], dict) else json.loads(row[0])
    
    def get_ticket_index(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT ticket_id, status, subject, requester_id, created_at, updated_at, is_solved
                    FROM tickets_index
                    WHERE ticket_id = :ticket_id
                """),
                {"ticket_id": ticket_id}
            )
            row = result.fetchone()
            if row is None:
                return None
            return {
                "ticket_id": row[0],
                "status": row[1],
                "subject": row[2],
                "requester_id": row[3],
                "created_at": self._format_timestamp(row[4]),
                "updated_at": self._format_timestamp(row[5]),
                "is_solved": bool(row[6])
            }
    
    def get_solved_ticket_ids(self, statuses: tuple = ("solved", "closed")) -> List[str]:
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT ticket_id FROM tickets_index WHERE is_solved = true ORDER BY ticket_id")
            )
            return [row[0] for row in result.fetchall()]
    
    def get_ticket_ids_without_detail(self) -> List[str]:
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT i.ticket_id
                    FROM tickets_index i
                    LEFT JOIN tickets_detail d ON i.ticket_id = d.ticket_id
                    WHERE i.is_solved = true AND d.ticket_id IS NULL
                    ORDER BY i.ticket_id
                """)
            )
            return [row[0] for row in result.fetchall()]
    
    def get_ticket_ids_needing_judgement(self, only_solved: bool = True, limit: Optional[int] = None) -> List[str]:
        with self.engine.connect() as conn:
            if only_solved:
                query = text("""
                    SELECT d.ticket_id
                    FROM tickets_detail d
                    INNER JOIN tickets_index i ON d.ticket_id = i.ticket_id
                    LEFT JOIN ticket_judgements j ON d.ticket_id = j.ticket_id
                    WHERE i.is_solved = true AND j.ticket_id IS NULL
                    ORDER BY d.ticket_id
                """)
            else:
                query = text("""
                    SELECT d.ticket_id
                    FROM tickets_detail d
                    LEFT JOIN ticket_judgements j ON d.ticket_id = j.ticket_id
                    WHERE j.ticket_id IS NULL
                    ORDER BY d.ticket_id
                """)
            
            if limit:
                query = text(str(query) + f" LIMIT {limit}")
            
            result = conn.execute(query)
            return [row[0] for row in result.fetchall()]
    
    def count_index(self) -> int:
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM tickets_index"))
            return result.scalar() or 0
    
    def count_detail(self) -> int:
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM tickets_detail"))
            return result.scalar() or 0
    
    def count_solved(self) -> int:
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM tickets_index WHERE is_solved = true"))
            return result.scalar() or 0
    
    def count_open(self) -> int:
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM tickets_index WHERE is_solved = false"))
            return result.scalar() or 0
    
    def count_judged(self) -> int:
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM ticket_judgements"))
            return result.scalar() or 0
    
    def count_cache_eligible(self) -> int:
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM ticket_judgements WHERE cache_eligible = true"))
            return result.scalar() or 0
    
    def ensure_scrape_runs_table(self) -> None:
        # Table should already exist from migration, but this is idempotent
        pass
    
    def get_active_scrape_run(self, max_age_hours: int = 2) -> Optional[Dict[str, Any]]:
        from datetime import timedelta
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        
        with self.engine.begin() as conn:
            # Mark stale runs as failed
            conn.execute(
                text("""
                    UPDATE scrape_runs
                    SET status = 'failed',
                        error = 'Job marked as stale (exceeded maximum age)',
                        completed_at = :completed_at
                    WHERE status IN ('pending', 'running')
                    AND started_at < :cutoff_time
                """),
                {
                    "completed_at": datetime.now(timezone.utc),
                    "cutoff_time": cutoff_time
                }
            )
            
            # Get active run
            result = conn.execute(
                text("""
                    SELECT run_id, status, stage, started_at, completed_at, error, summary_json, created_by
                    FROM scrape_runs
                    WHERE status IN ('pending', 'running')
                    ORDER BY started_at DESC
                    LIMIT 1
                """)
            )
            row = result.fetchone()
            if row:
                return {
                    "run_id": row[0],
                    "status": row[1],
                    "stage": row[2],
                    "started_at": self._format_timestamp(row[3]),
                    "completed_at": self._format_timestamp(row[4]),
                    "error": row[5],
                    "summary_json": json.dumps(row[6]) if row[6] else None,
                    "created_by": row[7]
                }
            return None
    
    def create_scrape_run(self, run_id: str, created_by: Optional[str] = None) -> None:
        started_at = datetime.now(timezone.utc)
        
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO scrape_runs (run_id, status, started_at, created_by)
                    VALUES (:run_id, 'pending', :started_at, :created_by)
                """),
                {
                    "run_id": run_id,
                    "started_at": started_at,
                    "created_by": created_by
                }
            )
    
    def update_scrape_run(
        self,
        run_id: str,
        *,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        error: Optional[str] = None,
        summary_json: Optional[str] = None,
        completed_at: Optional[str] = None
    ) -> None:
        updates = []
        params = {"run_id": run_id}
        
        if status is not None:
            updates.append("status = :status")
            params["status"] = status
        
        if stage is not None:
            updates.append("stage = :stage")
            params["stage"] = stage
        
        if error is not None:
            updates.append("error = :error")
            params["error"] = error
        
        if summary_json is not None:
            updates.append("summary_json = :summary_json::jsonb")
            params["summary_json"] = summary_json
        
        if completed_at is not None:
            updates.append("completed_at = :completed_at")
            params["completed_at"] = self._parse_timestamp(completed_at)
        
        if not updates:
            return
        
        with self.engine.begin() as conn:
            conn.execute(
                text(f"UPDATE scrape_runs SET {', '.join(updates)} WHERE run_id = :run_id"),
                params
            )
    
    def get_latest_scrape_run(self) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT run_id, status, stage, started_at, completed_at, error, summary_json, created_by
                    FROM scrape_runs
                    ORDER BY started_at DESC
                    LIMIT 1
                """)
            )
            row = result.fetchone()
            if row:
                return {
                    "run_id": row[0],
                    "status": row[1],
                    "stage": row[2],
                    "started_at": self._format_timestamp(row[3]),
                    "completed_at": self._format_timestamp(row[4]),
                    "error": row[5],
                    "summary_json": json.dumps(row[6]) if row[6] else None,
                    "created_by": row[7]
                }
            return None
