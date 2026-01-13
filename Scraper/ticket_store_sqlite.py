"""
SQLite implementation of TicketStore.

Wraps existing Scraper/db.py functions to provide TicketStore interface.
"""

import sqlite3
from typing import Any, Dict, List, Optional

from . import db
from .ticket_store import TicketStore


class SQLiteTicketStore(TicketStore):
    """
    SQLite-backed ticket store.
    
    Wraps existing Scraper/db.py functions for backward compatibility.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQLite store.
        
        Args:
            db_path: Optional path to SQLite database. Defaults to db.DEFAULT_DB_PATH.
        """
        self.db_path = db_path or db.DEFAULT_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create SQLite connection."""
        if self._conn is None:
            self._conn = db.get_connection(self.db_path)
        return self._conn
    
    def upsert_ticket_index(self, row: Dict[str, Any]) -> None:
        conn = self._get_connection()
        db.upsert_ticket_index(conn, row)
    
    def set_ticket_detail(self, ticket_id: str, conversation: Dict[str, Any]) -> None:
        conn = self._get_connection()
        db.set_ticket_detail(conn, ticket_id, conversation)
    
    def upsert_ticket_summary(self, summary_dict: Dict[str, Any], rebuild: bool = False) -> None:
        conn = self._get_connection()
        db.upsert_ticket_summary(conn, summary_dict, rebuild=rebuild)
    
    def upsert_ticket_judgement(self, judgement_dict: Dict[str, Any]) -> None:
        conn = self._get_connection()
        db.upsert_ticket_judgement(conn, judgement_dict)
    
    def upsert_ticket_triage(self, triage_dict: Dict[str, Any]) -> None:
        conn = self._get_connection()
        db.upsert_ticket_triage(conn, triage_dict)
    
    def upsert_manual_review(
        self,
        ticket_id: str,
        manual_status: str,
        manual_reason: Optional[str] = None,
        manual_confirmation_quote: Optional[str] = None,
        reviewer: Optional[str] = None
    ) -> None:
        conn = self._get_connection()
        db.upsert_manual_review(
            conn, ticket_id, manual_status,
            manual_reason=manual_reason,
            manual_confirmation_quote=manual_confirmation_quote,
            reviewer=reviewer
        )
    
    def get_ticket_detail_json(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        return db.get_ticket_detail_json(conn, ticket_id)
    
    def get_ticket_index(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        return db.get_ticket_index(conn, ticket_id)
    
    def get_solved_ticket_ids(self, statuses: tuple = ("solved", "closed")) -> List[str]:
        conn = self._get_connection()
        return db.get_solved_ticket_ids(conn, statuses=statuses)
    
    def get_ticket_ids_without_detail(self) -> List[str]:
        conn = self._get_connection()
        return db.get_ticket_ids_without_detail(conn)
    
    def get_ticket_ids_needing_judgement(self, only_solved: bool = True, limit: Optional[int] = None) -> List[str]:
        conn = self._get_connection()
        return db.get_ticket_ids_needing_judgement(conn, only_solved=only_solved, limit=limit)
    
    def count_index(self) -> int:
        conn = self._get_connection()
        return db.count_index(conn)
    
    def count_detail(self) -> int:
        conn = self._get_connection()
        return db.count_detail(conn)
    
    def count_solved(self) -> int:
        conn = self._get_connection()
        return db.count_solved(conn)
    
    def count_open(self) -> int:
        conn = self._get_connection()
        return db.count_open(conn)
    
    def count_judged(self) -> int:
        conn = self._get_connection()
        return db.count_judged(conn)
    
    def count_cache_eligible(self) -> int:
        conn = self._get_connection()
        return db.count_cache_eligible(conn)
    
    def ensure_scrape_runs_table(self) -> None:
        conn = self._get_connection()
        db.ensure_scrape_runs_table(conn)
    
    def get_active_scrape_run(self, max_age_hours: int = 2) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        return db.get_active_scrape_run(conn, max_age_hours=max_age_hours)
    
    def create_scrape_run(self, run_id: str, created_by: Optional[str] = None) -> None:
        conn = self._get_connection()
        db.create_scrape_run(conn, run_id, created_by=created_by)
    
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
        conn = self._get_connection()
        db.update_scrape_run(
            conn, run_id,
            status=status,
            stage=stage,
            error=error,
            summary_json=summary_json,
            completed_at=completed_at
        )
    
    def get_latest_scrape_run(self) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        return db.get_latest_scrape_run(conn)
    
    def close(self) -> None:
        """Close the connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
