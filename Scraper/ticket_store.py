"""
Ticket storage abstraction layer.

Provides a unified interface for ticket persistence that can work with
either SQLite (local dev) or Postgres (production).

Usage:
    store = get_ticket_store()
    store.upsert_ticket_index(row)
    store.set_ticket_detail(ticket_id, conversation)
    # etc.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


class TicketStore(ABC):
    """
    Abstract base class for ticket storage backends.
    
    All methods mirror the signatures from Scraper/db.py functions
    to enable drop-in replacement.
    """
    
    @abstractmethod
    def upsert_ticket_index(self, row: Dict[str, Any]) -> None:
        """Insert or update a ticket in tickets_index."""
        pass
    
    @abstractmethod
    def set_ticket_detail(self, ticket_id: str, conversation: Dict[str, Any]) -> None:
        """Insert or update a ticket's detailed conversation JSON."""
        pass
    
    @abstractmethod
    def upsert_ticket_summary(self, summary_dict: Dict[str, Any], rebuild: bool = False) -> None:
        """Insert or update a ticket summary."""
        pass
    
    @abstractmethod
    def upsert_ticket_judgement(self, judgement_dict: Dict[str, Any]) -> None:
        """Insert or update a ticket judgement."""
        pass
    
    @abstractmethod
    def upsert_ticket_triage(self, triage_dict: Dict[str, Any]) -> None:
        """Insert or update a ticket triage result."""
        pass
    
    @abstractmethod
    def upsert_manual_review(
        self,
        ticket_id: str,
        manual_status: str,
        manual_reason: Optional[str] = None,
        manual_confirmation_quote: Optional[str] = None,
        reviewer: Optional[str] = None
    ) -> None:
        """Insert or update a manual review override."""
        pass
    
    @abstractmethod
    def get_ticket_detail_json(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        """Get ticket detail conversation JSON."""
        pass
    
    @abstractmethod
    def get_ticket_index(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        """Get ticket index record."""
        pass
    
    @abstractmethod
    def get_solved_ticket_ids(self, statuses: tuple = ("solved", "closed")) -> List[str]:
        """Get list of solved ticket IDs."""
        pass
    
    @abstractmethod
    def get_ticket_ids_without_detail(self) -> List[str]:
        """Get list of solved ticket IDs that don't have detail records yet."""
        pass
    
    @abstractmethod
    def get_ticket_ids_needing_judgement(self, only_solved: bool = True, limit: Optional[int] = None) -> List[str]:
        """Get ticket IDs that need judgement."""
        pass
    
    @abstractmethod
    def count_index(self) -> int:
        """Count total tickets in tickets_index."""
        pass
    
    @abstractmethod
    def count_detail(self) -> int:
        """Count total tickets in tickets_detail."""
        pass
    
    @abstractmethod
    def count_solved(self) -> int:
        """Count solved tickets in tickets_index."""
        pass
    
    @abstractmethod
    def count_open(self) -> int:
        """Count open tickets in tickets_index."""
        pass
    
    @abstractmethod
    def count_judged(self) -> int:
        """Count tickets that have been judged."""
        pass
    
    @abstractmethod
    def count_cache_eligible(self) -> int:
        """Count tickets that are cache eligible."""
        pass
    
    # Scrape runs methods
    @abstractmethod
    def ensure_scrape_runs_table(self) -> None:
        """Ensure scrape_runs table exists."""
        pass
    
    @abstractmethod
    def get_active_scrape_run(self, max_age_hours: int = 2) -> Optional[Dict[str, Any]]:
        """Get the active scrape run."""
        pass
    
    @abstractmethod
    def create_scrape_run(self, run_id: str, created_by: Optional[str] = None) -> None:
        """Create a new scrape run record."""
        pass
    
    @abstractmethod
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
        """Update a scrape run record."""
        pass
    
    @abstractmethod
    def get_latest_scrape_run(self) -> Optional[Dict[str, Any]]:
        """Get the latest scrape run."""
        pass


def get_ticket_store(backend: Optional[str] = None, **kwargs) -> TicketStore:
    """
    Factory function to get the appropriate ticket store.
    
    Args:
        backend: 'sqlite' or 'postgres'. If None, reads from TICKETS_STORAGE_BACKEND env var.
        **kwargs: Additional arguments passed to store constructor:
            - For sqlite: db_path (optional)
            - For postgres: database_url (optional, reads from DATABASE_URL env var)
    
    Returns:
        TicketStore instance
    """
    if backend is None:
        backend = os.getenv("TICKETS_STORAGE_BACKEND", "sqlite").lower()
    
    if backend == "postgres":
        from .ticket_store_postgres import PostgresTicketStore
        return PostgresTicketStore(**kwargs)
    else:
        from .ticket_store_sqlite import SQLiteTicketStore
        return SQLiteTicketStore(**kwargs)
