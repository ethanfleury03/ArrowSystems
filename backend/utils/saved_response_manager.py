from __future__ import annotations

from typing import Any, Dict, List, Optional

from .database_manager import DatabaseManager


class SavedResponseManager:
    """Wrapper to manage saved responses using the DatabaseManager."""

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    async def save_response(
        self,
        query: str,
        answer: str,
        user: str,
        sources: Optional[List[str]] = None,
    ) -> bool:
        return await self._db.upsert_saved_response(
            query_text=query,
            answer_text=answer,
            user=user,
            sources=sources or [],
        )

    async def remove_response(self, query: str, user: str) -> bool:
        return await self._db.remove_saved_response(query=query, user=user)

    async def list_responses(self, user: Optional[str] = None) -> List[Dict[str, Any]]:
        return await self._db.list_saved_responses(user=user)

    async def is_saved(self, query: str, user: str) -> bool:
        return await self._db.is_saved(query=query, user=user)
