"""
Saved Response Manager backed by Prisma (PostgreSQL via Neon)
"""

from typing import Any, Dict, List, Optional

from prisma import Prisma
from prisma.enums import UserRole


class SavedResponseManager:
    """Manages saved (bookmarked) responses using Prisma."""

    def __init__(self, client: Prisma):
        self.client = client

    async def _ensure_user(self, email: str) -> str:
        if not email:
            email = "api_user"
        user = await self.client.user.upsert(
            where={"email": email},
            update={},
            create={
                "email": email,
                "name": email,
                "role": UserRole.TECHNICIAN,
                "passwordHash": "",
            },
        )
        return user.id

    async def save_response(
        self,
        query: str,
        answer: str,
        user: str,
        sources: Optional[List[str]] = None,
    ) -> bool:
        user_id = await self._ensure_user(user)
        await self.client.savedresponse.upsert(
            where={
                "userId_query": {
                    "userId": user_id,
                    "query": query.strip(),
                }
            },
            update={
                "answer": answer,
                "sources": sources or [],
            },
            create={
                "userId": user_id,
                "query": query.strip(),
                "answer": answer,
                "sources": sources or [],
            },
        )
        return True

    async def remove_response(self, query: str, user: str) -> bool:
        user_id = await self._ensure_user(user)
        result = await self.client.savedresponse.delete_many(
            where={
                "userId": user_id,
                "query": query.strip(),
            }
        )
        return result.count > 0

    async def list_responses(self, user: Optional[str] = None) -> List[Dict[str, Any]]:
        where_clause = {}
        if user:
            user_record = await self.client.user.find_unique(where={"email": user})
            if not user_record:
                return []
            where_clause["userId"] = user_record.id

        records = await self.client.savedresponse.find_many(
            where=where_clause or None,
            order={"updatedAt": "desc"},
        )

        return [
            {
                "id": saved.id,
                "query": saved.query,
                "answer": saved.answer,
                "sources": saved.sources or [],
                "created_at": saved.createdAt.isoformat(),
                "last_used": saved.updatedAt.isoformat(),
                "helpful_count": 1,
                "unhelpful_count": 0,
            }
            for saved in records
        ]

    async def is_saved(self, query: str, user: str) -> bool:
        user_record = await self.client.user.find_unique(where={"email": user})
        if not user_record:
            return False
        record = await self.client.savedresponse.find_first(
            where={
                "userId": user_record.id,
                "query": query.strip(),
            }
        )
        return record is not None