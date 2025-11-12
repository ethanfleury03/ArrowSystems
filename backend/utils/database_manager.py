from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select, func, desc
from sqlalchemy.exc import SQLAlchemyError

import os

import bcrypt

from .db import SessionLocal, User, QueryHistory, Feedback, SavedResponse, init_db
from .db import run_sync

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Async-friendly wrapper around SQLAlchemy sessions."""

    def __init__(self):
        init_db()

    def _ensure_user_sync(self, session, email: Optional[str]) -> int:
        email_normalized = (email or "api_user").strip().lower()
        user = session.execute(select(User).where(User.email == email_normalized)).scalar_one_or_none()
        if user:
            return user.id
        user = User(
            email=email_normalized,
            name=email_normalized,
            role="TECHNICIAN",
            password_hash="",
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user.id

    async def ensure_user(self, email: Optional[str]) -> int:
        def _ensure():
            with SessionLocal() as session:
                return self._ensure_user_sync(session, email)

        return await run_sync(_ensure)

    @staticmethod
    def _serialize_user(user: User) -> Dict[str, Any]:
        if not user:
            return {}
        return {
            "id": str(user.id),
            "email": user.email,
            "name": user.name,
            "role": user.role,
            "company_name": user.company_name,
            "contact_name": user.contact_name,
            "contact_phone": user.contact_phone,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "updated_at": user.updated_at.isoformat() if user.updated_at else None,
        }

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        def _get():
            with SessionLocal() as session:
                record = (
                    session.execute(
                        select(User).where(func.lower(User.email) == email.strip().lower())
                    ).scalars().first()
                )
                return self._serialize_user(record) if record else None

        return await run_sync(_get)

    async def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        def _get():
            with SessionLocal() as session:
                record = session.execute(select(User).where(User.id == user_id)).scalars().first()
                return self._serialize_user(record) if record else None

        return await run_sync(_get)

    async def create_user(
        self,
        email: str,
        password: str,
        role: str = "technician",
        name: Optional[str] = None,
        company_name: Optional[str] = None,
        contact_name: Optional[str] = None,
        contact_phone: Optional[str] = None,
    ) -> Dict[str, Any]:
        def _create():
            with SessionLocal() as session:
                normalized = email.strip().lower()
                existing = (
                    session.execute(select(User).where(User.email == normalized)).scalars().first()
                )
                if existing:
                    return self._serialize_user(existing)

                hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                user = User(
                    email=normalized,
                    name=name or normalized,
                    role=(role or "technician").upper(),
                    password_hash=hashed,
                    company_name=company_name,
                    contact_name=contact_name,
                    contact_phone=contact_phone,
                )
                session.add(user)
                session.commit()
                session.refresh(user)
                return self._serialize_user(user)

        return await run_sync(_create)

    async def list_users(self) -> List[Dict[str, Any]]:
        def _list() -> List[Dict[str, Any]]:
            with SessionLocal() as session:
                records = session.execute(select(User).order_by(User.created_at.asc())).scalars().all()
                return [self._serialize_user(record) for record in records]

        return await run_sync(_list)

    async def update_user(
        self,
        user_id: int,
        *,
        email: Optional[str] = None,
        name: Optional[str] = None,
        password: Optional[str] = None,
        role: Optional[str] = None,
        company_name: Optional[str] = None,
        contact_name: Optional[str] = None,
        contact_phone: Optional[str] = None,
    ) -> Dict[str, Any]:
        def _update() -> Dict[str, Any]:
            with SessionLocal() as session:
                user = session.get(User, user_id)
                if not user:
                    raise ValueError("User not found")

                if email:
                    normalized = email.strip().lower()
                    existing = (
                        session.execute(
                            select(User).where(func.lower(User.email) == normalized, User.id != user_id)
                        ).scalars().first()
                    )
                    if existing:
                        raise ValueError("Email already in use")
                    user.email = normalized

                if name is not None:
                    user.name = name

                if role:
                    user.role = role.strip().upper()

                if password:
                    user.password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

                if company_name is not None:
                    user.company_name = company_name

                if contact_name is not None:
                    user.contact_name = contact_name

                if contact_phone is not None:
                    user.contact_phone = contact_phone

                session.commit()
                session.refresh(user)
                return self._serialize_user(user)

        return await run_sync(_update)

    async def delete_user(self, user_id: int) -> bool:
        def _delete() -> bool:
            with SessionLocal() as session:
                user = session.get(User, user_id)
                if not user:
                    return False
                session.delete(user)
                session.commit()
                return True

        return await run_sync(_delete)

    async def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        def _auth():
            with SessionLocal() as session:
                record = (
                    session.execute(
                        select(User).where(func.lower(User.email) == email.strip().lower())
                    ).scalars().first()
                )
                if not record or not record.password_hash:
                    return None
                try:
                    if bcrypt.checkpw(password.encode("utf-8"), record.password_hash.encode("utf-8")):
                        return self._serialize_user(record)
                except ValueError:
                    logger.warning("User %s has invalid password hash", email)
                return None

        return await run_sync(_auth)

    async def seed_default_users(self) -> None:
        admin_email = os.getenv("SEED_ADMIN_EMAIL")
        admin_password = os.getenv("SEED_ADMIN_PASSWORD")
        admin_name = os.getenv("SEED_ADMIN_NAME", "Administrator")

        tech_email = os.getenv("SEED_TECH_EMAIL")
        tech_password = os.getenv("SEED_TECH_PASSWORD")
        tech_name = os.getenv("SEED_TECH_NAME", "Technician")

        if admin_email and admin_password:
            await self.create_user(admin_email, admin_password, role="ADMIN", name=admin_name)

        if tech_email and tech_password:
            await self.create_user(tech_email, tech_password, role="TECHNICIAN", name=tech_name)

    async def save_query(
        self,
        user: str,
        query_text: str,
        answer_text: str,
        intent_type: Optional[str] = None,
        intent_confidence: Optional[float] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        confidence: Optional[float] = None,
        response_time_ms: Optional[int] = None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        def _save() -> Optional[str]:
            try:
                with SessionLocal() as session:
                    user_id = self._ensure_user_sync(session, user)
                    metadata = {
                        "sessionId": session_id,
                        "intentType": intent_type,
                        "intentConfidence": intent_confidence,
                        "confidence": confidence,
                        "sources": sources or [],
                        **{k: v for k, v in kwargs.items() if v is not None},
                    }

                    record = QueryHistory(
                        user_id=user_id,
                        query_text=query_text,
                        answer_text=answer_text,
                        response_time_ms=response_time_ms,
                        metadata_json=metadata,
                    )
                    session.add(record)
                    session.commit()
                    session.refresh(record)
                    return str(record.id)
            except SQLAlchemyError as exc:  # pragma: no cover
                logger.warning("Failed to save query: %s", exc)
                return None

        return await run_sync(_save)

    async def get_query_history(self, user: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        def _fetch() -> List[Dict[str, Any]]:
            with SessionLocal() as session:
                query = select(QueryHistory).order_by(desc(QueryHistory.created_at)).limit(limit)
                if user:
                    query = query.join(QueryHistory.user).where(func.lower(User.email) == user.strip().lower())

                records = session.execute(query).scalars().all()
                history: List[Dict[str, Any]] = []
                for record in records:
                    metadata = record.metadata_json or {}
                    history.append(
                        {
                            "id": str(record.id),
                            "query": record.query_text,
                            "answer": record.answer_text or "",
                            "timestamp": record.created_at.isoformat() if record.created_at else "",
                            "intent_type": metadata.get("intentType"),
                            "confidence": metadata.get("confidence"),
                            "sources": metadata.get("sources", []),
                            "response_time_ms": metadata.get("response_time_ms") or record.response_time_ms,
                        }
                    )
                return history

        return await run_sync(_fetch)

    async def save_feedback(
        self,
        query_id: str,
        user: str,
        is_helpful: bool,
        confidence: Optional[float] = None,
        intent_type: Optional[str] = None,
    ) -> bool:
        def _save() -> bool:
            with SessionLocal() as session:
                query = (
                    session.execute(
                        select(QueryHistory)
                        .where(QueryHistory.query_text == query_id)
                        .order_by(desc(QueryHistory.created_at))
                    )
                    .scalars()
                    .first()
                )
                if not query:
                    logger.warning("Query text '%s' not found for feedback", query_id)
                    return False

                user_id = self._ensure_user_sync(session, user)
                feedback = Feedback(
                    user_id=user_id,
                    query_history_id=query.id,
                    is_helpful=is_helpful,
                    confidence=confidence,
                    intent_type=intent_type,
                )
                session.add(feedback)
                session.commit()
                return True

        return await run_sync(_save)

    async def list_saved_responses(self, user: Optional[str] = None) -> List[Dict[str, Any]]:
        def _list() -> List[Dict[str, Any]]:
            with SessionLocal() as session:
                query = select(SavedResponse).order_by(desc(SavedResponse.updated_at))
                if user:
                    query = query.join(SavedResponse.user).where(func.lower(User.email) == user.strip().lower())
                records = session.execute(query).scalars().all()
                return [
                    {
                        "id": record.id,
                        "query": record.query_text,
                        "answer": record.answer_text,
                        "sources": record.sources or [],
                        "created_at": record.created_at.isoformat() if record.created_at else "",
                        "last_used": record.updated_at.isoformat() if record.updated_at else "",
                        "helpful_count": 1,
                        "unhelpful_count": 0,
                    }
                    for record in records
                ]

        return await run_sync(_list)

    async def upsert_saved_response(
        self,
        query_text: str,
        answer_text: str,
        user: str,
        sources: Optional[List[str]] = None,
    ) -> bool:
        def _upsert() -> bool:
            with SessionLocal() as session:
                user_id = self._ensure_user_sync(session, user)
                existing = (
                    session.execute(
                        select(SavedResponse).where(
                            SavedResponse.user_id == user_id, SavedResponse.query_text == query_text.strip()
                        )
                    )
                    .scalars()
                    .first()
                )
                if existing:
                    existing.answer_text = answer_text
                    existing.sources = sources or []
                    session.commit()
                    return True

                saved = SavedResponse(
                    user_id=user_id,
                    query_text=query_text.strip(),
                    answer_text=answer_text,
                    sources=sources or [],
                )
                session.add(saved)
                session.commit()
                return True

        return await run_sync(_upsert)

    async def remove_saved_response(self, query: str, user: str) -> bool:
        def _remove() -> bool:
            with SessionLocal() as session:
                user_id = self._ensure_user_sync(session, user)
                record = (
                    session.execute(
                        select(SavedResponse).where(
                            SavedResponse.user_id == user_id, SavedResponse.query_text == query.strip()
                        )
                    )
                    .scalars()
                    .first()
                )
                if not record:
                    return False
                session.delete(record)
                session.commit()
                return True

        return await run_sync(_remove)

    async def is_saved(self, query: str, user: str) -> bool:
        def _check() -> bool:
            with SessionLocal() as session:
                user_id = self._ensure_user_sync(session, user)
                record = (
                    session.execute(
                        select(SavedResponse).where(
                            SavedResponse.user_id == user_id, SavedResponse.query_text == query.strip()
                        )
                    )
                    .scalars()
                    .first()
                )
                return record is not None

        return await run_sync(_check)

