from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar

from sqlalchemy import select, func, desc
from sqlalchemy.exc import SQLAlchemyError, OperationalError

import os

import bcrypt

from .db import SessionLocal, User, QueryHistory, Feedback, SavedResponse, init_db, DATABASE_URL
from .db import run_sync
from ..logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def _retry_on_locked(func: Callable[..., T], max_retries: int = 5, *args: Any, **kwargs: Any) -> T:
    """
    Execute a database operation.
    
    PostgreSQL handles concurrency natively, so no retry logic is needed.
    This function is kept for backward compatibility.
    """
    return func(*args, **kwargs)


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
        _retry_on_locked(session.commit)
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
        
        # Normalize machine_models using the helper function
        from ..config.machine_models import normalize_machine_models
        machine_models = normalize_machine_models(user.machine_models)
        
        return {
            "id": str(user.id),
            "email": user.email,
            "name": user.name,
            "role": user.role or "TECHNICIAN",  # Ensure role is always present
            "company_name": user.company_name,
            "contact_name": user.contact_name,
            "contact_phone": user.contact_phone,
            "machine_models": machine_models,  # Always a normalized list[str]
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
        password: Optional[str] = None,
        role: str = "technician",
        name: Optional[str] = None,
        company_name: Optional[str] = None,
        contact_name: Optional[str] = None,
        contact_phone: Optional[str] = None,
        machine_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        def _create():
            import secrets
            from ..config.machine_models import normalize_machine_models
            
            with SessionLocal() as session:
                normalized = email.strip().lower()
                existing = (
                    session.execute(select(User).where(User.email == normalized)).scalars().first()
                )
                if existing:
                    return self._serialize_user(existing)

                # Handle password: if provided, use it; otherwise generate random secret
                if password and password.strip():
                    # Password provided - hash and use it
                    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                else:
                    # No password provided - generate random secret and hash it
                    # This ensures password_hash is not empty and cannot be guessed
                    random_secret = secrets.token_urlsafe(32)
                    hashed = bcrypt.hashpw(random_secret.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                    # Do not return or store the random password anywhere
                
                # Normalize machine_models using the helper
                machine_models_list = normalize_machine_models(machine_models)
                user = User(
                    email=normalized,
                    name=name or normalized,
                    role=(role or "technician").upper(),
                    password_hash=hashed,
                    company_name=company_name,
                    contact_name=contact_name,
                    contact_phone=contact_phone,
                    machine_models=machine_models_list,
                )
                session.add(user)
                _retry_on_locked(session.commit)
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
        machine_models: Optional[List[str]] = None,
        machine_model_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        def _update() -> Dict[str, Any]:
            # Use context manager to ensure proper session lifecycle
            # ONE session for the entire operation - no ORM objects cross session boundaries
            with SessionLocal() as session:
                try:
                    # Load user in THIS session - critical for session.refresh() to work
                    # DO NOT use any User instance from outside this function
                    user = session.get(User, user_id)
                    
                    # Log user lookup result (no PII)
                    if user is None:
                        logger.warning(
                            "update_user_user_not_found",
                            user_id=user_id,
                            message=f"User {user_id} not found in database"
                        )
                        raise ValueError("User not found")
                    
                    logger.info(
                        "update_user_start",
                        user_id=user_id,
                        has_email=email is not None,
                        has_name=name is not None,
                        has_password=password is not None,
                        has_role=role is not None,
                        has_machine_model_ids=machine_model_ids is not None,
                        machine_model_ids_count=len(machine_model_ids) if machine_model_ids else 0,
                        has_machine_models=machine_models is not None,
                    )

                    if email:
                        normalized = email.strip().lower()
                        if not normalized:
                            raise ValueError("Email cannot be empty")
                        existing = (
                            session.execute(
                                select(User).where(func.lower(User.email) == normalized, User.id != user_id)
                            ).scalars().first()
                        )
                        if existing:
                            raise ValueError("Email already in use")
                        user.email = normalized

                    if name is not None:
                        if not name.strip():
                            raise ValueError("Name cannot be empty")
                        user.name = name.strip()

                    if role:
                        role_upper = role.strip().upper()
                        if role_upper not in ["ADMIN", "TECHNICIAN", "CUSTOMER"]:
                            raise ValueError(f"Invalid role: {role}. Must be ADMIN, TECHNICIAN, or CUSTOMER")
                        user.role = role_upper

                    if password:
                        if not password.strip():
                            raise ValueError("Password cannot be empty")
                        user.password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

                    if company_name is not None:
                        user.company_name = company_name.strip() if company_name else None

                    if contact_name is not None:
                        user.contact_name = contact_name.strip() if contact_name else None

                    if contact_phone is not None:
                        user.contact_phone = contact_phone.strip() if contact_phone else None

                    # Handle machine models - support both IDs and names
                    # Initialize at top to avoid UnboundLocalError
                    updated_machine_models = None
                    
                    if machine_model_ids is not None:
                        # Convert IDs to names via DB lookup in the same session
                        from ..utils.db import MachineModel
                        # Convert to list of ints and deduplicate
                        ids = [int(x) for x in machine_model_ids]
                        unique_ids = sorted(set(ids))
                        
                        logger.info(
                            "update_user_machine_model_ids",
                            user_id=user_id,
                            machine_model_ids_count=len(unique_ids),
                            machine_model_ids=unique_ids,
                        )
                        
                        if len(unique_ids) == 0:
                            # Empty list means clear all machine models
                            updated_machine_models = []
                        else:
                            # Load machine models in THIS session
                            models = session.execute(
                                select(MachineModel).where(MachineModel.id.in_(unique_ids))
                            ).scalars().all()
                            found_ids = {m.id for m in models}
                            missing_ids = sorted(set(unique_ids) - found_ids)
                            if missing_ids:
                                logger.warning(
                                    "update_user_invalid_machine_model_ids",
                                    user_id=user_id,
                                    missing_ids=missing_ids,
                                    requested_ids=unique_ids,
                                )
                                raise ValueError(f"Invalid machine model IDs: {missing_ids}")
                            # Store as names (JSON column)
                            from ..config.machine_models import normalize_machine_models
                            updated_machine_models = normalize_machine_models([m.name for m in models])
                    elif machine_models is not None:
                        # Normalize machine_models using the helper (names provided directly)
                        from ..config.machine_models import normalize_machine_models
                        # Validate input type
                        if not isinstance(machine_models, list):
                            raise ValueError(f"machine_models must be a list, got {type(machine_models).__name__}")
                        # Normalize and validate
                        updated_machine_models = normalize_machine_models(machine_models)
                    
                    # Only update user.machine_models if we have a value to set
                    if updated_machine_models is not None:
                        user.machine_models = updated_machine_models
                    # If both machine_model_ids and machine_models are None, don't touch user.machine_models

                    # Commit transaction with retry on lock
                    _retry_on_locked(session.commit)
                    
                    # Refresh to ensure we have latest state (user must be in this session)
                    session.refresh(user)
                    
                    # Access machine_models (JSON column) inside session to ensure it's loaded
                    # This prevents any potential lazy-loading issues after session closes
                    _ = user.machine_models  # Access the attribute while session is open
                    
                    # Serialize BEFORE session closes (user is still attached)
                    # Return a dict, not the ORM instance
                    result = self._serialize_user(user)
                    
                    logger.info(
                        "update_user_success",
                        user_id=user_id,
                        updated_machine_models_count=len(result.get("machine_models", [])),
                    )
                    
                    return result
                except ValueError:
                    # Re-raise validation errors
                    session.rollback()
                    raise
                except SQLAlchemyError as e:
                    # Database errors - rollback and re-raise
                    session.rollback()
                    logger.error(
                        "update_user_database_error",
                        user_id=user_id,
                        error=str(e),
                        error_type=type(e).__name__,
                        exc_info=True,
                    )
                    raise ValueError(f"Database error: {str(e)}")
                except Exception as e:
                    # Unexpected errors - rollback and re-raise
                    session.rollback()
                    logger.error(
                        "update_user_unexpected_error",
                        user_id=user_id,
                        error=str(e),
                        error_type=type(e).__name__,
                        exc_info=True,
                    )
                    raise ValueError(f"Failed to update user: {str(e)}")

        return await run_sync(_update)
    
    async def get_user_machine_models(self, user_id: int) -> List[str]:
        """
        Get machine models for a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of machine model strings (e.g., ["330R", "DuraFlex"])
        """
        def _get() -> List[str]:
            from ..config.machine_models import normalize_machine_models
            
            with SessionLocal() as session:
                user = session.get(User, user_id)
                if not user:
                    return []
                
                # Normalize using helper function
                return normalize_machine_models(user.machine_models)
        
        return await run_sync(_get)

    async def delete_user(self, user_id: int) -> bool:
        def _delete() -> bool:
            with SessionLocal() as session:
                user = session.get(User, user_id)
                if not user:
                    return False
                session.delete(user)
                _retry_on_locked(session.commit)
                return True

        return await run_sync(_delete)

    async def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        def _auth():
            try:
                with SessionLocal() as session:
                    record = (
                        session.execute(
                            select(User).where(func.lower(User.email) == email.strip().lower())
                        ).scalars().first()
                    )
                    if not record:
                        logger.debug(f"User not found: {email}")
                        return None
                    if not record.password_hash or record.password_hash.strip() == "":
                        logger.warning(f"User {email} has no password hash set")
                        return None
                    try:
                        if bcrypt.checkpw(password.encode("utf-8"), record.password_hash.encode("utf-8")):
                            return self._serialize_user(record)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"User {email} has invalid password hash: {e}")
                    except Exception as e:
                        logger.error(f"Error checking password for {email}: {e}", exc_info=True)
                    return None
            except Exception as e:
                logger.error(f"Error authenticating user {email}: {e}", exc_info=True)
                raise

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
        conversation_id: Optional[str] = None,
        machine_name: Optional[str] = None,
        token_input: Optional[int] = None,
        token_output: Optional[int] = None,
        token_total: Optional[int] = None,
        cost_usd: Optional[float] = None,
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
                    
                    # Serialize sources to JSON string for analytics
                    import json
                    sources_json_str = None
                    if sources:
                        try:
                            sources_json_str = json.dumps(sources)
                        except Exception:
                            pass

                    # Extract language fields from kwargs if present
                    detected_language = kwargs.get('detected_language')
                    language_confidence = kwargs.get('language_confidence')
                    query_retrieval = kwargs.get('query_retrieval')
                    translation_provider = kwargs.get('translation_provider')
                    
                    record = QueryHistory(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        query_text=query_text,
                        answer_text=answer_text,
                        response_time_ms=response_time_ms,
                        metadata_json=metadata,
                        machine_name=machine_name,
                        token_input=token_input,
                        token_output=token_output,
                        token_total=token_total,
                        cost_usd=cost_usd,
                        sources_json=sources_json_str,
                        # Language metadata
                        detected_language=detected_language,
                        language_confidence=language_confidence,
                        query_retrieval=query_retrieval,
                        translation_provider=translation_provider,
                    )
                    session.add(record)
                    _retry_on_locked(session.commit)
                    session.refresh(record)
                    return str(record.id)
            except SQLAlchemyError as exc:  # pragma: no cover
                logger.warning("Failed to save query: %s", exc)
                return None

        return await run_sync(_save)

    async def get_query_history(self, user: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        def _fetch() -> List[Dict[str, Any]]:
            with SessionLocal() as session:
                # Explicitly select only columns that exist to avoid updated_at issues
                # Use load_only to avoid loading updated_at if it doesn't exist in DB
                from sqlalchemy.orm import load_only
                query = select(QueryHistory).options(
                    load_only(
                        QueryHistory.id,
                        QueryHistory.user_id,
                        QueryHistory.query_text,
                        QueryHistory.answer_text,
                        QueryHistory.response_time_ms,
                        QueryHistory.metadata_json,
                        QueryHistory.created_at,
                        QueryHistory.machine_name,
                        QueryHistory.token_input,
                        QueryHistory.token_output,
                        QueryHistory.token_total,
                        QueryHistory.cost_usd,
                        QueryHistory.sources_json,
                    )
                ).order_by(desc(QueryHistory.created_at)).limit(limit)
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
                _retry_on_locked(session.commit)
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
                    _retry_on_locked(session.commit)
                    return True

                saved = SavedResponse(
                    user_id=user_id,
                    query_text=query_text.strip(),
                    answer_text=answer_text,
                    sources=sources or [],
                )
                session.add(saved)
                _retry_on_locked(session.commit)
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
                _retry_on_locked(session.commit)
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

