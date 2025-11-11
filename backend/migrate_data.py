"""
One-time migration script to move data from a Postgres (Neon) database
into the new local SQLite database used by the FastAPI backend.

Usage:
    python -m backend.migrate_data

Environment variables (optional):
    NEON_DATABASE_URL       Full Postgres connection string.
    DATABASE_URL            Alternate env var for connection string.

    or individual components:
    POSTGRES_HOST
    POSTGRES_PORT
    POSTGRES_DB
    POSTGRES_USER
    POSTGRES_PASSWORD
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:  # pragma: no cover - optional dependency
    psycopg2 = None
    RealDictCursor = None

from .utils.db import init_db, SessionLocal, User, QueryHistory, Feedback, SavedResponse


def _connect_postgres():
    if psycopg2 is None:
        print("❌ psycopg2 is not installed. Install it with `pip install psycopg2-binary` to run this migration.")
        return None
    url = os.getenv("NEON_DATABASE_URL") or os.getenv("DATABASE_URL")
    if url:
        return psycopg2.connect(url)

    host = os.getenv("POSTGRES_HOST")
    if not host:
        return None

    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "postgres")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")

    return psycopg2.connect(
        host=host,
        port=port,
        dbname=db,
        user=user,
        password=password,
    )


def _normalize_timestamp(value: Optional[Any]) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    return None


def _get(row: Dict[str, Any], *keys: str, default: Any = None):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def migrate_users(cursor, session):
    cursor.execute("SELECT * FROM users")
    rows = cursor.fetchall()
    for row in rows:
        role_value = (_get(row, "role", default="technician") or "technician").upper()
        user = User(
            id=row["id"],
            email=_get(row, "email"),
            name=_get(row, "name"),
            role=role_value,
            password_hash=_get(row, "password_hash", "passwordhash", "passwordHash", default=""),
            created_at=_normalize_timestamp(_get(row, "created_at", "createdAt")),
            updated_at=_normalize_timestamp(_get(row, "updated_at", "updatedAt")),
        )
        session.merge(user)


def migrate_query_history(cursor, session):
    cursor.execute("SELECT * FROM query_history")
    rows = cursor.fetchall()
    for row in rows:
        record = QueryHistory(
            id=row["id"],
            user_id=row["user_id"],
            query_text=_get(row, "query_text", "queryText", default=""),
            answer_text=_get(row, "answer_text", "answerText"),
            response_time_ms=_get(row, "response_time_ms", "responseTimeMs"),
            metadata=_get(row, "metadata", default={}),
            created_at=_normalize_timestamp(_get(row, "created_at", "createdAt")),
        )
        session.merge(record)


def migrate_feedback(cursor, session):
    cursor.execute("SELECT * FROM feedback")
    rows = cursor.fetchall()
    for row in rows:
        feedback = Feedback(
            id=row["id"],
            user_id=row["user_id"],
            query_history_id=_get(row, "query_history_id", "queryHistoryId"),
            is_helpful=_get(row, "is_helpful", "isHelpful", default=False),
            confidence=_get(row, "confidence"),
            intent_type=_get(row, "intent_type", "intentType"),
            created_at=_normalize_timestamp(_get(row, "created_at", "createdAt")),
        )
        session.merge(feedback)


def migrate_saved_responses(cursor, session):
    cursor.execute("SELECT * FROM saved_responses")
    rows = cursor.fetchall()
    for row in rows:
        saved = SavedResponse(
            id=row["id"],
            user_id=row["user_id"],
            query_text=_get(row, "query_text", "queryText", default=""),
            answer_text=_get(row, "answer_text", "answerText", default=""),
            sources=_get(row, "sources", default=[]),
            created_at=_normalize_timestamp(_get(row, "created_at", "createdAt")),
            updated_at=_normalize_timestamp(_get(row, "updated_at", "updatedAt")),
        )
        session.merge(saved)


def main():
    connection = _connect_postgres()
    if not connection:
        print("❌ Could not determine Postgres connection details. Set NEON_DATABASE_URL or POSTGRES_* variables.")
        return

    print("🔄 Connecting to Postgres...")
    init_db()

    with connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            with SessionLocal() as session:
                print("📥 Migrating users...")
                migrate_users(cursor, session)
                session.commit()

                print("📥 Migrating query history...")
                migrate_query_history(cursor, session)
                session.commit()

                print("📥 Migrating feedback...")
                migrate_feedback(cursor, session)
                session.commit()

                print("📥 Migrating saved responses...")
                migrate_saved_responses(cursor, session)
                session.commit()

    print("✅ Migration complete! SQLite database updated.")


if __name__ == "__main__":
    main()

