#!/usr/bin/env python3
"""
Postgres schema upgrade helper (no Alembic required).

Run:
    python -m backend.scripts.db_upgrade

Behavior:
- Reads DATABASE_URL
- Executes the SQL migration file (idempotent)
- Runs verification queries and prints results

This is intended for production operators who want a single command without psql.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _default_sql_path() -> Path:
    return (_repo_root() / "backend" / "migrations" / "20251218_add_document_machine_models.sql").resolve()


def _redact_database_url(database_url: str) -> str:
    # Basic redaction (works for postgresql://user:pass@host:port/db?...)
    try:
        parsed = urlparse(database_url)
        netloc = parsed.netloc
        if "@" in netloc:
            creds, hostpart = netloc.split("@", 1)
            if ":" in creds:
                user, _pw = creds.split(":", 1)
                netloc = f"{user}:***@{hostpart}"
            else:
                netloc = f"{creds}@{hostpart}"
        safe = parsed._replace(netloc=netloc).geturl()
        return safe
    except Exception:
        return re.sub(r":([^:@/]+)@", r":***@", database_url)


def _fail(msg: str) -> "NoReturn":
    print(f"[DB_UPGRADE] ❌ {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"[DB_UPGRADE] ✅ {msg}", flush=True)


def main() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        _fail("DATABASE_URL is not set")

    sql_path = Path(os.getenv("DB_UPGRADE_SQL", str(_default_sql_path()))).resolve()
    if not sql_path.exists():
        _fail(f"Migration SQL file not found: {sql_path}")

    print("[DB_UPGRADE] Starting DB upgrade", flush=True)
    print(f"[DB_UPGRADE] DATABASE_URL={_redact_database_url(database_url)}", flush=True)
    print(f"[DB_UPGRADE] SQL={sql_path}", flush=True)

    try:
        import psycopg2  # type: ignore
    except Exception as e:
        _fail(f"psycopg2 is required but not installed/available: {type(e).__name__}: {e}")

    sql_text = sql_path.read_text(encoding="utf-8")
    if not sql_text.strip():
        _fail(f"Migration SQL file is empty: {sql_path}")

    # Execute migration (idempotent)
    try:
        conn = psycopg2.connect(database_url)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql_text)
        _ok("Migration SQL executed")
    except Exception as e:
        _fail(f"Migration failed: {type(e).__name__}: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Verification queries (explicit)
    verify_sql: list[tuple[str, str]] = [
        ("table_exists", "SELECT to_regclass('public.document_machine_models');"),
        ("row_count", "SELECT count(*) FROM public.document_machine_models;"),
        ("distinct_documents", "SELECT count(DISTINCT document_id) FROM public.document_machine_models;"),
        ("distinct_machine_models", "SELECT count(DISTINCT machine_model_id) FROM public.document_machine_models;"),
        (
            "constraint",
            """
            SELECT conname, pg_get_constraintdef(oid)
            FROM pg_constraint
            WHERE conrelid = 'public.document_machine_models'::regclass
              AND conname = 'uq_document_machine_models'
            """,
        ),
        (
            "sample_join",
            """
            SELECT d.id, d.file_name, array_agg(mm.name ORDER BY mm.name) AS machine_models
            FROM public.documents d
            JOIN public.document_machine_models dmm ON dmm.document_id = d.id
            JOIN public.machine_models mm ON mm.id = dmm.machine_model_id
            GROUP BY d.id, d.file_name
            ORDER BY d.id DESC
            LIMIT 5
            """,
        ),
    ]

    try:
        conn2 = psycopg2.connect(database_url)
        conn2.autocommit = True
        with conn2.cursor() as cur:
            print("\n[DB_UPGRADE] Verification", flush=True)
            for name, q in verify_sql:
                cur.execute(q)
                rows = cur.fetchall()
                print(f"- {name}: {rows}", flush=True)
        _ok("Verification completed")
    except Exception as e:
        _fail(f"Verification failed: {type(e).__name__}: {e}")
    finally:
        try:
            conn2.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()


