"""Add document_machine_models join table (Document ↔ MachineModel many-to-many)

This migration:
- Creates document_machine_models join table
- Backfills mappings from legacy documents.machine_model (string or JSON array string)
- Best-effort fallback backfill from document_ingestion_metadata.machine_model when documents.machine_model is empty

Revision ID: 010_document_machine_models_m2m
Revises: 009_add_printer_machine_kind
Create Date: 2025-12-18 00:00:00.000000
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect, text


# revision identifiers, used by Alembic.
revision = "010_document_machine_models_m2m"
down_revision = "009_add_printer_machine_kind"
branch_labels = None
depends_on = None


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def _parse_legacy_machine_model(value: Any) -> list[str]:
    """
    Parse documents.machine_model legacy values:
    - None -> []
    - "DuraFlex" -> ["DuraFlex"]
    - '["DuraFlex","DuraCore"]' -> ["DuraFlex","DuraCore"]
    - "DuraFlex, DuraCore" -> ["DuraFlex","DuraCore"] (best-effort)
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if not isinstance(value, str):
        return [str(value).strip()] if str(value).strip() else []

    s = value.strip()
    if not s:
        return []

    # JSON array string
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            # fall through
            pass

    # CSV-ish fallback
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        return [p for p in parts if p]

    return [s]


def upgrade() -> None:
    bind = op.get_bind()

    if not _table_exists("documents") or not _table_exists("machine_models"):
        return

    # Create join table if not exists
    if not _table_exists("document_machine_models"):
        op.create_table(
            "document_machine_models",
            sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
            sa.Column("machine_model_id", sa.Integer(), sa.ForeignKey("machine_models.id", ondelete="CASCADE"), nullable=False),
            sa.UniqueConstraint("document_id", "machine_model_id", name="uq_document_machine_models"),
        )
        op.create_index("ix_document_machine_models_document_id", "document_machine_models", ["document_id"])
        op.create_index("ix_document_machine_models_machine_model_id", "document_machine_models", ["machine_model_id"])

    # Backfill mappings
    # Build name->id map
    mm_rows = bind.execute(text("SELECT id, name FROM machine_models")).fetchall()
    name_to_id = {str(r.name).strip(): int(r.id) for r in mm_rows if r.name is not None}

    # Helper: insert mapping (idempotent)
    def insert_mapping(document_id: int, machine_model_id: int) -> None:
        bind.execute(
            text(
                """
                INSERT INTO document_machine_models (document_id, machine_model_id)
                VALUES (:document_id, :machine_model_id)
                ON CONFLICT (document_id, machine_model_id) DO NOTHING
                """
            ),
            {"document_id": document_id, "machine_model_id": machine_model_id},
        )

    # Prefer documents.machine_model
    doc_rows = bind.execute(text("SELECT id, file_name, machine_model FROM documents")).fetchall()

    # Build metadata filename->machine_model fallback map (best-effort)
    meta_map: dict[str, str] = {}
    if _table_exists("document_ingestion_metadata"):
        meta_rows = bind.execute(
            text(
                """
                SELECT filename, machine_model
                FROM document_ingestion_metadata
                WHERE machine_model IS NOT NULL AND machine_model <> ''
                """
            )
        ).fetchall()
        for r in meta_rows:
            if r.filename and r.machine_model:
                # keep first seen
                meta_map.setdefault(str(r.filename), str(r.machine_model))

    for r in doc_rows:
        doc_id = int(r.id)
        file_name = str(r.file_name) if r.file_name is not None else ""
        legacy = r.machine_model
        models = _parse_legacy_machine_model(legacy)

        # fallback to metadata table if no models on document
        if not models and file_name in meta_map:
            models = _parse_legacy_machine_model(meta_map[file_name])

        for mname in models:
            if not mname:
                continue
            # if value is numeric, allow matching machine_models.id
            mm_id = None
            try:
                numeric = int(mname)
                if any(int(v) == numeric for v in name_to_id.values()):
                    mm_id = numeric
            except Exception:
                mm_id = None

            if mm_id is None:
                mm_id = name_to_id.get(mname)

            if mm_id is None:
                # Unknown name; skip (do not fail migration)
                continue

            insert_mapping(doc_id, int(mm_id))


def downgrade() -> None:
    if _table_exists("document_machine_models"):
        try:
            op.drop_index("ix_document_machine_models_document_id", table_name="document_machine_models")
        except Exception:
            pass
        try:
            op.drop_index("ix_document_machine_models_machine_model_id", table_name="document_machine_models")
        except Exception:
            pass
        op.drop_table("document_machine_models")


