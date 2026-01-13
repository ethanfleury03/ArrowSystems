"""Add ticket scraper tables (migrated from SQLite)

This migration creates all ticket-related tables in Postgres to mirror
the SQLite schema in Scraper/db.py, enabling migration from local SQLite
to Cloud SQL Postgres.

Tables created:
- tickets_index (Stage 1: cheap indexing)
- tickets_detail (Stage 2: detailed conversations)
- ticket_summaries (Stage 3: structured extraction)
- ticket_judgements (LLM cache eligibility)
- ticket_triage (cheap model triage)
- ticket_manual_reviews (manual overrides)
- ticket_machine_model_matches (machine model matches)
- ticket_machine_model_assignment (machine model assignments)
- scrape_runs (background scrape job tracking)

Revision ID: 011_ticket_tables_postgres
Revises: 010_document_machine_models_m2m
Create Date: 2025-01-14 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import inspect, text


# revision identifiers, used by Alembic.
revision = "011_ticket_tables_postgres"
down_revision = "010_document_machine_models_m2m"
branch_labels = None
depends_on = None


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    # Create tickets_index table
    if not _table_exists("tickets_index"):
        op.create_table(
            "tickets_index",
            sa.Column("ticket_id", sa.String(255), primary_key=True),
            sa.Column("status", sa.String(50), nullable=True),
            sa.Column("subject", sa.Text(), nullable=True),
            sa.Column("requester_id", sa.String(255), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("is_solved", sa.Boolean(), nullable=False, server_default="false"),
            sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("idx_tickets_index_is_solved", "tickets_index", ["is_solved"])
        op.create_index("idx_tickets_index_status", "tickets_index", ["status"])

    # Create tickets_detail table
    if not _table_exists("tickets_detail"):
        op.create_table(
            "tickets_detail",
            sa.Column("ticket_id", sa.String(255), sa.ForeignKey("tickets_index.ticket_id", ondelete="CASCADE"), primary_key=True),
            sa.Column("conversation_json", postgresql.JSONB(), nullable=False),
            sa.Column("built_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )

    # Create ticket_summaries table
    if not _table_exists("ticket_summaries"):
        op.create_table(
            "ticket_summaries",
            sa.Column("ticket_id", sa.String(255), sa.ForeignKey("tickets_detail.ticket_id", ondelete="CASCADE"), primary_key=True),
            sa.Column("subject", sa.Text(), nullable=True),
            sa.Column("status", sa.String(50), nullable=True),
            sa.Column("problem_text", sa.Text(), nullable=True),
            sa.Column("solution_text", sa.Text(), nullable=True),
            sa.Column("key_quotes", sa.Text(), nullable=True),
            sa.Column("resolution_confirmed", sa.Boolean(), nullable=False, server_default="false"),
            sa.Column("message_count", sa.Integer(), nullable=True),
            sa.Column("attachments_count", sa.Integer(), nullable=True),
            sa.Column("onsite_required", sa.Boolean(), nullable=False, server_default="false"),
            sa.Column("resolution_mode", sa.String(50), nullable=False, server_default="unknown"),
            sa.Column("resolution_mode_confidence", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("onsite_signals", sa.Text(), nullable=True),
            sa.Column("embedding_text", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("built_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("idx_ticket_summaries_resolution_confirmed", "ticket_summaries", ["resolution_confirmed"])
        op.create_index("idx_ticket_summaries_status", "ticket_summaries", ["status"])

    # Create ticket_judgements table
    if not _table_exists("ticket_judgements"):
        op.create_table(
            "ticket_judgements",
            sa.Column("ticket_id", sa.String(255), sa.ForeignKey("tickets_detail.ticket_id", ondelete="CASCADE"), primary_key=True),
            sa.Column("cache_eligible", sa.Boolean(), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=False),
            sa.Column("problem", sa.Text(), nullable=True),
            sa.Column("resolution_steps_json", postgresql.JSONB(), nullable=True),
            sa.Column("confirmation", sa.Text(), nullable=True),
            sa.Column("evidence_json", postgresql.JSONB(), nullable=True),
            sa.Column("blockers_json", postgresql.JSONB(), nullable=True),
            sa.Column("model", sa.String(255), nullable=False),
            sa.Column("prompt_version", sa.String(255), nullable=False),
            sa.Column("judged_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("raw_response_json", postgresql.JSONB(), nullable=False),
            sa.Column("review_status", sa.String(50), nullable=True),
            sa.Column("review_reason", sa.Text(), nullable=True),
            sa.Column("review_reasons_json", postgresql.JSONB(), nullable=True),
            sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("idx_ticket_judgements_cache_eligible", "ticket_judgements", ["cache_eligible"])
        op.create_index("idx_ticket_judgements_review_status", "ticket_judgements", ["review_status"])

    # Create ticket_triage table
    if not _table_exists("ticket_triage"):
        op.create_table(
            "ticket_triage",
            sa.Column("ticket_id", sa.String(255), sa.ForeignKey("tickets_detail.ticket_id", ondelete="CASCADE"), primary_key=True),
            sa.Column("triage_label", sa.String(50), nullable=False),
            sa.Column("triage_confidence", sa.Float(), nullable=False),
            sa.Column("triage_reason", sa.Text(), nullable=True),
            sa.Column("triaged_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("triage_model", sa.String(255), nullable=False),
            sa.Column("triage_prompt_version", sa.String(255), nullable=False),
            sa.Column("triage_raw_response_json", postgresql.JSONB(), nullable=False),
        )
        op.create_index("idx_ticket_triage_label", "ticket_triage", ["triage_label"])

    # Create ticket_manual_reviews table
    if not _table_exists("ticket_manual_reviews"):
        op.create_table(
            "ticket_manual_reviews",
            sa.Column("ticket_id", sa.String(255), sa.ForeignKey("ticket_judgements.ticket_id", ondelete="CASCADE"), primary_key=True),
            sa.Column("manual_status", sa.String(50), nullable=False),
            sa.Column("manual_reason", sa.Text(), nullable=True),
            sa.Column("manual_confirmation_quote", sa.Text(), nullable=True),
            sa.Column("reviewer", sa.String(255), nullable=True),
            sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.CheckConstraint("manual_status IN ('approved', 'rejected')", name="check_manual_status"),
        )
        op.create_index("idx_ticket_manual_reviews_status", "ticket_manual_reviews", ["manual_status"])

    # Create ticket_machine_model_matches table
    if not _table_exists("ticket_machine_model_matches"):
        op.create_table(
            "ticket_machine_model_matches",
            sa.Column("ticket_id", sa.String(255), nullable=False),
            sa.Column("machine_model_id", sa.Integer(), nullable=False),
            sa.Column("machine_model_name", sa.String(255), nullable=False),
            sa.Column("match_source", sa.String(50), nullable=False),
            sa.Column("score", sa.Integer(), nullable=False),
            sa.Column("evidence_snippet", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.PrimaryKeyConstraint("ticket_id", "machine_model_id", "match_source"),
        )
        op.create_index("idx_ticket_machine_model_matches_ticket_id", "ticket_machine_model_matches", ["ticket_id"])

    # Create ticket_machine_model_assignment table
    if not _table_exists("ticket_machine_model_assignment"):
        op.create_table(
            "ticket_machine_model_assignment",
            sa.Column("ticket_id", sa.String(255), primary_key=True),
            sa.Column("machine_model_ids", postgresql.JSONB(), nullable=False),
            sa.Column("status", sa.String(50), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=False),
            sa.Column("method", sa.String(255), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("idx_ticket_machine_model_assignment_status", "ticket_machine_model_assignment", ["status"])

    # Create scrape_runs table
    if not _table_exists("scrape_runs"):
        op.create_table(
            "scrape_runs",
            sa.Column("run_id", sa.String(255), primary_key=True),
            sa.Column("status", sa.String(50), nullable=False),
            sa.Column("stage", sa.String(50), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("summary_json", postgresql.JSONB(), nullable=True),
            sa.Column("created_by", sa.String(255), nullable=True),
            sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'cancelled')", name="check_scrape_status"),
            sa.CheckConstraint("stage IN ('indexing', 'building_details', 'judging') OR stage IS NULL", name="check_scrape_stage"),
        )
        op.create_index("idx_scrape_runs_status", "scrape_runs", ["status"])
        op.create_index("idx_scrape_runs_started_at", "scrape_runs", ["started_at"])


def downgrade() -> None:
    # Drop tables in reverse dependency order
    tables_to_drop = [
        "scrape_runs",
        "ticket_machine_model_assignment",
        "ticket_machine_model_matches",
        "ticket_manual_reviews",
        "ticket_triage",
        "ticket_judgements",
        "ticket_summaries",
        "tickets_detail",
        "tickets_index",
    ]
    
    for table_name in tables_to_drop:
        if _table_exists(table_name):
            op.drop_table(table_name)
