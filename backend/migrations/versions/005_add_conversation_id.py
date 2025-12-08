"""Add conversation_id to query_history

This migration adds:
- conversation_id column to query_history table for grouping related queries

Revision ID: 005_add_conversation_id
Revises: 004_documents_and_glossary
Create Date: 2025-12-08 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


def _table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def _column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    if not _table_exists(table_name):
        return False
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


# revision identifiers, used by Alembic.
revision = '005_add_conversation_id'
down_revision = '004_documents_and_glossary'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add conversation_id column to query_history if it doesn't exist
    if _table_exists('query_history') and not _column_exists('query_history', 'conversation_id'):
        op.add_column(
            'query_history',
            sa.Column('conversation_id', sa.String(length=255), nullable=True)
        )
        # Create index on conversation_id for faster queries
        op.create_index(
            'ix_query_history_conversation_id',
            'query_history',
            ['conversation_id']
        )
        # Note: No logger import needed - migration runs in Alembic context


def downgrade() -> None:
    # Remove conversation_id column and index
    if _column_exists('query_history', 'conversation_id'):
        op.drop_index('ix_query_history_conversation_id', table_name='query_history')
        op.drop_column('query_history', 'conversation_id')

