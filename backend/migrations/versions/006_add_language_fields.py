"""Add language fields to query_history

This migration adds:
- detected_language column (ISO 639-1 language code)
- language_confidence column (0.0-1.0)
- query_retrieval column (English query used for retrieval)
- translation_provider column (translation service used)

Revision ID: 006_add_language_fields
Revises: 005_add_conversation_id
Create Date: 2025-12-09 00:00:00.000000

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
revision = '006_add_language_fields'
down_revision = '005_add_conversation_id'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add language metadata columns to query_history if they don't exist
    if _table_exists('query_history'):
        if not _column_exists('query_history', 'detected_language'):
            op.add_column(
                'query_history',
                sa.Column('detected_language', sa.String(length=10), nullable=True)
            )
        
        if not _column_exists('query_history', 'language_confidence'):
            op.add_column(
                'query_history',
                sa.Column('language_confidence', sa.Float(), nullable=True)
            )
        
        if not _column_exists('query_history', 'query_retrieval'):
            op.add_column(
                'query_history',
                sa.Column('query_retrieval', sa.Text(), nullable=True)
            )
        
        if not _column_exists('query_history', 'translation_provider'):
            op.add_column(
                'query_history',
                sa.Column('translation_provider', sa.String(length=50), nullable=True)
            )


def downgrade() -> None:
    # Remove language metadata columns
    if _table_exists('query_history'):
        if _column_exists('query_history', 'translation_provider'):
            op.drop_column('query_history', 'translation_provider')
        
        if _column_exists('query_history', 'query_retrieval'):
            op.drop_column('query_history', 'query_retrieval')
        
        if _column_exists('query_history', 'language_confidence'):
            op.drop_column('query_history', 'language_confidence')
        
        if _column_exists('query_history', 'detected_language'):
            op.drop_column('query_history', 'detected_language')












