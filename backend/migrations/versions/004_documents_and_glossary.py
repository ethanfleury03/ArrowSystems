"""Phase 1: Documents and glossary tables

This migration adds:
- documents table (replaces document_metadata.json)
- glossary_terms table (replaces glossary.csv)

Revision ID: 004_documents_and_glossary
Revises: 003_ingestion_phase1
Create Date: 2025-01-XX XX:XX:XX.XXXXXX

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect, text

# revision identifiers, used by Alembic.
revision = '004_documents_and_glossary'
down_revision = '003_ingestion_phase1'
branch_labels = None
depends_on = None


def _table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    # SAFETY: This migration is designed to be non-destructive:
    # - Only creates tables if they don't exist
    # - No data modification, deletion, or overwriting
    # - Safe to run on existing databases with data
    
    # Documents table
    if not _table_exists('documents'):
        op.create_table(
            'documents',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('file_name', sa.String(length=500), nullable=False),
            sa.Column('gcs_path', sa.String(length=1000), nullable=True),
            sa.Column('display_name', sa.String(length=500), nullable=True),
            sa.Column('machine_model', sa.String(length=255), nullable=True),
            sa.Column('category', sa.String(length=255), nullable=True),
            sa.Column('product_family', sa.String(length=255), nullable=True),
            sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
            sa.Column('requires_admin_review', sa.Boolean(), nullable=False, server_default='false'),
            sa.Column('file_size_bytes', sa.Integer(), nullable=True),
            sa.Column('last_ingestion_date', sa.DateTime(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
            sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_documents_id'), 'documents', ['id'], unique=False)
        op.create_index(op.f('ix_documents_file_name'), 'documents', ['file_name'], unique=False)
        op.create_index(op.f('ix_documents_is_active'), 'documents', ['is_active'], unique=False)
        op.create_index(op.f('ix_documents_machine_model'), 'documents', ['machine_model'], unique=False)
        op.create_index(op.f('ix_documents_created_at'), 'documents', ['created_at'], unique=False)
    
    # Glossary terms table
    if not _table_exists('glossary_terms'):
        op.create_table(
            'glossary_terms',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('term', sa.String(length=255), nullable=False),
            sa.Column('definition', sa.Text(), nullable=False),
            sa.Column('aliases', sa.JSON(), nullable=True),  # PostgreSQL JSON, SQLite TEXT
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
            sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_glossary_terms_id'), 'glossary_terms', ['id'], unique=False)
        op.create_index(op.f('ix_glossary_terms_term'), 'glossary_terms', ['term'], unique=False)


def downgrade() -> None:
    # Drop glossary terms table
    if _table_exists('glossary_terms'):
        op.drop_index(op.f('ix_glossary_terms_term'), table_name='glossary_terms')
        op.drop_index(op.f('ix_glossary_terms_id'), table_name='glossary_terms')
        op.drop_table('glossary_terms')
    
    # Drop documents table
    if _table_exists('documents'):
        op.drop_index(op.f('ix_documents_created_at'), table_name='documents')
        op.drop_index(op.f('ix_documents_machine_model'), table_name='documents')
        op.drop_index(op.f('ix_documents_is_active'), table_name='documents')
        op.drop_index(op.f('ix_documents_file_name'), table_name='documents')
        op.drop_index(op.f('ix_documents_id'), table_name='documents')
        op.drop_table('documents')

