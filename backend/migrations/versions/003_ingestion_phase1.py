"""Ingestion Phase 1: Document ingestion metadata and machine models

This migration adds:
- document_ingestion_metadata table (tracks ingestion status)
- machine_models table (stores machine models dynamically)

Revision ID: 003_ingestion_phase1
Revises: 002_schema_fixes
Create Date: 2025-01-XX XX:XX:XX.XXXXXX

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect, text
import uuid

# revision identifiers, used by Alembic.
revision = '003_ingestion_phase1'
down_revision = '002_schema_fixes'
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
    
    # Machine models table
    if not _table_exists('machine_models'):
        op.create_table(
            'machine_models',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('name', sa.String(length=255), nullable=False, unique=True),
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
            sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('name')
        )
        op.create_index(op.f('ix_machine_models_id'), 'machine_models', ['id'], unique=False)
        op.create_index(op.f('ix_machine_models_name'), 'machine_models', ['name'], unique=True)
        
        # Populate with existing machine models from config
        # This ensures existing models are available
        op.execute("""
            INSERT OR IGNORE INTO machine_models (name, created_at, updated_at)
            VALUES 
                ('2800 Series Mini Laser Pro', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                ('Duraflex', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                ('Anycut', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                ('anyCutII', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                ('anyCutIII', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                ('Anytron AnyJet', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                ('ANYTRON Any-002', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                ('Digital Die Cutter VR350', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                ('DuraLink', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                ('DuraBolt', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                ('DuraCore', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                ('EZCut 330', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                ('EZCut 350R', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
                ('GENERAL', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """)
    
    # Document ingestion metadata table
    if not _table_exists('document_ingestion_metadata'):
        op.create_table(
            'document_ingestion_metadata',
            sa.Column('id', sa.String(length=36), nullable=False),  # UUID as string
            sa.Column('filename', sa.String(length=500), nullable=False),
            sa.Column('machine_model', sa.String(length=255), nullable=False),
            sa.Column('status', sa.String(length=50), nullable=False, server_default='PENDING_INGESTION'),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('file_path', sa.String(length=1000), nullable=True),
            sa.Column('file_size_bytes', sa.Integer(), nullable=True),
            sa.Column('error_message', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
            sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_document_ingestion_metadata_id'), 'document_ingestion_metadata', ['id'], unique=True)
        op.create_index(op.f('ix_document_ingestion_metadata_filename'), 'document_ingestion_metadata', ['filename'], unique=False)
        op.create_index(op.f('ix_document_ingestion_metadata_status'), 'document_ingestion_metadata', ['status'], unique=False)
        op.create_index(op.f('ix_document_ingestion_metadata_machine_model'), 'document_ingestion_metadata', ['machine_model'], unique=False)
        op.create_index(op.f('ix_document_ingestion_metadata_created_at'), 'document_ingestion_metadata', ['created_at'], unique=False)


def downgrade() -> None:
    # Drop document ingestion metadata table
    op.drop_index(op.f('ix_document_ingestion_metadata_created_at'), table_name='document_ingestion_metadata')
    op.drop_index(op.f('ix_document_ingestion_metadata_machine_model'), table_name='document_ingestion_metadata')
    op.drop_index(op.f('ix_document_ingestion_metadata_status'), table_name='document_ingestion_metadata')
    op.drop_index(op.f('ix_document_ingestion_metadata_filename'), table_name='document_ingestion_metadata')
    op.drop_index(op.f('ix_document_ingestion_metadata_id'), table_name='document_ingestion_metadata')
    op.drop_table('document_ingestion_metadata')
    
    # Drop machine models table
    op.drop_index(op.f('ix_machine_models_name'), table_name='machine_models')
    op.drop_index(op.f('ix_machine_models_id'), table_name='machine_models')
    op.drop_table('machine_models')

