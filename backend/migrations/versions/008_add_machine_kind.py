"""Add machine_kind field to machine_models table

This migration adds:
- machine_kind column to machine_models table (TEXT, NOT NULL)
- Check constraint to enforce only 3 allowed values: 'Print Engine', 'Blade Cutter', 'Laser Cutter'
- Backfills all existing rows with 'Print Engine' as default

Revision ID: 008_add_machine_kind
Revises: 007_add_auth_tokens
Create Date: 2025-01-27 00:00:00.000000

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
revision = '008_add_machine_kind'
down_revision = '007_add_auth_tokens'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Only proceed if machine_models table exists
    if not _table_exists('machine_models'):
        return
    
    # Add machine_kind column if it doesn't exist
    if not _column_exists('machine_models', 'machine_kind'):
        # Step 1: Add column as nullable first (to allow backfill)
        op.add_column('machine_models', sa.Column('machine_kind', sa.String(length=50), nullable=True))
        
        # Step 2: Backfill all existing rows with default value 'Print Engine'
        op.execute("""
            UPDATE machine_models 
            SET machine_kind = 'Print Engine' 
            WHERE machine_kind IS NULL
        """)
        
        # Step 3: Make column NOT NULL
        op.alter_column('machine_models', 'machine_kind', nullable=False, server_default='Print Engine')
        
        # Step 4: Add check constraint to enforce only 3 allowed values
        op.create_check_constraint(
            'check_machine_kind',
            'machine_models',
            "machine_kind IN ('Print Engine', 'Blade Cutter', 'Laser Cutter')"
        )
        
        # Step 5: Remove server default (we want explicit values, not defaults)
        op.alter_column('machine_models', 'machine_kind', server_default=None)


def downgrade() -> None:
    # Remove machine_kind column if it exists
    if _table_exists('machine_models') and _column_exists('machine_models', 'machine_kind'):
        # Drop check constraint first
        try:
            op.drop_constraint('check_machine_kind', 'machine_models', type_='check')
        except Exception:
            # Constraint might not exist, ignore
            pass
        
        # Drop column
        op.drop_column('machine_models', 'machine_kind')

