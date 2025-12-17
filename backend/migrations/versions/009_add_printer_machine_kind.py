"""Add 'Printer' to machine_kind allowed values

This migration updates the check constraint to allow 'Printer' as a fourth machine kind option.

Revision ID: 009_add_printer_machine_kind
Revises: 008_add_machine_kind
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


def _constraint_exists(table_name: str, constraint_name: str) -> bool:
    """Check if a constraint exists on a table."""
    if not _table_exists(table_name):
        return False
    bind = op.get_bind()
    inspector = inspect(bind)
    constraints = inspector.get_check_constraints(table_name)
    return any(c['name'] == constraint_name for c in constraints)


# revision identifiers, used by Alembic.
revision = '009_add_printer_machine_kind'
down_revision = '008_add_machine_kind'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Only proceed if machine_models table exists
    if not _table_exists('machine_models'):
        return
    
    # Drop the existing check constraint if it exists
    if _constraint_exists('machine_models', 'check_machine_kind'):
        op.drop_constraint('check_machine_kind', 'machine_models', type_='check')
    
    # Create new check constraint with 'Printer' added
    op.create_check_constraint(
        'check_machine_kind',
        'machine_models',
        "machine_kind IN ('Print Engine', 'Blade Cutter', 'Laser Cutter', 'Printer')"
    )


def downgrade() -> None:
    # Only proceed if machine_models table exists
    if not _table_exists('machine_models'):
        return
    
    # Drop the updated check constraint
    if _constraint_exists('machine_models', 'check_machine_kind'):
        op.drop_constraint('check_machine_kind', 'machine_models', type_='check')
    
    # Recreate the original constraint without 'Printer'
    op.create_check_constraint(
        'check_machine_kind',
        'machine_models',
        "machine_kind IN ('Print Engine', 'Blade Cutter', 'Laser Cutter')"
    )

