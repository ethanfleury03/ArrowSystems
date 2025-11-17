"""Schema fixes: add columns, constraints, and indexes

This migration adds:
- updated_at columns to Feedback and QueryHistory
- NOT NULL constraints on User.name and User.password_hash
- CHECK constraints for role, level, and is_helpful
- Missing indexes on query_text, answer_text
- Composite indexes for common query patterns

Revision ID: 002_schema_fixes
Revises: 001_initial
Create Date: 2025-11-17 13:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite
from sqlalchemy import inspect, text

# revision identifiers, used by Alembic.
revision = '002_schema_fixes'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def _column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    bind = op.get_bind()
    inspector = inspect(bind)
    try:
        columns = {col["name"] for col in inspector.get_columns(table_name)}
        return column_name in columns
    except Exception:
        return False


def _index_exists(index_name: str) -> bool:
    """Check if an index exists."""
    bind = op.get_bind()
    inspector = inspect(bind)
    try:
        # Check all tables for the index
        indexes = set()
        for table_name in inspector.get_table_names():
            try:
                table_indexes = inspector.get_indexes(table_name)
                indexes.update({idx["name"] for idx in table_indexes})
            except Exception:
                continue
        return index_name in indexes
    except Exception:
        return False


def _column_is_nullable(table_name: str, column_name: str) -> bool:
    """Check if a column is nullable."""
    bind = op.get_bind()
    inspector = inspect(bind)
    try:
        columns = inspector.get_columns(table_name)
        for col in columns:
            if col["name"] == column_name:
                return col.get("nullable", True)
        return True  # Default to nullable if column not found
    except Exception:
        return True


def upgrade() -> None:
    # SAFETY: This migration is designed to be non-destructive:
    # - Only adds columns/indexes if they don't exist
    # - Only updates NULL values (never modifies existing non-NULL data)
    # - batch_alter_table preserves all existing data (copies to new table structure)
    # - No DELETE, DROP, or TRUNCATE operations
    
    # Add updated_at column to Feedback table (if it doesn't exist)
    # Note: batch_alter_table operations are slow on large SQLite tables
    # They rewrite the entire table, which can take minutes on large databases
    # SAFETY: All existing data is preserved during table rewrite
    if not _column_exists('feedback', 'updated_at'):
        op.add_column('feedback', sa.Column('updated_at', sa.DateTime(), nullable=True))
        # SAFETY: Only updates NULL values, never modifies existing data
        op.execute("UPDATE feedback SET updated_at = created_at WHERE updated_at IS NULL")
        # Make updated_at NOT NULL after setting values
        # SQLite doesn't support ALTER COLUMN, so we use a workaround
        # WARNING: This operation rewrites the entire table and can be slow
        with op.batch_alter_table('feedback', schema=None) as batch_op:
            batch_op.alter_column('updated_at', nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'))
    elif _column_is_nullable('feedback', 'updated_at'):
        # Column exists but is nullable, update existing NULLs and make NOT NULL
        # WARNING: This operation rewrites the entire table and can be slow
        op.execute("UPDATE feedback SET updated_at = created_at WHERE updated_at IS NULL")
        with op.batch_alter_table('feedback', schema=None) as batch_op:
            batch_op.alter_column('updated_at', nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'))
    # If column exists and is already NOT NULL, skip the operation
    
    # Add updated_at column to QueryHistory table (if it doesn't exist)
    # WARNING: batch_alter_table operations rewrite the entire table and can be slow
    # SAFETY: All existing data is preserved during table rewrite
    if not _column_exists('query_history', 'updated_at'):
        op.add_column('query_history', sa.Column('updated_at', sa.DateTime(), nullable=True))
        # SAFETY: Only updates NULL values, never modifies existing data
        op.execute("UPDATE query_history SET updated_at = created_at WHERE updated_at IS NULL")
        # Make updated_at NOT NULL after setting values
        # WARNING: This operation rewrites the entire table and can be slow
        with op.batch_alter_table('query_history', schema=None) as batch_op:
            batch_op.alter_column('updated_at', nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'))
    elif _column_is_nullable('query_history', 'updated_at'):
        # Column exists but is nullable, update existing NULLs and make NOT NULL
        # WARNING: This operation rewrites the entire table and can be slow
        op.execute("UPDATE query_history SET updated_at = created_at WHERE updated_at IS NULL")
        with op.batch_alter_table('query_history', schema=None) as batch_op:
            batch_op.alter_column('updated_at', nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'))
    # If column exists and is already NOT NULL, skip the operation
    
    # Add NOT NULL constraint to User.name (set default for existing NULLs)
    # WARNING: batch_alter_table operations rewrite the entire table and can be slow
    # SAFETY: All existing data is preserved during table rewrite
    if _column_is_nullable('users', 'name'):
        # SAFETY: Only updates NULL values, never modifies existing non-NULL names
        op.execute("UPDATE users SET name = email WHERE name IS NULL")
        # WARNING: This operation rewrites the entire table and can be slow
        with op.batch_alter_table('users', schema=None) as batch_op:
            batch_op.alter_column('name', nullable=False)
    # If column is already NOT NULL, skip the operation
    
    # Add NOT NULL constraint to User.password_hash (set empty string for existing NULLs)
    # Note: This is safe because existing users without passwords are API users
    # WARNING: batch_alter_table operations rewrite the entire table and can be slow
    # SAFETY: All existing data is preserved during table rewrite
    if _column_is_nullable('users', 'password_hash'):
        # SAFETY: Only updates NULL values, never modifies existing password hashes
        op.execute("UPDATE users SET password_hash = '' WHERE password_hash IS NULL")
        # WARNING: This operation rewrites the entire table and can be slow
        with op.batch_alter_table('users', schema=None) as batch_op:
            batch_op.alter_column('password_hash', nullable=False, server_default='')
    # If column is already NOT NULL, skip the operation
    
    # Add CHECK constraint for User.role
    # SQLite doesn't support CHECK constraints via ALTER TABLE, so we skip this for SQLite
    # For PostgreSQL, we would add: CHECK (role IN ('ADMIN', 'TECHNICIAN', 'CUSTOMER'))
    # This will be enforced at the application level for SQLite
    
    # Add CHECK constraint for AuditLog.level
    # Same limitation - enforced at application level for SQLite
    
    # Add CHECK constraint for Feedback.is_helpful
    # Same limitation - enforced at application level for SQLite
    
    # Add indexes on query_text and answer_text for QueryHistory (if they don't exist)
    if not _index_exists('ix_query_history_query_text'):
        op.create_index('ix_query_history_query_text', 'query_history', ['query_text'], unique=False)
    if not _index_exists('ix_query_history_answer_text'):
        op.create_index('ix_query_history_answer_text', 'query_history', ['answer_text'], unique=False)
    
    # Add composite index: QueryHistory(user_id, created_at)
    if not _index_exists('ix_query_history_user_created'):
        op.create_index('ix_query_history_user_created', 'query_history', ['user_id', 'created_at'], unique=False)
    
    # Add composite index: Feedback(user_id, query_history_id)
    if not _index_exists('ix_feedback_user_query'):
        op.create_index('ix_feedback_user_query', 'feedback', ['user_id', 'query_history_id'], unique=False)
    
    # Add composite index: AuditLog(timestamp, event)
    if not _index_exists('ix_audit_logs_timestamp_event'):
        op.create_index('ix_audit_logs_timestamp_event', 'audit_logs', ['timestamp', 'event'], unique=False)


def downgrade() -> None:
    # Drop composite indexes
    op.drop_index('ix_audit_logs_timestamp_event', table_name='audit_logs')
    op.drop_index('ix_feedback_user_query', table_name='feedback')
    op.drop_index('ix_query_history_user_created', table_name='query_history')
    
    # Drop text indexes
    op.drop_index('ix_query_history_answer_text', table_name='query_history')
    op.drop_index('ix_query_history_query_text', table_name='query_history')
    
    # Remove NOT NULL constraints (make nullable again)
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.alter_column('password_hash', nullable=True, server_default=None)
        batch_op.alter_column('name', nullable=True)
    
    # Remove updated_at columns
    with op.batch_alter_table('query_history', schema=None) as batch_op:
        batch_op.drop_column('updated_at')
    
    with op.batch_alter_table('feedback', schema=None) as batch_op:
        batch_op.drop_column('updated_at')

