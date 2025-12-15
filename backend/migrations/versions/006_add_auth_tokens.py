"""Add auth_tokens table for invite and password reset flows

This migration adds:
- auth_tokens table for storing hashed invite and reset tokens
- Indexes on token_hash and user_id for fast lookups
- Foreign key to users.id with CASCADE delete

Revision ID: 006_add_auth_tokens
Revises: 005_add_conversation_id
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
revision = '006_add_auth_tokens'
down_revision = '005_add_conversation_id'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create auth_tokens table if it doesn't exist
    if not _table_exists('auth_tokens'):
        op.create_table(
            'auth_tokens',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('token_hash', sa.String(length=255), nullable=False),
            sa.Column('purpose', sa.String(length=50), nullable=False),
            sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
            sa.Column('used', sa.Boolean(), nullable=False, server_default='false'),
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id')
        )
        # Create indexes
        op.create_index('ix_auth_tokens_id', 'auth_tokens', ['id'])
        op.create_index('ix_auth_tokens_user_id', 'auth_tokens', ['user_id'])
        op.create_index('ix_auth_tokens_token_hash', 'auth_tokens', ['token_hash'])


def downgrade() -> None:
    # Remove auth_tokens table and indexes
    if _table_exists('auth_tokens'):
        op.drop_index('ix_auth_tokens_token_hash', table_name='auth_tokens')
        op.drop_index('ix_auth_tokens_user_id', table_name='auth_tokens')
        op.drop_index('ix_auth_tokens_id', table_name='auth_tokens')
        op.drop_table('auth_tokens')

