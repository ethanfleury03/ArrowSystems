"""Initial schema migration

Creates the initial database schema based on existing models.
This migration captures the current state of the database.

Revision ID: 001_initial
Revises: 
Create Date: 2025-11-17 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite
from sqlalchemy import inspect, text

# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def _table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    # SAFETY: This migration is designed to be non-destructive:
    # - Only creates tables if they don't exist (won't touch existing tables)
    # - No data modification, deletion, or overwriting
    # - Safe to run on existing databases with data
    # Note: This migration may be applied to existing databases
    # We check if tables exist before creating them to avoid errors
    
    # Users table
    if not _table_exists('users'):
        op.create_table(
            'users',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('email', sa.String(length=255), nullable=False),
            sa.Column('name', sa.String(length=255), nullable=True),
            sa.Column('role', sa.String(length=50), nullable=False, server_default='technician'),
            sa.Column('password_hash', sa.String(length=255), nullable=True),
            sa.Column('company_name', sa.String(length=255), nullable=True),
            sa.Column('contact_name', sa.String(length=255), nullable=True),
            sa.Column('contact_phone', sa.String(length=50), nullable=True),
            sa.Column('machine_models', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('email')
        )
        op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
        op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    # Query history table
    if not _table_exists('query_history'):
        op.create_table(
            'query_history',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('query_text', sa.Text(), nullable=False),
            sa.Column('answer_text', sa.Text(), nullable=True),
            sa.Column('response_time_ms', sa.Integer(), nullable=True),
            sa.Column('metadata', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.Column('machine_name', sa.String(length=255), nullable=True),
            sa.Column('token_input', sa.Integer(), nullable=True),
            sa.Column('token_output', sa.Integer(), nullable=True),
            sa.Column('token_total', sa.Integer(), nullable=True),
            sa.Column('cost_usd', sa.Float(), nullable=True),
            sa.Column('sources_json', sa.Text(), nullable=True),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_query_history_id'), 'query_history', ['id'], unique=False)
        op.create_index(op.f('ix_query_history_user_id'), 'query_history', ['user_id'], unique=False)
        op.create_index(op.f('ix_query_history_created_at'), 'query_history', ['created_at'], unique=False)

    # Feedback table
    if not _table_exists('feedback'):
        op.create_table(
            'feedback',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('query_history_id', sa.Integer(), nullable=False),
            sa.Column('is_helpful', sa.Boolean(), nullable=False),
            sa.Column('confidence', sa.Float(), nullable=True),
            sa.Column('intent_type', sa.String(length=100), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(['query_history_id'], ['query_history.id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_feedback_id'), 'feedback', ['id'], unique=False)
        op.create_index(op.f('ix_feedback_user_id'), 'feedback', ['user_id'], unique=False)
        op.create_index(op.f('ix_feedback_query_history_id'), 'feedback', ['query_history_id'], unique=False)

    # Saved responses table
    if not _table_exists('saved_responses'):
        op.create_table(
            'saved_responses',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('query_text', sa.Text(), nullable=False),
            sa.Column('answer_text', sa.Text(), nullable=False),
            sa.Column('sources', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('user_id', 'query_text', name='uq_saved_response_user_query')
        )
        op.create_index(op.f('ix_saved_responses_id'), 'saved_responses', ['id'], unique=False)
        op.create_index(op.f('ix_saved_responses_user_id'), 'saved_responses', ['user_id'], unique=False)

    # Audit logs table
    if not _table_exists('audit_logs'):
        op.create_table(
            'audit_logs',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('timestamp', sa.DateTime(), nullable=False),
            sa.Column('level', sa.String(length=20), nullable=False, server_default='info'),
            sa.Column('event', sa.String(length=100), nullable=False),
            sa.Column('user_id', sa.String(length=255), nullable=True),
            sa.Column('role', sa.String(length=50), nullable=True),
            sa.Column('ip_address', sa.String(length=45), nullable=True),
            sa.Column('metadata', sa.JSON(), nullable=True),
            sa.Column('request_id', sa.String(length=255), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_audit_logs_id'), 'audit_logs', ['id'], unique=False)
        op.create_index(op.f('ix_audit_logs_timestamp'), 'audit_logs', ['timestamp'], unique=False)
        op.create_index(op.f('ix_audit_logs_event'), 'audit_logs', ['event'], unique=False)
        op.create_index(op.f('ix_audit_logs_user_id'), 'audit_logs', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_audit_logs_user_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_event'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_timestamp'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_id'), table_name='audit_logs')
    op.drop_table('audit_logs')
    
    op.drop_index(op.f('ix_saved_responses_user_id'), table_name='saved_responses')
    op.drop_index(op.f('ix_saved_responses_id'), table_name='saved_responses')
    op.drop_table('saved_responses')
    
    op.drop_index(op.f('ix_feedback_query_history_id'), table_name='feedback')
    op.drop_index(op.f('ix_feedback_user_id'), table_name='feedback')
    op.drop_index(op.f('ix_feedback_id'), table_name='feedback')
    op.drop_table('feedback')
    
    op.drop_index(op.f('ix_query_history_created_at'), table_name='query_history')
    op.drop_index(op.f('ix_query_history_user_id'), table_name='query_history')
    op.drop_index(op.f('ix_query_history_id'), table_name='query_history')
    op.drop_table('query_history')
    
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_table('users')
