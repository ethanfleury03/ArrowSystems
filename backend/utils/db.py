from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import Session, declarative_base, relationship, scoped_session, sessionmaker

_backend_dir = Path(__file__).resolve().parent.parent
_env_db_path = os.getenv("SQLITE_DB_PATH")
if _env_db_path:
    DEFAULT_DB_PATH = str(Path(_env_db_path).resolve())
else:
    DEFAULT_DB_PATH = str((_backend_dir / "database.sqlite").resolve())

DATABASE_URL = f"sqlite:///{DEFAULT_DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)

SessionLocal = scoped_session(
    sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255))
    role = Column(String(50), default="technician", nullable=False)
    password_hash = Column(String(255))
    company_name = Column(String(255))
    contact_name = Column(String(255))
    contact_phone = Column(String(50))
    machine_models = Column(JSON, default=list)  # List of machine model strings (e.g., ["330R", "DuraFlex"])
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    query_history = relationship("QueryHistory", back_populates="user", cascade="all, delete", passive_deletes=True)
    feedback = relationship("Feedback", back_populates="user", cascade="all, delete", passive_deletes=True)
    saved_responses = relationship("SavedResponse", back_populates="user", cascade="all, delete", passive_deletes=True)


class QueryHistory(Base):
    __tablename__ = "query_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    answer_text = Column(Text)
    response_time_ms = Column(Integer)
    metadata_json = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Analytics columns
    machine_name = Column(String(255), nullable=True)
    token_input = Column(Integer, nullable=True)
    token_output = Column(Integer, nullable=True)
    token_total = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)
    sources_json = Column(Text, nullable=True)

    user = relationship("User", back_populates="query_history")
    feedback = relationship("Feedback", back_populates="query", cascade="all, delete", passive_deletes=True)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    query_history_id = Column(Integer, ForeignKey("query_history.id", ondelete="CASCADE"), nullable=False, index=True)
    is_helpful = Column(Boolean, nullable=False)
    confidence = Column(Float)
    intent_type = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="feedback")
    query = relationship("QueryHistory", back_populates="feedback")


class SavedResponse(Base):
    __tablename__ = "saved_responses"
    __table_args__ = (UniqueConstraint("user_id", "query_text", name="uq_saved_response_user_query"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=False)
    sources = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="saved_responses")


class AuditLog(Base):
    """Audit log table for admin-facing events."""
    __tablename__ = "audit_logs"
    # Don't use __table_args__ with Index() to avoid conflicts with manual table creation
    # Indexes are created manually in ensure_audit_logs_table()

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    level = Column(String(20), nullable=False, default="info")  # "info", "warning", "error"
    event = Column(String(100), nullable=False, index=True)  # Event name like "user_login", "manual_upload_start"
    user_id = Column(String(255), nullable=True, index=True)  # User email or ID
    role = Column(String(50), nullable=True)  # User role
    ip_address = Column(String(45), nullable=True)  # IPv4 or IPv6
    event_metadata = Column("metadata", JSON, nullable=True, default=dict)  # Additional structured data (database column name: metadata)
    request_id = Column(String(255), nullable=True)  # Request ID from context


def ensure_analytics_columns() -> None:
    """Ensure analytics columns exist in query_history table (SQLite-safe migration)."""
    with engine.begin() as connection:
        inspector = inspect(connection)
        try:
            existing_columns = {column["name"] for column in inspector.get_columns("query_history")}
        except Exception:
            # Table doesn't exist yet, will be created by create_all
            return
        
        # Add analytics columns if they don't exist
        if "machine_name" not in existing_columns:
            connection.execute(text("ALTER TABLE query_history ADD COLUMN machine_name VARCHAR(255)"))
        if "token_input" not in existing_columns:
            connection.execute(text("ALTER TABLE query_history ADD COLUMN token_input INTEGER"))
        if "token_output" not in existing_columns:
            connection.execute(text("ALTER TABLE query_history ADD COLUMN token_output INTEGER"))
        if "token_total" not in existing_columns:
            connection.execute(text("ALTER TABLE query_history ADD COLUMN token_total INTEGER"))
        if "cost_usd" not in existing_columns:
            connection.execute(text("ALTER TABLE query_history ADD COLUMN cost_usd REAL"))
        if "sources_json" not in existing_columns:
            connection.execute(text("ALTER TABLE query_history ADD COLUMN sources_json TEXT"))


def ensure_audit_logs_table() -> None:
    """Ensure audit_logs table exists (SQLite-safe migration)."""
    with engine.begin() as connection:
        inspector = inspect(connection)
        try:
            # Check if table exists
            tables = inspector.get_table_names()
            if "audit_logs" not in tables:
                # Create table manually (before SQLAlchemy tries to create it)
                # Use TEXT for metadata column (SQLite doesn't have native JSON)
                connection.execute(text("""
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        level VARCHAR(20) NOT NULL DEFAULT 'info',
                        event VARCHAR(100) NOT NULL,
                        user_id VARCHAR(255),
                        role VARCHAR(50),
                        ip_address VARCHAR(45),
                        metadata TEXT,
                        request_id VARCHAR(255)
                    )
                """))
                # Create indexes if they don't exist
                try:
                    connection.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp DESC)"))
                except Exception:
                    pass  # Index might already exist
                try:
                    connection.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id)"))
                except Exception:
                    pass  # Index might already exist
                try:
                    connection.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_logs_event ON audit_logs(event)"))
                except Exception:
                    pass  # Index might already exist
                return
            
            # Table exists, check if columns exist and add missing ones
            existing_columns = {column["name"] for column in inspector.get_columns("audit_logs")}
            
            if "request_id" not in existing_columns:
                try:
                    connection.execute(text("ALTER TABLE audit_logs ADD COLUMN request_id VARCHAR(255)"))
                    try:
                        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_logs_request_id ON audit_logs(request_id)"))
                    except Exception:
                        pass  # Index might already exist
                except Exception:
                    pass  # Column might already exist
            
            # Ensure metadata column exists
            if "metadata" not in existing_columns:
                try:
                    connection.execute(text("ALTER TABLE audit_logs ADD COLUMN metadata TEXT"))
                except Exception:
                    pass  # Column might already exist
        except Exception as e:
            # Table creation/update failed, but that's okay if it already exists or was created by SQLAlchemy
            # Log the error for debugging but don't raise
            import logging
            logging.getLogger(__name__).warning(f"Audit logs table migration warning: {e}")


def ensure_user_columns() -> None:
    """Ensure user columns exist (SQLite-safe migration)."""
    with engine.begin() as connection:
        inspector = inspect(connection)
        try:
            existing_columns = {column["name"] for column in inspector.get_columns("users")}
        except Exception:
            # Table doesn't exist yet, will be created by create_all
            return
        
        # Add user columns if they don't exist
        if "company_name" not in existing_columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN company_name VARCHAR(255)"))
        if "contact_name" not in existing_columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN contact_name VARCHAR(255)"))
        if "contact_phone" not in existing_columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN contact_phone VARCHAR(50)"))
        if "machine_models" not in existing_columns:
            # SQLite doesn't have native JSON type, so we use TEXT and store JSON string
            connection.execute(text("ALTER TABLE users ADD COLUMN machine_models TEXT DEFAULT '[]'"))


def init_db() -> None:
    os.makedirs(os.path.dirname(DEFAULT_DB_PATH) or ".", exist_ok=True)
    
    # First ensure audit_logs table exists (manual creation to handle migrations)
    # This must be done BEFORE Base.metadata.create_all() to avoid conflicts
    ensure_audit_logs_table()
    
    # Create all other tables that don't exist yet
    # We need to exclude audit_logs from SQLAlchemy's automatic creation
    # since we're managing it manually to avoid "table already exists" errors
    metadata = Base.metadata
    
    # Temporarily remove audit_logs from metadata to prevent SQLAlchemy from creating it
    # We'll create other tables individually to avoid conflicts
    audit_logs_table = metadata.tables.get('audit_logs')
    audit_logs_removed = False
    
    if audit_logs_table is not None:
        try:
            # Remove audit_logs from metadata temporarily
            metadata.remove(audit_logs_table)
            audit_logs_removed = True
        except (KeyError, ValueError, AttributeError):
            # Table not in metadata or already removed
            pass
    
    # Create all other tables (excluding audit_logs)
    try:
        # Create all tables except audit_logs
        # checkfirst=True will skip tables that already exist
        metadata.create_all(bind=engine, checkfirst=True)
    except Exception as e:
        # If there's an error, check if it's related to audit_logs
        # If so, that's expected since we manage it manually
        error_str = str(e).lower()
        if "already exists" not in error_str and "audit_logs" not in error_str:
            # Unexpected error - log it
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Table creation warning: {e}")
    
    # IMPORTANT: Add audit_logs back to metadata so ORM queries work
    # We removed it temporarily to prevent auto-creation, but we need it back for queries
    if audit_logs_removed and audit_logs_table is not None:
        try:
            # Re-add the table to metadata so queries work
            # This doesn't recreate the table, it just registers it for ORM use
            Base.metadata._add_table(audit_logs_table.name, audit_logs_table.schema, audit_logs_table)
            # Alternative: Just ensure the model is bound correctly
            # The model should still work even if removed from metadata temporarily
        except Exception:
            # If re-adding fails, that's okay - the model class should still work
            pass
    
    # Ensure user columns exist
    ensure_user_columns()
    
    # Ensure analytics columns exist
    ensure_analytics_columns()


T = TypeVar("T")


async def run_sync(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    return await asyncio.to_thread(func, *args, **kwargs)

