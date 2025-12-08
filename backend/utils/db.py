from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
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
    event,
    func,
    inspect,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, relationship, scoped_session, sessionmaker

from ..config.env import settings

logger = logging.getLogger(__name__)

_backend_dir = Path(__file__).resolve().parent.parent


def _get_database_url() -> str:
    """
    Get database URL from DATABASE_URL environment variable.
    
    DATABASE_URL is the single source of truth for database connections.
    Must be a PostgreSQL connection string (e.g., postgresql://user:pass@host:port/dbname).
    
    In production, this must be loaded from Google Secret Manager.
    In development, this can come from .env file via python-dotenv.
    """
    # Use settings.DATABASE_URL if available (loaded from env config)
    # Otherwise fall back to direct env var access
    from ..config.env import settings
    if hasattr(settings, 'DATABASE_URL'):
        database_url = settings.DATABASE_URL
    else:
        # Fallback for backwards compatibility
        if settings.is_prod:
            try:
                database_url = os.environ["DATABASE_URL"]
            except KeyError:
                raise RuntimeError(
                    "DATABASE_URL environment variable is REQUIRED in production but not set. "
                    "Ensure Cloud Run is configured to load this from Google Secret Manager."
                )
        else:
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                raise RuntimeError(
                    "DATABASE_URL environment variable is required. "
                    "Set it in your .env file for local development."
                )
    
    # Fail fast if SQLite is detected
    if database_url.startswith("sqlite://") or database_url.startswith("sqlite:///"):
        raise RuntimeError(
            "SQLite is not supported. "
            "Provide a PostgreSQL connection string via DATABASE_URL. "
            f"Received: {database_url[:50]}..."
        )
    
    return database_url


def _validate_database_connection(engine_instance: Engine, database_url: str) -> None:
    """
    Validate database connection on startup.
    Raises RuntimeError if connection fails.
    """
    try:
        with engine_instance.connect() as connection:
            result = connection.execute(text("SELECT 1")).scalar()
            if result != 1:
                raise RuntimeError("Database connection test failed: SELECT 1 did not return 1")
        # Extract host info for logging (mask password)
        host_info = database_url.split('@')[1] if '@' in database_url else 'database'
        logger.info(f"Connected to PostgreSQL at {host_info}")
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}", exc_info=True)
        raise RuntimeError(f"Database connection failed: {e}") from e


def get_engine() -> Engine:
    """
    Get database engine configured for PostgreSQL.
    """
    database_url = _get_database_url()
    
    # PostgreSQL connection configuration
    engine = create_engine(
        database_url,
        pool_pre_ping=True,  # Verify connections before using them
        pool_recycle=3600,   # Recycle connections after 1 hour
        future=True,
    )
    
    return engine


# Initialize engine
DATABASE_URL = _get_database_url()
engine = get_engine()

SessionLocal = scoped_session(
    sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)
)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)  # NOT NULL after migration
    role = Column(String(50), default="technician", nullable=False)
    password_hash = Column(String(255), nullable=False, server_default="")  # NOT NULL after migration
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

    # NOTE: query_history table in DB does NOT currently have an updated_at column.
    # We only use created_at for Query Insights.
    # created_at is timezone-aware UTC to ensure proper serialization and frontend display.
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    conversation_id = Column(String(255), nullable=True, index=True)  # Groups queries into conversations
    query_text = Column(Text, nullable=False)
    answer_text = Column(Text)
    response_time_ms = Column(Integer)
    metadata_json = Column("metadata", JSON, default=dict)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )
    
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
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

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


class MachineModel(Base):
    """Machine model registry table."""
    __tablename__ = "machine_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class DocumentIngestionMetadata(Base):
    """Document ingestion metadata table for tracking ingestion status."""
    __tablename__ = "document_ingestion_metadata"

    id = Column(String(36), primary_key=True, index=True)  # UUID as string
    filename = Column(String(500), nullable=False, index=True)
    machine_model = Column(String(255), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="PENDING_INGESTION", index=True)
    description = Column(Text, nullable=True)
    file_path = Column(String(1000), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class Document(Base):
    """
    Document metadata table for storing document information.
    Replaces the document_metadata.json file.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(500), nullable=False, index=True)  # Original filename
    gcs_path = Column(String(1000), nullable=True)  # Cloud Storage path: gs://bucket/path or relative path
    display_name = Column(String(500), nullable=True)  # Name to show in UI (defaults to file_name)
    machine_model = Column(String(255), nullable=True, index=True)  # Machine model(s) - can be JSON array string for multiple
    category = Column(String(255), nullable=True)
    product_family = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    requires_admin_review = Column(Boolean, default=False, nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    last_ingestion_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index('ix_documents_file_name', 'file_name'),
        Index('ix_documents_is_active', 'is_active'),
        Index('ix_documents_machine_model', 'machine_model'),
    )


class GlossaryTerm(Base):
    """
    Glossary terms table for storing glossary definitions.
    Replaces the glossary.csv file.
    """
    __tablename__ = "glossary_terms"

    id = Column(Integer, primary_key=True, index=True)
    term = Column(String(255), nullable=False, index=True)
    definition = Column(Text, nullable=False)
    aliases = Column(JSON, nullable=True)  # List of alias strings (JSON array)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index('ix_glossary_terms_term', 'term'),
    )


def ensure_analytics_columns() -> None:
    """Ensure analytics columns exist in query_history table."""
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
    """Ensure audit_logs table exists."""
    # Table is managed by SQLAlchemy Base.metadata.create_all()
    # This function is kept for backward compatibility but does nothing
    # as PostgreSQL handles JSON columns natively
    pass


def ensure_user_columns() -> None:
    """Ensure user columns exist."""
    # Columns are managed by SQLAlchemy Base.metadata.create_all()
    # This function is kept for backward compatibility but does nothing
    # as PostgreSQL handles JSON columns natively
    pass


def check_database_integrity() -> tuple[bool, str]:
    """
    Check database connection health.
    
    Returns:
        Tuple of (is_healthy, message)
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1")).scalar()
            if result == 1:
                return True, "ok"
            else:
                return False, "Connection test failed"
    except Exception as e:
        logger.error(f"Database connection check failed: {e}", exc_info=True)
        return False, f"Connection error: {str(e)}"


def init_db() -> None:
    """Initialize database: create tables and run migrations."""
    # Create all tables that don't exist yet
    # checkfirst=True will skip tables that already exist
    Base.metadata.create_all(bind=engine, checkfirst=True)
    
    # Ensure user columns exist (for backward compatibility)
    ensure_user_columns()
    
    # Ensure analytics columns exist (for backward compatibility)
    ensure_analytics_columns()


T = TypeVar("T")


async def run_sync(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    return await asyncio.to_thread(func, *args, **kwargs)

