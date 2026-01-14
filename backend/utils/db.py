from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import urlparse

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
    PrimaryKeyConstraint,
    create_engine,
    event,
    func,
    inspect,
    text,
    CheckConstraint,
    Table,
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
    
    # Extract connection info for logging (mask password)
    parsed = urlparse(database_url)
    # Parse scheme to extract dialect and driver separately
    scheme = parsed.scheme
    if "+" in scheme:
        dialect = scheme.split("+", 1)[0]
        driver = scheme.split("+", 1)[1]
    else:
        dialect = scheme
        driver = None
    
    # Normalize dialect names
    if dialect in ("postgres", "postgresql"):
        dialect = "postgresql"
    
    host = parsed.hostname or "unknown"
    port = parsed.port or 5432
    db_name = parsed.path.lstrip("/") if parsed.path else "unknown"
    user = parsed.username or "unknown"
    
    # Log database connection info once at startup
    driver_str = f" driver={driver}" if driver else ""
    logger.info(
        f"Database engine initialized: dialect={dialect}{driver_str}, host={host}, port={port}, "
        f"database={db_name}, user={user}, sqlite_fallback=False"
    )
    print(
        f"[DB_INIT] dialect={dialect}{driver_str} host={host} port={port} database={db_name} user={user} sqlite_fallback=False",
        flush=True
    )
    
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

    # NOTE (Query Insights associations):
    # - Customers are identified by role == "CUSTOMER"
    # - Technicians are identified by role == "TECHNICIAN"
    # - There is no explicit organization/customer link table.
    # - For Query Insights we treat users with the same company_name as belonging
    #   to the same customer org, so a customer's queries include:
    #   * Their own queries (role "CUSTOMER")
    #   * Queries from technicians (role "TECHNICIAN") that share company_name.

    query_history = relationship("QueryHistory", back_populates="user", cascade="all, delete", passive_deletes=True)
    feedback = relationship("Feedback", back_populates="user", cascade="all, delete", passive_deletes=True)
    saved_responses = relationship("SavedResponse", back_populates="user", cascade="all, delete", passive_deletes=True)
    auth_tokens = relationship("AuthToken", back_populates="user", cascade="all, delete", passive_deletes=True)


class AuthToken(Base):
    """Authentication tokens for invite and password reset flows."""
    __tablename__ = "auth_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash = Column(String(255), nullable=False, index=True)
    purpose = Column(String(50), nullable=False)  # e.g., "invite", "reset"
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    user = relationship("User", back_populates="auth_tokens")


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
    
    # Language metadata columns
    detected_language = Column(String(10), nullable=True)  # ISO 639-1 language code (e.g., "en", "es")
    language_confidence = Column(Float, nullable=True)  # 0.0 to 1.0
    query_retrieval = Column(Text, nullable=True)  # English query used for retrieval
    translation_provider = Column(String(50), nullable=True)  # "llm", "none", etc.

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


class MachineKind(str, Enum):
    """Machine kind enumeration."""
    PRINT_ENGINE = "Print Engine"
    BLADE_CUTTER = "Blade Cutter"
    LASER_CUTTER = "Laser Cutter"
    PRINTER = "Printer"


class MachineModel(Base):
    """Machine model registry table."""
    __tablename__ = "machine_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    machine_kind = Column(String(50), nullable=False, default=MachineKind.PRINT_ENGINE.value)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        CheckConstraint(
            "machine_kind IN ('Print Engine', 'Blade Cutter', 'Laser Cutter', 'Printer')",
            name='check_machine_kind'
        ),
    )

    # Many-to-many: documents tagged with this machine model
    documents = relationship(
        "Document",
        secondary="document_machine_models",
        back_populates="machine_models",
        lazy="selectin",
    )


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

    # Many-to-many: machine models associated with this document (canonical source)
    machine_models = relationship(
        "MachineModel",
        secondary="document_machine_models",
        back_populates="documents",
        lazy="selectin",
    )


# Association table: documents ↔ machine_models
document_machine_models = Table(
    "document_machine_models",
    Base.metadata,
    Column("document_id", Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
    Column("machine_model_id", Integer, ForeignKey("machine_models.id", ondelete="CASCADE"), nullable=False),
    UniqueConstraint("document_id", "machine_model_id", name="uq_document_machine_models"),
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


# ============================================================================
# Ticket Scraper Tables (migrated from SQLite)
# ============================================================================

class TicketIndex(Base):
    """
    Stage 1: Cheap indexing of all tickets.
    Mirrors Scraper/db.py tickets_index table.
    """
    __tablename__ = "tickets_index"

    ticket_id = Column(String(255), primary_key=True)
    status = Column(String(50), nullable=True)
    subject = Column(Text, nullable=True)
    requester_id = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=True)
    is_solved = Column(Boolean, nullable=False, default=False)
    indexed_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index('idx_tickets_index_is_solved', 'is_solved'),
        Index('idx_tickets_index_status', 'status'),
    )


class TicketDetail(Base):
    """
    Stage 2: Detailed conversation JSON for solved tickets.
    Mirrors Scraper/db.py tickets_detail table.
    """
    __tablename__ = "tickets_detail"

    ticket_id = Column(String(255), ForeignKey("tickets_index.ticket_id", ondelete="CASCADE"), primary_key=True)
    conversation_json = Column(JSON, nullable=False)  # JSONB in Postgres, stored as JSON
    built_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class TicketSummary(Base):
    """
    Stage 3: Structured problem/solution extraction.
    Mirrors Scraper/db.py ticket_summaries table.
    """
    __tablename__ = "ticket_summaries"

    ticket_id = Column(String(255), ForeignKey("tickets_detail.ticket_id", ondelete="CASCADE"), primary_key=True)
    subject = Column(Text, nullable=True)
    status = Column(String(50), nullable=True)
    problem_text = Column(Text, nullable=True)
    solution_text = Column(Text, nullable=True)
    key_quotes = Column(Text, nullable=True)
    resolution_confirmed = Column(Boolean, nullable=False, default=False)
    message_count = Column(Integer, nullable=True)
    attachments_count = Column(Integer, nullable=True)
    onsite_required = Column(Boolean, nullable=False, default=False)
    resolution_mode = Column(String(50), nullable=False, default="unknown")
    resolution_mode_confidence = Column(Float, nullable=False, default=0.0)
    onsite_signals = Column(Text, nullable=True)
    embedding_text = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=True)
    built_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index('idx_ticket_summaries_resolution_confirmed', 'resolution_confirmed'),
        Index('idx_ticket_summaries_status', 'status'),
    )


class TicketJudgement(Base):
    """
    LLM-based cache eligibility classification.
    Mirrors Scraper/db.py ticket_judgements table.
    """
    __tablename__ = "ticket_judgements"

    ticket_id = Column(String(255), ForeignKey("tickets_detail.ticket_id", ondelete="CASCADE"), primary_key=True)
    cache_eligible = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    problem = Column(Text, nullable=True)
    resolution_steps_json = Column(JSON, nullable=True)  # JSONB in Postgres
    confirmation = Column(Text, nullable=True)
    evidence_json = Column(JSON, nullable=True)  # JSONB in Postgres
    blockers_json = Column(JSON, nullable=True)  # JSONB in Postgres
    model = Column(String(255), nullable=False)
    prompt_version = Column(String(255), nullable=False)
    judged_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    raw_response_json = Column(JSON, nullable=False)  # JSONB in Postgres
    review_status = Column(String(50), nullable=True)
    review_reason = Column(Text, nullable=True)
    review_reasons_json = Column(JSON, nullable=True)  # JSONB in Postgres
    reviewed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index('idx_ticket_judgements_cache_eligible', 'cache_eligible'),
        Index('idx_ticket_judgements_review_status', 'review_status'),
    )


class TicketTriage(Base):
    """
    Cheap model triage stage.
    Mirrors Scraper/db.py ticket_triage table.
    """
    __tablename__ = "ticket_triage"

    ticket_id = Column(String(255), ForeignKey("tickets_detail.ticket_id", ondelete="CASCADE"), primary_key=True)
    triage_label = Column(String(50), nullable=False)
    triage_confidence = Column(Float, nullable=False)
    triage_reason = Column(Text, nullable=True)
    triaged_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    triage_model = Column(String(255), nullable=False)
    triage_prompt_version = Column(String(255), nullable=False)
    triage_raw_response_json = Column(JSON, nullable=False)  # JSONB in Postgres

    __table_args__ = (
        Index('idx_ticket_triage_label', 'triage_label'),
    )


class TicketManualReview(Base):
    """
    Manual override layer.
    Mirrors Scraper/db.py ticket_manual_reviews table.
    """
    __tablename__ = "ticket_manual_reviews"

    ticket_id = Column(String(255), ForeignKey("ticket_judgements.ticket_id", ondelete="CASCADE"), primary_key=True)
    manual_status = Column(String(50), nullable=False)
    manual_reason = Column(Text, nullable=True)
    manual_confirmation_quote = Column(Text, nullable=True)
    reviewer = Column(String(255), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        CheckConstraint("manual_status IN ('approved', 'rejected')", name='check_manual_status'),
        Index('idx_ticket_manual_reviews_status', 'manual_status'),
    )


class TicketMachineModelMatch(Base):
    """
    Machine model matches (one row per match).
    Mirrors Scraper/scripts/backfill_ticket_machine_models.py ticket_machine_model_matches table.
    """
    __tablename__ = "ticket_machine_model_matches"

    ticket_id = Column(String(255), nullable=False)
    machine_model_id = Column(Integer, nullable=False)
    machine_model_name = Column(String(255), nullable=False)
    match_source = Column(String(50), nullable=False)
    score = Column(Integer, nullable=False)
    evidence_snippet = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        PrimaryKeyConstraint('ticket_id', 'machine_model_id', 'match_source'),
        Index('idx_ticket_machine_model_matches_ticket_id', 'ticket_id'),
    )


class TicketMachineModelAssignment(Base):
    """
    Machine model assignment summary (one row per ticket).
    Mirrors Scraper/scripts/backfill_ticket_machine_models.py ticket_machine_model_assignment table.
    """
    __tablename__ = "ticket_machine_model_assignment"

    ticket_id = Column(String(255), primary_key=True)
    machine_model_ids = Column(JSON, nullable=False)  # JSONB in Postgres, array of integers
    status = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    method = Column(String(255), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index('idx_ticket_machine_model_assignment_status', 'status'),
    )


class ScrapeRun(Base):
    """
    Background scrape job tracking.
    Mirrors Scraper/db.py scrape_runs table.
    """
    __tablename__ = "scrape_runs"

    run_id = Column(String(255), primary_key=True)
    status = Column(String(50), nullable=False)
    stage = Column(String(50), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(Text, nullable=True)
    summary_json = Column(JSON, nullable=True)  # JSONB in Postgres
    created_by = Column(String(255), nullable=True)

    __table_args__ = (
        CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'cancelled')", name='check_scrape_status'),
        CheckConstraint("stage IN ('indexing', 'building_details', 'judging') OR stage IS NULL", name='check_scrape_stage'),
        Index('idx_scrape_runs_status', 'status'),
        Index('idx_scrape_runs_started_at', 'started_at'),
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

