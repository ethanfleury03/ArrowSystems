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
    Base.metadata.create_all(bind=engine)
    
    # Ensure user columns exist
    ensure_user_columns()
    
    # Ensure analytics columns exist
    ensure_analytics_columns()


T = TypeVar("T")


async def run_sync(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    return await asyncio.to_thread(func, *args, **kwargs)

