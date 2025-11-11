from __future__ import annotations

import asyncio
import os
from datetime import datetime
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
)
from sqlalchemy.orm import Session, declarative_base, relationship, scoped_session, sessionmaker

DEFAULT_DB_PATH = os.getenv("SQLITE_DB_PATH", "./database.sqlite")
DATABASE_URL = f"sqlite:///{DEFAULT_DB_PATH.lstrip('./') if DEFAULT_DB_PATH.startswith('./') else DEFAULT_DB_PATH}"

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


def init_db() -> None:
    os.makedirs(os.path.dirname(DEFAULT_DB_PATH) or ".", exist_ok=True)
    Base.metadata.create_all(bind=engine)


T = TypeVar("T")


async def run_sync(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    return await asyncio.to_thread(func, *args, **kwargs)

