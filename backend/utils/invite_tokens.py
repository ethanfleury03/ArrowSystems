"""Utilities for managing invite tokens for password setup flow."""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from .db import AuthToken, User

INVITE_TOKEN_TTL_DAYS = 7


def _hash_token(raw_token: str) -> str:
    """Hash a raw token using SHA-256."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def create_invite_token(db: Session, user: User, purpose: str = "invite") -> str:
    """
    Create a new invite token for a user.
    
    Single-use semantics: marks any existing unused invite tokens for this user as used
    before creating a new one.
    
    Args:
        db: Database session
        user: User instance to create token for
        purpose: Token purpose (default: "invite", can be "reset" for future use)
        
    Returns:
        Raw token string (to be sent to user, not stored in DB)
    """
    # Mark existing unused invite tokens for this user as used
    db.query(AuthToken).filter(
        AuthToken.user_id == user.id,
        AuthToken.purpose == purpose,
        AuthToken.used.is_(False),
    ).update({AuthToken.used: True})
    db.commit()

    # Generate new token
    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash_token(raw_token)
    expires_at = datetime.now(timezone.utc) + timedelta(days=INVITE_TOKEN_TTL_DAYS)

    auth_token = AuthToken(
        user_id=user.id,
        token_hash=token_hash,
        purpose=purpose,
        expires_at=expires_at,
        used=False,
    )
    db.add(auth_token)
    db.commit()
    db.refresh(auth_token)

    return raw_token


def validate_invite_token(db: Session, raw_token: str, purpose: str = "invite") -> Optional[User]:
    """
    Validate an invite token and return the associated user if valid.
    
    Args:
        db: Database session
        raw_token: Raw token string from the invite link
        purpose: Token purpose (default: "invite")
        
    Returns:
        User instance if token is valid, None otherwise
    """
    token_hash = _hash_token(raw_token)
    now = datetime.now(timezone.utc)

    result = (
        db.query(AuthToken, User)
        .filter(
            AuthToken.token_hash == token_hash,
            AuthToken.purpose == purpose,
            AuthToken.used.is_(False),
            AuthToken.expires_at > now,
        )
        .join(User, AuthToken.user_id == User.id)
        .first()
    )

    if not result:
        return None

    auth_token, user = result
    return user


def mark_invite_token_used(db: Session, raw_token: str, purpose: str = "invite") -> None:
    """
    Mark an invite token as used.
    
    Args:
        db: Database session
        raw_token: Raw token string
        purpose: Token purpose (default: "invite")
    """
    token_hash = _hash_token(raw_token)
    token = (
        db.query(AuthToken)
        .filter(
            AuthToken.token_hash == token_hash,
            AuthToken.purpose == purpose,
            AuthToken.used.is_(False),
        )
        .first()
    )
    if token:
        token.used = True
        db.add(token)
        db.commit()

