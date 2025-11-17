from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt

from .config.env import settings

# Use JWT secret from centralized settings (validated at startup)
JWT_SECRET_KEY = settings.JWT_SECRET_KEY
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))


def create_access_token(
    claims: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT access token."""
    to_encode = claims.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT access token."""
    return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

