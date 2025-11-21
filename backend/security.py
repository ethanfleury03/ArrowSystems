from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt
from fastapi import HTTPException, Request, status

from .config.env import settings
from .config.auth import auth_config

# Use JWT secret from centralized settings (validated at startup)
JWT_SECRET_KEY = settings.JWT_SECRET_KEY
JWT_ALGORITHM = auth_config.JWT_ALGORITHM
DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES = auth_config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES


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


def get_jwt_from_request(request: Request) -> Optional[str]:
    """
    Extract JWT from request, checking both Authorization header and cookie.
    
    Priority:
    1. Authorization header (Bearer token)
    2. Cookie (access_token or configured cookie name)
    
    Args:
        request: FastAPI request object
        
    Returns:
        JWT token string if found, None otherwise
    """
    # Check Authorization header first
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.replace("Bearer ", "")
    
    # Check cookie
    cookie_name = auth_config.AUTH_COOKIE_NAME
    token = request.cookies.get(cookie_name)
    if token:
        return token
    
    return None


async def get_current_user_from_token(request: Request) -> Dict[str, Any]:
    """
    Dependency function to get current authenticated user from JWT.
    
    Extracts JWT from request (header or cookie), validates it, and returns
    the decoded payload. Does NOT fetch user from database - returns JWT claims only.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Dict containing JWT claims (email, role, etc.)
        
    Raises:
        HTTPException: 401 if token is missing, invalid, or expired
    """
    token = get_jwt_from_request(request)
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = decode_access_token(token)
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

