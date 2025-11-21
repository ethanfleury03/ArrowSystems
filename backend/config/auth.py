"""
Centralized authentication configuration for JWT-based auth with HTTP-only cookies.

This module provides a single source of truth for all authentication-related
settings including JWT configuration, cookie options, and token expiration.
"""

import os
from typing import Optional
from .env import settings


class AuthConfig:
    """
    Centralized authentication configuration.
    
    Manages JWT settings, cookie configuration, and token expiration times.
    All values can be overridden via environment variables.
    """
    
    def __init__(self):
        # JWT Configuration
        self.JWT_SECRET_KEY = settings.JWT_SECRET_KEY
        self.JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
        
        # Token Expiration
        self.JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(
            os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60")
        )
        self.JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(
            os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7")
        )
        
        # Cookie Configuration
        self.AUTH_COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "access_token")
        self.AUTH_COOKIE_DOMAIN = os.getenv("AUTH_COOKIE_DOMAIN", None)  # Optional
        
        # Cookie Security Attributes
        # In production, always use secure cookies
        # In development, allow HTTP for local testing
        self.AUTH_COOKIE_SECURE = self._get_cookie_secure()
        # SameSite=None is required for cross-origin cookies (frontend/backend on different domains)
        # This requires Secure=true (HTTPS only)
        self.AUTH_COOKIE_SAMESITE = os.getenv("AUTH_COOKIE_SAMESITE", "none" if settings.is_prod else "lax")
        self.AUTH_COOKIE_HTTPONLY = True  # Always HTTP-only for security
        self.AUTH_COOKIE_PATH = "/"
        
    def _get_cookie_secure(self) -> bool:
        """
        Determine if cookies should have the secure flag.
        
        Returns:
            bool: True for secure cookies (HTTPS only), False for HTTP
        """
        # Check explicit override first
        secure_env = os.getenv("AUTH_COOKIE_SECURE", "").lower()
        if secure_env in {"true", "1", "yes"}:
            return True
        if secure_env in {"false", "0", "no"}:
            return False
        
        # Default: secure in production, insecure in development
        return settings.is_prod
    
    def get_cookie_max_age(self) -> int:
        """
        Get cookie max age in seconds based on access token expiration.
        
        Returns:
            int: Cookie max age in seconds
        """
        return self.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
    
    def get_cookie_options(self) -> dict:
        """
        Get cookie options as a dictionary suitable for response.set_cookie().
        
        Returns:
            dict: Cookie configuration options
        """
        options = {
            "key": self.AUTH_COOKIE_NAME,
            "httponly": self.AUTH_COOKIE_HTTPONLY,
            "secure": self.AUTH_COOKIE_SECURE,
            "samesite": self.AUTH_COOKIE_SAMESITE,
            # SESSION COOKIE: No max_age means cookie expires when browser closes
            # Users must log in again each time they open the browser
            # "max_age": self.get_cookie_max_age(),  # Commented out for session-only cookies
            "path": self.AUTH_COOKIE_PATH,
        }
        
        # Only include domain if explicitly set
        if self.AUTH_COOKIE_DOMAIN:
            options["domain"] = self.AUTH_COOKIE_DOMAIN
        
        return options


# Global auth config instance
auth_config = AuthConfig()

