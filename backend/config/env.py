"""
Centralized environment configuration for dev/prod separation.

This module provides a single source of truth for environment detection
and configuration values that differ between development and production.
"""

import os
from typing import List, Optional


class Settings:
    """
    Centralized application settings based on environment.
    
    Usage:
        from backend.config.env import settings
        
        if settings.is_dev:
            # dev-only code
        if settings.is_prod:
            # prod-only code
    """
    
    def __init__(self):
        # Environment detection - ALWAYS prioritize environment variables over .env files
        # This ensures Cloud Run's ENV=prod always wins, even if .env file exists
        self.ENV = os.getenv("ENV", "dev").lower()
        self.is_dev = self.ENV in {"dev", "development"}
        self.is_prod = self.ENV in {"prod", "production", "cloud"}
        
        # Log the effective ENV value for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info("env_runtime_value", 
                   env=self.ENV,
                   env_var_value=os.getenv("ENV"),
                   is_prod=self.is_prod,
                   is_dev=self.is_dev,
                   message=f"Runtime environment detected: {self.ENV} (is_prod={self.is_prod}, is_dev={self.is_dev})")
        
        # Secret Configuration - REQUIRED in production
        self._load_secrets()
        
        # JWT Configuration
        self._load_jwt_secret()
        
        # CORS Configuration
        self._load_cors_origins()
        
        # Rate Limiting Configuration
        self._load_rate_limit_config()
    
    def _load_secrets(self) -> None:
        """Load required secrets - REQUIRED in production, optional in dev."""
        # DATABASE_URL - required in all environments
        if self.is_prod:
            try:
                self.DATABASE_URL = os.environ["DATABASE_URL"]
            except KeyError:
                raise RuntimeError(
                    "DATABASE_URL environment variable is REQUIRED in production but not set. "
                    "Ensure Cloud Run is configured to load this from Google Secret Manager."
                )
        else:
            # Development: allow fallback via os.getenv() for local .env file support
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                raise RuntimeError(
                    "DATABASE_URL environment variable is required in all environments. "
                    "Set it in your .env file for local development."
                )
            self.DATABASE_URL = database_url
        
        # FRONTEND_SESSION_SECRET - required in production
        if self.is_prod:
            try:
                self.FRONTEND_SESSION_SECRET = os.environ["FRONTEND_SESSION_SECRET"]
            except KeyError:
                raise RuntimeError(
                    "FRONTEND_SESSION_SECRET environment variable is REQUIRED in production but not set. "
                    "Ensure Cloud Run is configured to load this from Google Secret Manager."
                )
            
            # Validate secret is not empty
            if not self.FRONTEND_SESSION_SECRET or self.FRONTEND_SESSION_SECRET.strip() == "":
                raise RuntimeError(
                    "FRONTEND_SESSION_SECRET is set but empty. Provide a valid secret via Google Secret Manager."
                )
        else:
            # Development: allow fallback via os.getenv() for local .env file support
            self.FRONTEND_SESSION_SECRET = os.getenv("FRONTEND_SESSION_SECRET", "dev-session-secret-not-for-production")
    
    def _load_jwt_secret(self) -> None:
        """Load and validate JWT secret key - REQUIRED in production."""
        if self.is_prod:
            # Production: JWT_SECRET_KEY MUST be set explicitly via Secret Manager
            # Cloud Run injects Secret Manager values as environment variables
            # Use os.environ[] to fail fast if missing
            try:
                env_secret = os.environ["JWT_SECRET_KEY"]
            except KeyError:
                raise RuntimeError(
                    "JWT_SECRET_KEY environment variable is REQUIRED in production but not set. "
                    "Ensure Cloud Run is configured to load this from Google Secret Manager. "
                    "Generate a secure secret with: python -c 'import secrets; print(secrets.token_urlsafe(64))'"
                )
            
            # Validate secret is not empty
            if not env_secret or env_secret.strip() == "":
                raise RuntimeError(
                    "JWT_SECRET_KEY is set but empty. Provide a valid secret via Google Secret Manager."
                )
            
            # Check for unsafe defaults
            unsafe_defaults = [
                "change-this-secret",
                "secret",
                "password",
                "default-secret",
            ]
            if env_secret in unsafe_defaults or len(env_secret) < 32:
                raise RuntimeError(
                    f"JWT_SECRET_KEY is set to an unsafe default or is too short. "
                    f"In production, JWT_SECRET_KEY must be at least 32 characters "
                    f"and not be a common default value."
                )
            self.JWT_SECRET_KEY = env_secret
        else:
            # Development: allow fallback via os.getenv() for local .env file support
            env_secret = os.getenv("JWT_SECRET_KEY")
            self.JWT_SECRET_KEY = env_secret or "dev-secret-key-not-for-production-use-only"
    
    def _load_cors_origins(self) -> None:
        """Load and validate CORS allowed origins."""
        env_origins = os.getenv("CORS_ALLOWED_ORIGINS")
        
        if self.is_prod:
            # Production: require CORS_ALLOWED_ORIGINS
            if not env_origins:
                raise RuntimeError(
                    "CORS_ALLOWED_ORIGINS environment variable is required in production. "
                    "Set ENV=prod and provide a comma-separated list of allowed origins, "
                    "e.g., 'https://example.com,https://www.example.com'"
                )
            
            # Parse comma-separated string into list
            self.CORS_ALLOWED_ORIGINS = [
                origin.strip() for origin in env_origins.split(",") if origin.strip()
            ]
            
            # Validate no wildcard in production
            if "*" in self.CORS_ALLOWED_ORIGINS:
                raise RuntimeError(
                    "CORS_ALLOWED_ORIGINS cannot contain '*' in production. "
                    "Provide specific allowed origins."
                )
        else:
            # Development: default to localhost origins if not provided
            if env_origins:
                # Parse if provided
                self.CORS_ALLOWED_ORIGINS = [
                    origin.strip() for origin in env_origins.split(",") if origin.strip()
                ]
            else:
                # Default dev origins
                self.CORS_ALLOWED_ORIGINS = [
                    "http://localhost:3000",
                    "http://127.0.0.1:3000",
                ]
    
    def _load_rate_limit_config(self) -> None:
        """Load and validate rate limiting configuration."""
        # Rate limiting enabled flag
        rate_limit_enabled_str = os.getenv("RATE_LIMIT_ENABLED", "true").lower()
        self.RATE_LIMIT_ENABLED = rate_limit_enabled_str in {"true", "1", "yes", "on"}
        
        # Global rate limit (applies to all endpoints unless overridden)
        # Format: "number/period" (e.g., "100/minute", "20/second")
        self.RATE_LIMIT_GLOBAL = os.getenv("RATE_LIMIT_GLOBAL", "100/minute")
        
        # Per-endpoint rate limits
        # Login endpoint: stricter limit to prevent brute force attacks
        self.RATE_LIMIT_LOGIN = os.getenv("RATE_LIMIT_LOGIN", "5/minute")
        
        # Query endpoint: moderate limit to prevent abuse
        self.RATE_LIMIT_QUERY = os.getenv("RATE_LIMIT_QUERY", "10/minute")


# Global settings instance
# This is initialized at module import time
settings = Settings()

