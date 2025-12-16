"""
Centralized environment configuration for dev/prod separation.

This module provides a single source of truth for environment detection
and configuration values that differ between development and production.
"""

import os
import logging
from typing import List, Optional

# Module-level logger - must be defined before Settings() instantiation
logger = logging.getLogger(__name__)


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
        # Use standard logger format since structlog may not be configured yet
        logger.info(f"Runtime environment detected: {self.ENV} (is_prod={self.is_prod}, is_dev={self.is_dev}, env_var={os.getenv('ENV')})")
        
        # Secret Configuration - REQUIRED in production
        self._load_secrets()
        
        # JWT Configuration
        self._load_jwt_secret()
        
        # CORS Configuration
        self._load_cors_origins()
        
        # Rate Limiting Configuration
        self._load_rate_limit_config()
        
        # Anthropic API Key (optional - for Claude LLM integration)
        self._load_anthropic_key()
        
        # Ingestion Safety Flag - prevents automatic ingestion from web app
        self._load_ingestion_config()
        
        # GCS Document Storage Configuration
        self._load_gcs_config()
    
    def _load_ingestion_config(self) -> None:
        """
        Load ingestion configuration flag.
        
        IMPORTANT: This flag controls whether the web app/API can trigger
        document ingestion (chunking, embedding, index writes).
        
        Default: False (ingestion disabled from app)
        Set ARROW_ALLOW_APP_INGESTION=true ONLY in dedicated GPU ingestion environments,
        NOT in the main production frontend/backend.
        """
        allow_ingestion_str = os.getenv("ARROW_ALLOW_APP_INGESTION", "false").lower()
        self.allow_app_ingestion = allow_ingestion_str in {"true", "1", "yes", "on"}
        
        if self.allow_app_ingestion:
            logger.warning(
                "ingestion_enabled_from_app",
                message="⚠️ WARNING: App-based ingestion is ENABLED. This should only be set in dedicated GPU ingestion environments."
            )
        else:
            logger.info(
                "ingestion_disabled_from_app",
                message="✅ App-based ingestion is DISABLED (default). Ingestion must be triggered via external GPU pipeline."
            )
    
    def _load_gcs_config(self) -> None:
        """
        Load Google Cloud Storage configuration for document storage.
        
        Required:
        - DOCS_GCS_BUCKET: GCS bucket name for storing documents
        
        Optional:
        - DOCS_GCS_PREFIX: Prefix/path within bucket (default: "documents/")
        - DOCS_LOCAL_SAVE_ENABLED: Whether to also save files locally (default: false)
        """
        # Required: GCS bucket name
        self.DOCS_GCS_BUCKET = os.getenv("DOCS_GCS_BUCKET")
        if not self.DOCS_GCS_BUCKET:
            if self.is_prod:
                raise RuntimeError(
                    "DOCS_GCS_BUCKET environment variable is REQUIRED in production. "
                    "Set it to the GCS bucket name where documents should be stored."
                )
            else:
                logger.warning(
                    "gcs_bucket_not_set",
                    message="⚠️ DOCS_GCS_BUCKET not set. Document uploads will fail unless configured."
                )
        
        # Optional: GCS prefix (default: "documents/")
        self.DOCS_GCS_PREFIX = os.getenv("DOCS_GCS_PREFIX", "documents/").rstrip("/")
        if not self.DOCS_GCS_PREFIX.endswith("/"):
            self.DOCS_GCS_PREFIX += "/"
        
        # Optional: Local save fallback (default: false)
        local_save_str = os.getenv("DOCS_LOCAL_SAVE_ENABLED", "false").lower()
        self.DOCS_LOCAL_SAVE_ENABLED = local_save_str in {"true", "1", "yes", "on"}
        
        if self.DOCS_GCS_BUCKET:
            logger.info(
                "gcs_config_loaded",
                bucket=self.DOCS_GCS_BUCKET,
                prefix=self.DOCS_GCS_PREFIX,
                local_save_enabled=self.DOCS_LOCAL_SAVE_ENABLED,
                message=f"GCS document storage configured: gs://{self.DOCS_GCS_BUCKET}/{self.DOCS_GCS_PREFIX}"
            )
    
    def _load_anthropic_key(self) -> None:
        """Load Anthropic API key - optional, used for Claude LLM integration."""
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.anthropic_api_key:
            self.anthropic_api_key = self.anthropic_api_key.strip().rstrip('\r\n')
            print("[SETTINGS] ANTHROPIC_API_KEY detected in environment.", flush=True)
        else:
            print("[SETTINGS] ANTHROPIC_API_KEY not set; Anthropic LLM disabled.", flush=True)
    
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

