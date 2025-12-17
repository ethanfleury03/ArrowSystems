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

        # RAG Index Storage Configuration (GCS source + local target dir)
        self._load_rag_index_config()
    
    def _load_ingestion_config(self) -> None:
        """
        Load ingestion configuration flags.
        
        DEPRECATED: ARROW_ALLOW_APP_INGESTION is no longer used for gating single-document operations.
        Single-document ingestion (upload -> chunking -> embedding) is always allowed.
        
        NEW: ARROW_ENABLE_BULK_INGEST_ENDPOINTS controls bulk ingestion endpoints.
        - Default: False (bulk endpoints disabled)
        - Set to true ONLY if you need to expose bulk rebuild/ingest endpoints via API
        - Full ingestion should normally be done via CLI: python ingest.py
        """
        # DEPRECATED: Keep for backward compatibility but don't use for gating
        allow_ingestion_str = os.getenv("ARROW_ALLOW_APP_INGESTION", "false").lower().strip()
        self.allow_app_ingestion = allow_ingestion_str in {"true", "1", "yes", "on"}
        
        # NEW: Bulk ingestion endpoints flag
        bulk_endpoints_str = os.getenv("ARROW_ENABLE_BULK_INGEST_ENDPOINTS", "false").lower().strip()
        self.enable_bulk_ingest_endpoints = bulk_endpoints_str in {"true", "1", "yes", "on"}
        
        # Log configuration at startup
        logger.info(
            f"Ingestion configuration: "
            f"ARROW_ALLOW_APP_INGESTION={self.allow_app_ingestion} (DEPRECATED - not used for gating), "
            f"ARROW_ENABLE_BULK_INGEST_ENDPOINTS={self.enable_bulk_ingest_endpoints}"
        )
        
        if self.enable_bulk_ingest_endpoints:
            logger.warning("⚠️ WARNING: Bulk ingestion endpoints are ENABLED. Full index rebuilds are available via API.")
        else:
            logger.info("✅ Bulk ingestion endpoints are DISABLED (default). Full ingestion must be done via CLI: python ingest.py")
    
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
                logger.warning("⚠️ DOCS_GCS_BUCKET not set. Document uploads will fail unless configured.")
        
        # Optional: GCS prefix (default: "documents/")
        self.DOCS_GCS_PREFIX = os.getenv("DOCS_GCS_PREFIX", "documents/").rstrip("/")
        if not self.DOCS_GCS_PREFIX.endswith("/"):
            self.DOCS_GCS_PREFIX += "/"
        
        # Optional: Local save fallback (default: false)
        local_save_str = os.getenv("DOCS_LOCAL_SAVE_ENABLED", "false").lower()
        self.DOCS_LOCAL_SAVE_ENABLED = local_save_str in {"true", "1", "yes", "on"}
        
        if self.DOCS_GCS_BUCKET:
            logger.info(f"GCS document storage configured: gs://{self.DOCS_GCS_BUCKET}/{self.DOCS_GCS_PREFIX} (local_save={self.DOCS_LOCAL_SAVE_ENABLED})")
            
            # Log GCS authentication info if available
            try:
                from backend.utils.gcs_client import _get_auth_info, _is_cloud_run
                auth_info = _get_auth_info()
                is_cloud_run = _is_cloud_run()
                
                if is_cloud_run:
                    logger.info(
                        f"GCS authentication: Using Cloud Run service account identity "
                        f"(project: {auth_info.get('project', 'unknown')}, "
                        f"service account: {auth_info.get('service_account_email', 'unknown')})"
                    )
                else:
                    logger.info(
                        f"GCS authentication: Using Application Default Credentials "
                        f"(project: {auth_info.get('project', 'unknown')}, "
                        f"has_goog_app_creds: {auth_info.get('has_goog_app_creds', False)})"
                    )
            except Exception as e:
                logger.debug(f"Could not log GCS auth info: {e}")

    def _load_rag_index_config(self) -> None:
        """
        Load configuration for the RAG index artifacts.

        - Source of truth is GCS (bucket + prefix).
        - On Cloud Run, the index is downloaded at startup into a writable local directory.

        Env vars:
        - RAG_INDEX_GCS_BUCKET: GCS bucket containing index artifacts (default: arrow-rag-support-prod-rag)
        - RAG_INDEX_GCS_PREFIX: GCS prefix containing index artifacts (default: latest_model/)
          - Can be "" to indicate bucket root.
        - RAG_INDEX_LOCAL_DIR: Local directory where index will be downloaded/loaded from.
          - Default: /tmp/latest_model on Cloud Run / prod, latest_model in dev.
        """
        is_cloud_run = bool(os.getenv("K_SERVICE") or os.getenv("K_REVISION"))

        self.RAG_INDEX_GCS_BUCKET = os.getenv("RAG_INDEX_GCS_BUCKET", "arrow-rag-support-prod-rag").strip()

        raw_prefix = os.getenv("RAG_INDEX_GCS_PREFIX", "latest_model/")
        raw_prefix = (raw_prefix or "").strip()
        if raw_prefix:
            # Normalize to "<prefix>/" with no leading slash
            normalized = raw_prefix.strip("/")
            self.RAG_INDEX_GCS_PREFIX = f"{normalized}/" if normalized else ""
        else:
            self.RAG_INDEX_GCS_PREFIX = ""

        default_local_dir = "/tmp/latest_model" if (self.is_prod or is_cloud_run) else "latest_model"
        local_dir = os.getenv("RAG_INDEX_LOCAL_DIR", default_local_dir)
        local_dir = (local_dir or "").strip() or default_local_dir
        self.RAG_INDEX_LOCAL_DIR = local_dir

        # Loud startup log to confirm configuration
        gcs_location = f"gs://{self.RAG_INDEX_GCS_BUCKET}/{self.RAG_INDEX_GCS_PREFIX}" if self.RAG_INDEX_GCS_PREFIX else f"gs://{self.RAG_INDEX_GCS_BUCKET}/"
        logger.info(
            f"RAG index configured: source={gcs_location} local_dir={self.RAG_INDEX_LOCAL_DIR} "
            f"(env={self.ENV}, cloud_run={is_cloud_run})"
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

