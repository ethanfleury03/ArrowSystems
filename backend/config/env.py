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
        # Environment detection
        self.ENV = os.getenv("ENV", "dev").lower()
        self.is_dev = self.ENV in {"dev", "development"}
        self.is_prod = self.ENV in {"prod", "production", "cloud"}
        
        # JWT Configuration
        self._load_jwt_secret()
        
        # CORS Configuration
        self._load_cors_origins()
    
    def _load_jwt_secret(self) -> None:
        """Load and validate JWT secret key."""
        env_secret = os.getenv("JWT_SECRET_KEY")
        
        if self.is_prod:
            # Production: require JWT_SECRET_KEY from environment
            if not env_secret:
                raise RuntimeError(
                    "JWT_SECRET_KEY environment variable is required in production. "
                    "Set ENV=prod and provide a secure JWT_SECRET_KEY."
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
            # Development: allow fallback to a dev-specific secret
            # This is different from the production default to avoid confusion
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


# Global settings instance
# This is initialized at module import time
settings = Settings()

