"""
Unit tests for backend configuration (Settings class).
"""

import os
import pytest
from backend.config.env import Settings


def test_settings_loads_with_default_env():
    """Test that Settings loads with default dev environment."""
    # Clear ENV to test default
    original_env = os.environ.get("ENV")
    if "ENV" in os.environ:
        del os.environ["ENV"]
    
    # Reload module to get fresh Settings instance
    import importlib
    import backend.config.env
    importlib.reload(backend.config.env)
    
    settings = backend.config.env.Settings()
    
    # Should default to dev
    assert settings.ENV == "dev"
    assert settings.is_dev is True
    assert settings.is_prod is False
    
    # Restore original ENV
    if original_env:
        os.environ["ENV"] = original_env
    elif "ENV" in os.environ:
        del os.environ["ENV"]


def test_settings_respects_env_variable(monkeypatch):
    """Test that Settings respects ENV environment variable."""
    # Set ENV to dev
    monkeypatch.setenv("ENV", "dev")
    # Clear JWT_SECRET_KEY to test dev defaults
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    
    # Reload module to get fresh Settings instance
    import importlib
    import backend.config.env
    importlib.reload(backend.config.env)
    
    settings = backend.config.env.Settings()
    
    assert settings.ENV == "dev"
    assert settings.is_dev is True
    assert settings.is_prod is False
    
    # Set ENV to prod (requires JWT_SECRET_KEY)
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-key-for-testing-only-at-least-32-chars")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000")
    importlib.reload(backend.config.env)
    
    settings = backend.config.env.Settings()
    
    assert settings.ENV == "prod"
    assert settings.is_dev is False
    assert settings.is_prod is True


def test_settings_dev_defaults():
    """Test that dev mode has correct default values."""
    import importlib
    import backend.config.env
    
    # Ensure we're in dev mode
    original_env = os.environ.get("ENV")
    os.environ["ENV"] = "dev"
    if "JWT_SECRET_KEY" in os.environ:
        del os.environ["JWT_SECRET_KEY"]
    if "CORS_ALLOWED_ORIGINS" in os.environ:
        del os.environ["CORS_ALLOWED_ORIGINS"]
    
    importlib.reload(backend.config.env)
    settings = backend.config.env.Settings()
    
    # Should have dev defaults
    assert settings.JWT_SECRET_KEY == "dev-secret-key-not-for-production-use-only"
    assert "http://localhost:3000" in settings.CORS_ALLOWED_ORIGINS
    
    # Restore
    if original_env:
        os.environ["ENV"] = original_env
    elif "ENV" in os.environ:
        del os.environ["ENV"]

