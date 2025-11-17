"""
Unit tests for security functions (password hashing, JWT tokens).
"""

import bcrypt
import pytest
from datetime import timedelta
from backend.security import create_access_token, decode_access_token, JWT_SECRET_KEY


def test_password_hashing_workflow():
    """Test that password hashing and verification works correctly."""
    # Test password
    password = "test_password_123"
    
    # Hash the password
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    
    # Verify the hash is different from original
    assert hashed != password
    assert len(hashed) > 0
    
    # Verify password matches hash
    assert bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8")) is True
    
    # Verify wrong password doesn't match
    wrong_password = "wrong_password"
    assert bcrypt.checkpw(wrong_password.encode("utf-8"), hashed.encode("utf-8")) is False


def test_password_hashing_produces_different_hashes():
    """Test that hashing the same password produces different hashes (due to salt)."""
    password = "same_password"
    
    # Hash twice
    hashed1 = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    hashed2 = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    
    # Hashes should be different (due to random salt)
    assert hashed1 != hashed2
    
    # But both should verify correctly
    assert bcrypt.checkpw(password.encode("utf-8"), hashed1.encode("utf-8")) is True
    assert bcrypt.checkpw(password.encode("utf-8"), hashed2.encode("utf-8")) is True


def test_jwt_token_creation_and_verification():
    """Test that JWT tokens can be created and decoded."""
    # Create token with test claims
    claims = {
        "email": "test@example.com",
        "role": "ADMIN",
        "user_id": 123
    }
    
    token = create_access_token(claims)
    
    # Verify token is a string
    assert isinstance(token, str)
    assert len(token) > 0
    
    # Decode token
    decoded = decode_access_token(token)
    
    # Verify claims are present
    assert decoded["email"] == "test@example.com"
    assert decoded["role"] == "ADMIN"
    assert decoded["user_id"] == 123
    assert "exp" in decoded  # Expiration should be set


def test_jwt_token_expiration():
    """Test that JWT tokens have expiration set."""
    claims = {"email": "test@example.com"}
    
    # Create token with custom expiration
    token = create_access_token(claims, expires_delta=timedelta(minutes=30))
    
    decoded = decode_access_token(token)
    
    # Verify expiration is set
    assert "exp" in decoded
    assert decoded["exp"] > 0

