"""
Integration tests for health check endpoint.
"""

import pytest
from fastapi.testclient import TestClient
from backend.api import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    # Note: This will use the actual app, which may try to initialize
    # RAG pipeline and database. For a minimal smoke test, we'll just
    # check that the endpoint exists and returns a response.
    return TestClient(app)


def test_health_endpoint_exists(client):
    """Test that /health endpoint exists and returns HTTP 200."""
    response = client.get("/health")
    
    # Should return 200 (even if unhealthy, endpoint should respond)
    assert response.status_code in [200, 503]  # 503 if services not initialized
    
    # Should return JSON
    assert response.headers["content-type"] == "application/json"


def test_health_endpoint_returns_json_structure(client):
    """Test that /health endpoint returns expected JSON structure."""
    response = client.get("/health")
    
    # Should return JSON
    data = response.json()
    
    # Should have status field
    assert "status" in data
    assert data["status"] in ["healthy", "unhealthy"]
    
    # Should have basic health check fields
    assert "rag_pipeline_initialized" in data
    assert "database_connected" in data
    assert isinstance(data["rag_pipeline_initialized"], bool)
    assert isinstance(data["database_connected"], bool)


def test_health_endpoint_has_uptime(client):
    """Test that /health endpoint includes uptime information."""
    response = client.get("/health")
    data = response.json()
    
    # Should have uptime_seconds field
    assert "uptime_seconds" in data
    assert isinstance(data["uptime_seconds"], (int, float))
    assert data["uptime_seconds"] >= 0

