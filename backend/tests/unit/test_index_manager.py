"""
Unit tests for RAG index manager.

Tests the IndexLoadState singleton and /api/readyz endpoint behavior.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock
from backend.rag.index_manager import IndexLoadState, get_index_load_state


def test_index_load_state_singleton():
    """Test that IndexLoadState is a singleton."""
    state1 = IndexLoadState()
    state2 = IndexLoadState()
    assert state1 is state2
    
    # get_index_load_state should return the same instance
    state3 = get_index_load_state()
    assert state1 is state3


def test_index_load_state_initial_state():
    """Test initial state of IndexLoadState."""
    state = IndexLoadState()
    assert state.status == "not_started"
    assert state.error is None
    assert state.started_at is None
    assert state.finished_at is None
    
    state_dict = state.get_state()
    assert state_dict["status"] == "not_started"
    assert state_dict["error"] is None


@pytest.mark.asyncio
async def test_index_load_state_wait_for_ready_not_started():
    """Test wait_for_ready when status is not_started."""
    state = IndexLoadState()
    state._status = "not_started"
    state._ready_event.set()  # Set event so wait doesn't block
    
    # Should return False immediately since not ready
    result = await state.wait_for_ready(timeout=0.1)
    assert result is False


@pytest.mark.asyncio
async def test_index_load_state_wait_for_ready_already_ready():
    """Test wait_for_ready when already ready."""
    state = IndexLoadState()
    state._status = "ready"
    state._ready_event.set()
    
    result = await state.wait_for_ready(timeout=0.1)
    assert result is True


@pytest.mark.asyncio
async def test_index_load_state_wait_timeout():
    """Test wait_for_ready with timeout."""
    state = IndexLoadState()
    state._status = "loading"
    state._ready_event.clear()
    
    # Should timeout and return False
    result = await state.wait_for_ready(timeout=0.1)
    assert result is False


@pytest.mark.asyncio
async def test_ensure_loaded_mock_success():
    """Test ensure_loaded with mocked successful download and load."""
    state = IndexLoadState()
    
    # Reset state
    state._status = "not_started"
    state._error = None
    state._ready_event.clear()
    
    with patch('backend.rag.index_manager.settings') as mock_settings, \
         patch('backend.rag.index_manager.resolve_storage_path') as mock_resolve, \
         patch('backend.rag.index_manager.is_test_mode', return_value=False), \
         patch('backend.rag.index_manager.download_index_from_gcs', return_value=True), \
         patch('backend.rag.index_manager.get_rag_pipeline') as mock_get_pipeline, \
         patch('backend.rag.index_manager.get_db_manager_instance', return_value=None):
        
        # Setup mocks
        mock_settings.is_prod = True
        mock_settings.RAG_INDEX_GCS_BUCKET = "test-bucket"
        mock_settings.RAG_INDEX_GCS_PREFIX = "latest_model/"
        mock_settings.RAG_INDEX_LOCAL_DIR = "/tmp/test"
        
        mock_path = MagicMock()
        mock_path.resolve.return_value = Path("/tmp/test")
        mock_resolve.return_value = mock_path
        
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.ensure_initialized.return_value = True
        mock_pipeline.is_initialized.return_value = True
        mock_pipeline.debug_status.return_value = {"last_error": None}
        mock_get_pipeline.return_value = mock_pipeline
        
        # Mock file existence check (files exist)
        with patch('os.path.exists', return_value=True):
            await state.ensure_loaded()
        
        assert state.status == "ready"
        assert state.error is None
        assert state.started_at is not None
        assert state.finished_at is not None


@pytest.mark.asyncio
async def test_ensure_loaded_mock_failure():
    """Test ensure_loaded with mocked failure."""
    state = IndexLoadState()
    
    # Reset state
    state._status = "not_started"
    state._error = None
    state._ready_event.clear()
    
    with patch('backend.rag.index_manager.settings') as mock_settings, \
         patch('backend.rag.index_manager.resolve_storage_path') as mock_resolve, \
         patch('backend.rag.index_manager.is_test_mode', return_value=False), \
         patch('backend.rag.index_manager.download_index_from_gcs', return_value=False), \
         patch('backend.rag.index_manager.get_last_download_error', return_value="Download failed"):
        
        # Setup mocks
        mock_settings.is_prod = True
        mock_settings.RAG_INDEX_GCS_BUCKET = "test-bucket"
        mock_settings.RAG_INDEX_GCS_PREFIX = "latest_model/"
        mock_settings.RAG_INDEX_LOCAL_DIR = "/tmp/test"
        
        mock_path = MagicMock()
        mock_path.resolve.return_value = Path("/tmp/test")
        mock_resolve.return_value = mock_path
        
        # Mock file existence check (files missing)
        with patch('os.path.exists', return_value=False):
            with pytest.raises(RuntimeError, match="Index download failed"):
                await state.ensure_loaded()
        
        assert state.status == "failed"
        assert state.error is not None
        assert "Download failed" in state.error

