"""
Tests for the embedding runner (Phase 3).

Tests ensure:
- Successful embedding transitions: READY_FOR_EMBEDDING → EMBEDDING → COMPLETE
- Summarization fallback works when summarizer fails
- Vector index persistence after embedding
- Failure mode: Embedding raises exception → ingestion = FAILED
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from sqlalchemy.orm import Session
from llama_index.core import VectorStoreIndex, StorageContext
from backend.utils.db import SessionLocal, DocumentIngestionMetadata, Base, engine
from backend.utils.embedding_runner import run_embedding


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def chunks_file(temp_dir):
    """Create a sample chunks JSON file for testing."""
    chunks_dir = Path(temp_dir) / "data" / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    chunks_data = {
        "metadata_id": "test-metadata-id",
        "filename": "test.pdf",
        "machine_model": "TestMachine",
        "machine_model_ids": [1, 2],
        "machine_model_names": ["DuraFlex", "DuraCore"],
        "created_at": "2024-01-01T00:00:00",
        "chunks": [
            {
                "text": "This is a test chunk with some content.",
                "metadata": {
                    "file_name": "test.pdf",
                    "page_label": "1",
                    "chunk_index": 0,
                    "machine_model_ids": [1, 2],
                    "machine_model_names": ["DuraFlex", "DuraCore"],
                    "machine_model": ["DuraFlex", "DuraCore"],
                },
                "node_id": None,
            },
            {
                "text": "Another test chunk with different content.",
                "metadata": {
                    "file_name": "test.pdf",
                    "page_label": "1",
                    "chunk_index": 1,
                    "machine_model_ids": [1, 2],
                    "machine_model_names": ["DuraFlex", "DuraCore"],
                    "machine_model": ["DuraFlex", "DuraCore"],
                },
                "node_id": None,
            }
        ]
    }
    
    chunks_file = chunks_dir / "test-metadata-id.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f)
    
    return chunks_file


@pytest.fixture
def db_session():
    """Create a test database session."""
    Base.metadata.create_all(engine)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def sample_metadata(db_session):
    """Create a sample DocumentIngestionMetadata record."""
    import uuid
    metadata = DocumentIngestionMetadata(
        id="test-metadata-id",
        filename="test.pdf",
        machine_model="TestMachine",
        status="READY_FOR_EMBEDDING",
        file_path="/tmp/test.pdf",
        file_size_bytes=1024,
    )
    db_session.add(metadata)
    db_session.commit()
    db_session.refresh(metadata)
    return metadata


def test_embedding_success_transition(sample_metadata, chunks_file, temp_dir):
    """Test that successful embedding transitions through correct statuses."""
    metadata_id = sample_metadata.id
    
    # Mock the index loading and insertion
    with patch('backend.utils.embedding_runner.load_index_from_storage') as mock_load, \
         patch('backend.utils.embedding_runner.VectorStoreIndex') as mock_index_class, \
         patch('backend.utils.embedding_runner.QuerySummarizer') as mock_summarizer_class, \
         patch('backend.utils.embedding_runner.Settings') as mock_settings, \
         patch('backend.utils.embedding_runner.HuggingFaceEmbedding') as mock_embed, \
         patch('os.path.exists', return_value=True), \
         patch('os.makedirs'):
        
        # Setup mocks
        mock_index = Mock()
        mock_index.insert_nodes = Mock()
        mock_index.storage_context = Mock()
        mock_index.storage_context.persist = Mock()
        mock_load.return_value = mock_index
        
        # Mock summarizer
        mock_summarizer = Mock()
        mock_summarizer.summarize = Mock(return_value=("Summary", True, None))
        mock_summarizer_class.return_value = mock_summarizer
        
        # Mock Settings
        mock_settings.embed_model = None
        
        # Change to temp directory for chunks file
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            run_embedding(metadata_id)
        finally:
            os.chdir(original_cwd)

        # Verify we inserted nodes with propagated machine_model_ids
        assert mock_index.insert_nodes.called
        inserted_nodes = []
        for call in mock_index.insert_nodes.call_args_list:
            batch = call.args[0] if call.args else []
            inserted_nodes.extend(batch)
        assert len(inserted_nodes) > 0
        for n in inserted_nodes:
            md = getattr(n, "metadata", {}) or {}
            assert md.get("machine_model_ids") == [1, 2]
        
        # Verify status transition
        session = SessionLocal()
        try:
            updated_metadata = session.query(DocumentIngestionMetadata).filter(
                DocumentIngestionMetadata.id == metadata_id
            ).first()
            assert updated_metadata is not None
            assert updated_metadata.status == "COMPLETE"
        finally:
            session.close()


def test_embedding_summarization_fallback(sample_metadata, chunks_file, temp_dir):
    """Test that summarization fallback works when summarizer fails."""
    metadata_id = sample_metadata.id
    
    with patch('backend.utils.embedding_runner.load_index_from_storage') as mock_load, \
         patch('backend.utils.embedding_runner.VectorStoreIndex') as mock_index_class, \
         patch('backend.utils.embedding_runner.QuerySummarizer') as mock_summarizer_class, \
         patch('backend.utils.embedding_runner.Settings') as mock_settings, \
         patch('backend.utils.embedding_runner.HuggingFaceEmbedding') as mock_embed, \
         patch('os.path.exists', return_value=True), \
         patch('os.makedirs'):
        
        # Setup mocks
        mock_index = Mock()
        mock_index.insert_nodes = Mock()
        mock_index.storage_context = Mock()
        mock_index.storage_context.persist = Mock()
        mock_load.return_value = mock_index
        
        # Mock summarizer to raise exception (simulating failure)
        mock_summarizer = Mock()
        mock_summarizer.summarize = Mock(side_effect=Exception("Summarizer failed"))
        mock_summarizer_class.return_value = mock_summarizer
        
        mock_settings.embed_model = None
        
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            run_embedding(metadata_id)
        finally:
            os.chdir(original_cwd)
        
        # Verify that fallback summary was used (first 200 chars)
        # The function should still complete successfully
        session = SessionLocal()
        try:
            updated_metadata = session.query(DocumentIngestionMetadata).filter(
                DocumentIngestionMetadata.id == metadata_id
            ).first()
            assert updated_metadata is not None
            # Should still complete (fallback works)
            assert updated_metadata.status == "COMPLETE"
        finally:
            session.close()


def test_embedding_failure_transition(sample_metadata, temp_dir):
    """Test that failures during embedding transition to FAILED status."""
    metadata_id = sample_metadata.id
    
    # Mock chunks file not found
    with patch('os.path.exists', return_value=False):
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            run_embedding(metadata_id)
        finally:
            os.chdir(original_cwd)
        
        # Verify status is FAILED
        session = SessionLocal()
        try:
            updated_metadata = session.query(DocumentIngestionMetadata).filter(
                DocumentIngestionMetadata.id == metadata_id
            ).first()
            assert updated_metadata is not None
            assert updated_metadata.status == "FAILED"
            assert updated_metadata.error_message is not None
        finally:
            session.close()


def test_embedding_index_persistence(sample_metadata, chunks_file, temp_dir):
    """Test that index is persisted after embedding."""
    metadata_id = sample_metadata.id
    
    with patch('backend.utils.embedding_runner.load_index_from_storage') as mock_load, \
         patch('backend.utils.embedding_runner.VectorStoreIndex') as mock_index_class, \
         patch('backend.utils.embedding_runner.QuerySummarizer') as mock_summarizer_class, \
         patch('backend.utils.embedding_runner.Settings') as mock_settings, \
         patch('backend.utils.embedding_runner.HuggingFaceEmbedding') as mock_embed, \
         patch('os.path.exists', return_value=True), \
         patch('os.makedirs'):
        
        # Setup mocks
        mock_index = Mock()
        mock_index.insert_nodes = Mock()
        mock_storage_context = Mock()
        mock_storage_context.persist = Mock()
        mock_index.storage_context = mock_storage_context
        mock_load.return_value = mock_index
        
        mock_summarizer = Mock()
        mock_summarizer.summarize = Mock(return_value=("Summary", True, None))
        mock_summarizer_class.return_value = mock_summarizer
        
        mock_settings.embed_model = None
        
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            run_embedding(metadata_id)
        finally:
            os.chdir(original_cwd)
        
        # Verify index was persisted
        mock_storage_context.persist.assert_called_once_with(persist_dir="latest_model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

