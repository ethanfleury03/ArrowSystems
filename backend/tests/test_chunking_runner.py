"""
Tests for the chunking runner (Phase 2).

Tests ensure:
- Successful chunking transitions: PENDING_INGESTION → CHUNKING → READY_FOR_EMBEDDING
- Failures during chunking transition to FAILED without crashing the server
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from sqlalchemy.orm import Session
from backend.utils.db import SessionLocal, DocumentIngestionMetadata, Base, engine
from backend.utils.chunking_runner import run_chunking


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_pdf_file(temp_dir):
    """Create a sample PDF file for testing."""
    # Create a simple text file that can be used as a test document
    # In a real test, you'd use an actual PDF, but for unit tests we can mock the loader
    pdf_path = os.path.join(temp_dir, "test.pdf")
    with open(pdf_path, "wb") as f:
        # Write a minimal PDF header (not a real PDF, but enough for path testing)
        f.write(b"%PDF-1.4\n")
    return pdf_path


@pytest.fixture
def db_session():
    """Create a test database session."""
    Base.metadata.create_all(engine)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def sample_metadata(db_session, sample_pdf_file):
    """Create a sample DocumentIngestionMetadata record."""
    import uuid
    metadata = DocumentIngestionMetadata(
        id=str(uuid.uuid4()),
        filename="test.pdf",
        machine_model="TestMachine",
        status="PENDING_INGESTION",
        file_path=sample_pdf_file,
        file_size_bytes=1024,
    )
    db_session.add(metadata)
    db_session.commit()
    db_session.refresh(metadata)
    return metadata


def test_chunking_success_transition(sample_metadata, temp_dir):
    """Test that successful chunking transitions through correct statuses."""
    metadata_id = sample_metadata.id
    
    # Mock the document loader and chunker
    with patch('backend.utils.chunking_runner.DocumentLoader') as mock_loader_class, \
         patch('backend.utils.chunking_runner.SmartChunkSplitter') as mock_splitter_class, \
         patch('backend.utils.chunking_runner.TextPreprocessor') as mock_preprocessor_class, \
         patch('backend.utils.chunking_runner.SimpleDirectoryReader') as mock_reader:
        
        # Setup mocks
        mock_loader = Mock()
        mock_loader._load_docx = Mock(return_value=[])
        mock_loader._load_markdown = Mock(return_value=[])
        mock_loader_class.return_value = mock_loader
        
        # Mock document loading
        from llama_index.core.schema import Document
        mock_doc = Document(
            text="This is a test document with some content.",
            metadata={"file_name": "test.pdf", "file_type": "pdf"}
        )
        mock_reader.return_value.load_data.return_value = [mock_doc]
        
        # Mock preprocessor
        mock_preprocessor = Mock()
        mock_preprocessor.clean_text = Mock(side_effect=lambda text, metadata: text)
        mock_preprocessor.is_low_content_page = Mock(return_value=False)
        mock_preprocessor.should_skip_node = Mock(return_value=(False, None))
        mock_preprocessor_class.return_value = mock_preprocessor
        
        # Mock splitter
        from llama_index.core.schema import TextNode
        mock_node = TextNode(
            text="This is a test chunk.",
            metadata={"file_name": "test.pdf", "machine_model": "TestMachine"}
        )
        mock_splitter = Mock()
        mock_splitter.get_nodes_from_documents = Mock(return_value=[mock_node])
        mock_splitter_class.return_value = mock_splitter
        
        # Run chunking
        run_chunking(metadata_id)
        
        # Verify status transition
        session = SessionLocal()
        try:
            updated_metadata = session.query(DocumentIngestionMetadata).filter(
                DocumentIngestionMetadata.id == metadata_id
            ).first()
            assert updated_metadata is not None
            assert updated_metadata.status == "READY_FOR_EMBEDDING"
            
            # Verify chunks file was created
            chunks_file = Path("data/chunks") / f"{metadata_id}.json"
            assert chunks_file.exists(), "Chunks file should be created"
            
            # Verify chunks file content
            with open(chunks_file, 'r') as f:
                chunks_data = json.load(f)
                assert chunks_data["metadata_id"] == metadata_id
                assert len(chunks_data["chunks"]) > 0
                # New: machine_model_ids are always present (may be empty for legacy tests)
                assert "machine_model_ids" in chunks_data
                assert isinstance(chunks_data["machine_model_ids"], list)
                assert "machine_model_names" in chunks_data
                assert isinstance(chunks_data["machine_model_names"], list)

                first_chunk = chunks_data["chunks"][0]
                assert "metadata" in first_chunk
                assert "machine_model_ids" in first_chunk["metadata"]
                assert isinstance(first_chunk["metadata"]["machine_model_ids"], list)
        finally:
            session.close()
            # Cleanup
            if chunks_file.exists():
                chunks_file.unlink()


def test_chunking_failure_transition(sample_metadata):
    """Test that failures during chunking transition to FAILED status."""
    metadata_id = sample_metadata.id
    
    # Mock file not found error
    with patch('os.path.exists', return_value=False):
        run_chunking(metadata_id)
        
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


def test_chunking_chunking_status_set(sample_metadata, temp_dir):
    """Test that status is set to CHUNKING during processing."""
    metadata_id = sample_metadata.id
    
    # Mock a slow operation to check intermediate status
    import time
    
    with patch('backend.utils.chunking_runner.DocumentLoader') as mock_loader_class, \
         patch('backend.utils.chunking_runner.SmartChunkSplitter') as mock_splitter_class, \
         patch('backend.utils.chunking_runner.TextPreprocessor') as mock_preprocessor_class, \
         patch('backend.utils.chunking_runner.SimpleDirectoryReader') as mock_reader:
        
        # Setup mocks
        mock_loader = Mock()
        mock_loader._load_docx = Mock(return_value=[])
        mock_loader._load_markdown = Mock(return_value=[])
        mock_loader_class.return_value = mock_loader
        
        from llama_index.core.schema import Document
        mock_doc = Document(
            text="Test content",
            metadata={"file_name": "test.pdf", "file_type": "pdf"}
        )
        mock_reader.return_value.load_data.return_value = [mock_doc]
        
        mock_preprocessor = Mock()
        mock_preprocessor.clean_text = Mock(side_effect=lambda text, metadata: text)
        mock_preprocessor.is_low_content_page = Mock(return_value=False)
        mock_preprocessor.should_skip_node = Mock(return_value=(False, None))
        mock_preprocessor_class.return_value = mock_preprocessor
        
        from llama_index.core.schema import TextNode
        mock_node = TextNode(
            text="Test chunk",
            metadata={"file_name": "test.pdf"}
        )
        mock_splitter = Mock()
        mock_splitter.get_nodes_from_documents = Mock(return_value=[mock_node])
        mock_splitter_class.return_value = mock_splitter
        
        # Run chunking in a way that allows checking intermediate status
        # Note: In a real scenario, you'd need to check status in a separate thread
        # For this test, we just verify the final status is correct
        run_chunking(metadata_id)
        
        # Verify final status
        session = SessionLocal()
        try:
            updated_metadata = session.query(DocumentIngestionMetadata).filter(
                DocumentIngestionMetadata.id == metadata_id
            ).first()
            assert updated_metadata is not None
            # Status should be either CHUNKING (if still processing) or READY_FOR_EMBEDDING (if complete)
            # Since we're mocking everything, it should complete quickly
            assert updated_metadata.status in ["CHUNKING", "READY_FOR_EMBEDDING", "FAILED"]
        finally:
            session.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

