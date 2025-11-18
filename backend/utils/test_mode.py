"""
Test Mode utility for safe testing of ingestion and deletion.

When TEST_MODE=true, all paths are routed to test-only directories
to prevent contamination of production data.
"""

import os
from typing import Optional


def is_test_mode() -> bool:
    """Check if test mode is enabled."""
    return os.getenv("TEST_MODE", "false").lower() == "true"


def get_index_dir() -> str:
    """Get the index directory based on test mode."""
    if is_test_mode():
        return "latest_model_test"
    return "latest_model"


def get_chunks_dir() -> str:
    """Get the chunks directory based on test mode."""
    if is_test_mode():
        return "data/chunks_test"
    return "data/chunks"


def get_original_pdfs_dir() -> str:
    """Get the original PDFs directory based on test mode."""
    if is_test_mode():
        return "data/original_pdfs_test"
    return "data/original_pdfs"


def get_temp_index_dir() -> str:
    """Get the temporary index directory for atomic swaps based on test mode."""
    if is_test_mode():
        return "latest_model_test_tmp"
    return "latest_model_tmp"

