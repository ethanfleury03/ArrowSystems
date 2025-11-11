"""
Document Metadata Management

Stores and manages document metadata including:
- is_active flag (enable/disable)
- machine_model
- category
- product_family
- last_ingestion_date
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

METADATA_FILE = "data/document_metadata.json"


def load_metadata() -> Dict[str, Dict[str, Any]]:
    """Load document metadata from JSON file."""
    if not os.path.exists(METADATA_FILE):
        return {}
    
    try:
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return {}


def save_metadata(metadata: Dict[str, Dict[str, Any]]):
    """Save document metadata to JSON file."""
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    
    try:
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")
        raise


def get_document_metadata(filename: str) -> Dict[str, Any]:
    """Get metadata for a specific document."""
    metadata = load_metadata()
    return metadata.get(filename, {
        "is_active": True,  # Default to active
        "machine_model": None,
        "category": None,
        "product_family": None,
        "last_ingestion_date": None
    })


def update_document_metadata(filename: str, updates: Dict[str, Any]):
    """Update metadata for a specific document."""
    metadata = load_metadata()
    
    if filename not in metadata:
        metadata[filename] = {
            "is_active": True,
            "machine_model": None,
            "category": None,
            "product_family": None,
            "last_ingestion_date": None
        }
    
    # Update fields
    for key, value in updates.items():
        if key in metadata[filename]:
            metadata[filename][key] = value
    
    save_metadata(metadata)


def set_document_active(filename: str, is_active: bool):
    """Set document active/inactive status."""
    update_document_metadata(filename, {"is_active": is_active})


def update_ingestion_date(filename: str):
    """Update last ingestion date to current timestamp."""
    update_document_metadata(filename, {
        "last_ingestion_date": datetime.now().isoformat()
    })


def delete_document_metadata(filename: str):
    """Remove metadata for a deleted document."""
    metadata = load_metadata()
    if filename in metadata:
        del metadata[filename]
        save_metadata(metadata)


def is_document_active(filename: str) -> bool:
    """Check if document is active."""
    doc_meta = get_document_metadata(filename)
    return doc_meta.get("is_active", True)  # Default to active if not set

