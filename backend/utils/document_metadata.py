"""
Document Metadata Management

Stores and manages document metadata including:
- is_active flag (enable/disable)
- machine_model (MUST be from ALLOWED_MACHINE_MODELS - inferred from filename if not provided)
- category
- product_family
- last_ingestion_date
- requires_admin_review (flag for documents needing manual review)
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from ..config.machine_models import (
        is_valid_machine_model,
        is_valid_machine_model_list,
        get_allowed_machine_models,
        ANY_MACHINE
    )
except ImportError:
    # Fallback if config not available (shouldn't happen in production)
    def is_valid_machine_model(model: str | None) -> bool:
        return model is not None
    
    def is_valid_machine_model_list(models: list[str] | None) -> bool:
        return models is not None and len(models) > 0
    
    def get_allowed_machine_models() -> list[str]:
        return []
    
    ANY_MACHINE = "Any"

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
    default_meta = metadata.get(filename, {
        "is_active": True,  # Default to active
        "machine_model": None,  # Can be None, a list of strings, or a single string (for backwards compatibility)
        "category": None,
        "product_family": None,
        "last_ingestion_date": None,
        "requires_admin_review": False
    })
    
    # Normalize machine_model: convert single string to list for consistency
    machine_model = default_meta.get("machine_model")
    if isinstance(machine_model, str):
        default_meta["machine_model"] = [machine_model]
    elif machine_model is None:
        default_meta["machine_model"] = None
    
    return default_meta


def update_document_metadata(filename: str, updates: Dict[str, Any]):
    """
    Update metadata for a specific document.
    
    Validates that machine_model (if provided) is a list of values from ALLOWED_MACHINE_MODELS.
    If invalid, sets machine_model to None and requires_admin_review to True.
    
    machine_model can be:
    - None (no machine model assigned)
    - A list of strings (e.g., ["EZCut 330", "EZCut 350"])
    - An empty list (treated as None)
    - "Any" as a single-item list (indicates document applies to any machine)
    """
    metadata = load_metadata()
    
    if filename not in metadata:
        metadata[filename] = {
            "is_active": True,
            "machine_model": None,
            "category": None,
            "product_family": None,
            "last_ingestion_date": None,
            "requires_admin_review": False
        }
    
    # Validate and update machine_model if provided
    if "machine_model" in updates:
        machine_model = updates["machine_model"]
        
        # Normalize: convert to list format
        # Handle various input formats for backwards compatibility
        if machine_model is None:
            machine_models_list = None
        elif machine_model == "":
            machine_models_list = None
        elif isinstance(machine_model, str):
            # Single string -> convert to list
            machine_models_list = [machine_model] if machine_model else None
        elif isinstance(machine_model, list):
            # Filter out empty strings and None values
            machine_models_list = [m for m in machine_model if m and isinstance(m, str)]
            if len(machine_models_list) == 0:
                machine_models_list = None
            # If "Any" is present, it should be the only item
            if ANY_MACHINE in machine_models_list and len(machine_models_list) > 1:
                logger.warning(f"Invalid machine_model list for {filename}: 'Any' cannot be combined with other models. Using only 'Any'.")
                machine_models_list = [ANY_MACHINE]
        else:
            logger.warning(f"Invalid machine_model type for {filename}: {type(machine_model)}. Setting to None.")
            machine_models_list = None
        
        # Validate: must be None or a valid list
        if machine_models_list is not None and not is_valid_machine_model_list(machine_models_list):
            invalid_models = [m for m in machine_models_list if not is_valid_machine_model(m)]
            logger.warning(f"Invalid machine_model(s) {invalid_models} for {filename} - not in allowed list. Setting to None and marking for review.")
            machine_models_list = None
            updates["requires_admin_review"] = True
        
        # If valid machine_model is set, clear requires_admin_review
        if machine_models_list is not None and is_valid_machine_model_list(machine_models_list):
            metadata[filename]["requires_admin_review"] = False
        elif machine_models_list is None:
            # If setting to None, mark for review
            updates["requires_admin_review"] = True
        
        updates["machine_model"] = machine_models_list
    
    # Update fields
    for key, value in updates.items():
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


def infer_machine_model_from_filename(filename: str) -> Optional[list[str]]:
    """
    Infer machine model(s) from filename using pattern matching.
    ONLY returns values that are in ALLOWED_MACHINE_MODELS.
    Can return multiple models if the filename suggests multiple machines.
    
    Examples:
        "EZCut_330_Manual.pdf" -> ["EZCut 330"] (if in allowed list)
        "DuraFlex_ServiceGuide.pdf" -> ["Duraflex"] (if in allowed list)
        "EZCut_330_and_350_Manual.pdf" -> ["EZCut 330", "EZCut 350"] (if both in allowed list)
    
    Returns:
        List of inferred machine model strings from ALLOWED_MACHINE_MODELS, or None if inference fails
    """
    # Get allowed models list (excluding "Any" from inference)
    allowed_models = [m for m in get_allowed_machine_models() if m != ANY_MACHINE]
    if not allowed_models:
        # If no allowed models configured, can't infer anything
        return None
    
    # Remove file extension
    name_without_ext = Path(filename).stem
    name_lower = name_without_ext.lower()
    
    inferred_models = []
    
    # Check if any allowed model appears in the filename (case-insensitive)
    for model in allowed_models:
        model_lower = model.lower()
        # Check if model appears as a word boundary match (at start, before underscore, or standalone)
        # Pattern: model at start, or model before underscore/dash/space
        patterns = [
            r'^' + re.escape(model_lower) + r'(?:[_\s-]|$)',  # At start
            r'(?:[_\s-])' + re.escape(model_lower) + r'(?:[_\s-]|$)',  # After separator
            r'\b' + re.escape(model_lower) + r'\b',  # Word boundary
        ]
        for pattern in patterns:
            if re.search(pattern, name_lower):
                # Avoid duplicates
                if model not in inferred_models:
                    inferred_models.append(model)  # Add original case from allowed list
                break  # Found this model, move to next
    
    # Also check if the first token before underscore matches an allowed model
    match = re.match(r'^([A-Za-z0-9]+?)(?:_|$)', name_without_ext)
    if match:
        candidate = match.group(1)
        # Check if candidate (case-insensitive) matches any allowed model
        for model in allowed_models:
            if candidate.lower() == model.lower():
                if model not in inferred_models:
                    inferred_models.append(model)
                break
    
    # Return list if any models found, otherwise None
    return inferred_models if inferred_models else None


def require_machine_model(filename: str) -> bool:
    """
    Returns True if metadata exists and machine_model is not None or empty list.
    
    Args:
        filename: Document filename
        
    Returns:
        True if machine_model is set (not None and not empty), False otherwise
    """
    doc_meta = get_document_metadata(filename)
    machine_model = doc_meta.get("machine_model")
    if machine_model is None:
        return False
    if isinstance(machine_model, list):
        return len(machine_model) > 0
    # Handle legacy string format
    if isinstance(machine_model, str):
        return len(machine_model) > 0
    return machine_model is not None


def ensure_metadata_entry(filename: str, machine_model: Optional[list[str] | str] = None) -> Dict[str, Any]:
    """
    Ensure a metadata entry exists for a document with all required fields.
    If machine_model is not provided, attempts automatic inference.
    
    Validates that machine_model is a list of values from ALLOWED_MACHINE_MODELS.
    
    Args:
        filename: Document filename
        machine_model: Optional machine model (can be string, list of strings, or None)
                      Must be in ALLOWED_MACHINE_MODELS if provided
        
    Returns:
        Dictionary with metadata entry (including requires_admin_review flag)
    """
    metadata = load_metadata()
    
    # Normalize machine_model to list format
    machine_models_list = None
    if machine_model is not None:
        if isinstance(machine_model, str):
            machine_models_list = [machine_model]
        elif isinstance(machine_model, list):
            machine_models_list = machine_model
        else:
            logger.warning(f"Invalid machine_model type for {filename}: {type(machine_model)}")
            machine_models_list = None
    
    # Validate provided machine_model
    if machine_models_list is not None and not is_valid_machine_model_list(machine_models_list):
        logger.warning(f"Invalid machine_model '{machine_models_list}' for {filename} - not in allowed list. Attempting inference.")
        machine_models_list = None
    
    if filename not in metadata:
        # Attempt to infer machine_model if not provided
        inferred_models = machine_models_list
        requires_review = False
        
        if not inferred_models:
            inferred_models = infer_machine_model_from_filename(filename)
            if not inferred_models or not is_valid_machine_model_list(inferred_models):
                requires_review = True
                inferred_models = None
                logger.warning(f"Could not infer valid machine_model for {filename}, marking for admin review")
        
        metadata[filename] = {
            "is_active": True,
            "machine_model": inferred_models,
            "category": None,
            "product_family": None,
            "last_ingestion_date": datetime.now().isoformat(),
            "requires_admin_review": requires_review
        }
        save_metadata(metadata)
    else:
        # Update ingestion date if entry exists
        metadata[filename]["last_ingestion_date"] = datetime.now().isoformat()
        
        # If machine_model was provided and differs, update it (with validation)
        if machine_models_list is not None:
            if is_valid_machine_model_list(machine_models_list):
                metadata[filename]["machine_model"] = machine_models_list
                metadata[filename]["requires_admin_review"] = False
            else:
                logger.warning(f"Invalid machine_model '{machine_models_list}' for {filename} - keeping existing value")
        
        # Ensure requires_admin_review field exists and current machine_model is valid
        if "requires_admin_review" not in metadata[filename]:
            current_model = metadata[filename].get("machine_model")
            # Normalize current_model to list for validation
            if isinstance(current_model, str):
                current_model_list = [current_model]
            elif isinstance(current_model, list):
                current_model_list = current_model
            else:
                current_model_list = None
            
            metadata[filename]["requires_admin_review"] = (
                current_model_list is None or not is_valid_machine_model_list(current_model_list)
            )
        else:
            # Validate existing machine_model - if invalid, mark for review
            current_model = metadata[filename].get("machine_model")
            # Normalize to list
            if isinstance(current_model, str):
                current_model_list = [current_model]
            elif isinstance(current_model, list):
                current_model_list = current_model
            else:
                current_model_list = None
            
            if current_model_list is not None and not is_valid_machine_model_list(current_model_list):
                logger.warning(f"Existing invalid machine_model '{current_model_list}' for {filename} - marking for review")
                metadata[filename]["machine_model"] = None
                metadata[filename]["requires_admin_review"] = True
        
        save_metadata(metadata)
    
    return metadata[filename]

