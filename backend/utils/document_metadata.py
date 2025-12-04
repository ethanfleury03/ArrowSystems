"""
Document Metadata Management

Stores and manages document metadata in the database.
Replaces the document_metadata.json file.

All functions now use SQLAlchemy sessions to interact with the documents table.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import or_

from .db import Document, SessionLocal
try:
    from ..config.machine_models import (
        is_valid_machine_model,
        is_valid_machine_model_list,
        get_allowed_machine_models,
        ANY_MACHINE,
        GENERAL_MACHINE
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
    GENERAL_MACHINE = "GENERAL"

logger = logging.getLogger(__name__)


def get_all_documents(session: Optional[Session] = None, active_only: bool = True) -> List[Document]:
    """
    Get all documents from the database.
    
    Args:
        session: Optional SQLAlchemy session. If None, creates a new one.
        active_only: If True, only return active documents.
    
    Returns:
        List of Document objects
    """
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True
    
    try:
        query = session.query(Document)
        if active_only:
            query = query.filter(Document.is_active == True)
        return query.order_by(Document.file_name).all()
    finally:
        if close_session:
            session.close()


def get_document_by_id(session: Session, doc_id: int) -> Optional[Document]:
    """Get a document by ID."""
    return session.query(Document).filter(Document.id == doc_id).first()


def get_document_by_filename(session: Session, filename: str) -> Optional[Document]:
    """Get a document by filename."""
    return session.query(Document).filter(Document.file_name == filename).first()


def get_document_metadata(filename: str, session: Optional[Session] = None) -> Dict[str, Any]:
    """
    Get metadata for a specific document by filename.
    Maintains backward compatibility with the old JSON-based API.
    
    Returns a dictionary with the same structure as before for compatibility.
    """
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True
    
    try:
        doc = get_document_by_filename(session, filename)
        
        if doc is None:
            # Return default structure for backward compatibility
            return {
                "is_active": True,
                "machine_model": None,
                "category": None,
                "product_family": None,
                "last_ingestion_date": None,
                "requires_admin_review": False
            }
        
        # Parse machine_model if it's a JSON string
        machine_model = doc.machine_model
        if machine_model and isinstance(machine_model, str):
            try:
                # Try to parse as JSON array
                machine_model = json.loads(machine_model)
            except (json.JSONDecodeError, TypeError):
                # If not JSON, treat as single string
                machine_model = [machine_model] if machine_model else None
        
        return {
            "is_active": doc.is_active,
            "machine_model": machine_model,
            "category": doc.category,
            "product_family": doc.product_family,
            "last_ingestion_date": doc.last_ingestion_date.isoformat() if doc.last_ingestion_date else None,
            "requires_admin_review": doc.requires_admin_review
        }
    finally:
        if close_session:
            session.close()


def upsert_document(
    session: Session,
    file_name: str,
    gcs_path: Optional[str] = None,
    display_name: Optional[str] = None,
    machine_model: Optional[str | List[str]] = None,
    category: Optional[str] = None,
    product_family: Optional[str] = None,
    is_active: bool = True,
    requires_admin_review: Optional[bool] = None,
    file_size_bytes: Optional[int] = None,
    last_ingestion_date: Optional[datetime] = None
) -> Document:
    """
    Upsert a document record.
    Creates a new record if file_name doesn't exist, updates if it does.
    
    Args:
        session: SQLAlchemy session
        file_name: Original filename
        gcs_path: Cloud Storage path
        display_name: Display name (defaults to file_name)
        machine_model: Machine model(s) - can be string, list, or JSON string
        category: Document category
        product_family: Product family
        is_active: Active status
        requires_admin_review: Requires admin review flag
        file_size_bytes: File size in bytes
        last_ingestion_date: Last ingestion timestamp
    
    Returns:
        Document object
    """
    # Normalize machine_model to JSON string
    machine_model_str = None
    if machine_model is not None:
        if isinstance(machine_model, list):
            # Validate all models
            if is_valid_machine_model_list(machine_model):
                machine_model_str = json.dumps(machine_model)
            else:
                invalid_models = [m for m in machine_model if not is_valid_machine_model(m)]
                logger.warning(f"Invalid machine_model(s) {invalid_models} for {file_name}")
                machine_model_str = None
                if requires_admin_review is None:
                    requires_admin_review = True
        elif isinstance(machine_model, str):
            if is_valid_machine_model(machine_model):
                machine_model_str = machine_model
            else:
                logger.warning(f"Invalid machine_model '{machine_model}' for {file_name}")
                machine_model_str = None
                if requires_admin_review is None:
                    requires_admin_review = True
    
    # Check if document exists
    doc = get_document_by_filename(session, file_name)
    
    if doc is None:
        # Create new document
        doc = Document(
            file_name=file_name,
            gcs_path=gcs_path,
            display_name=display_name or file_name,
            machine_model=machine_model_str,
            category=category,
            product_family=product_family,
            is_active=is_active,
            requires_admin_review=requires_admin_review if requires_admin_review is not None else False,
            file_size_bytes=file_size_bytes,
            last_ingestion_date=last_ingestion_date or datetime.utcnow()
        )
        session.add(doc)
    else:
        # Update existing document
        if gcs_path is not None:
            doc.gcs_path = gcs_path
        if display_name is not None:
            doc.display_name = display_name
        if machine_model_str is not None:
            doc.machine_model = machine_model_str
        if category is not None:
            doc.category = category
        if product_family is not None:
            doc.product_family = product_family
        if requires_admin_review is not None:
            doc.requires_admin_review = requires_admin_review
        if file_size_bytes is not None:
            doc.file_size_bytes = file_size_bytes
        if last_ingestion_date is not None:
            doc.last_ingestion_date = last_ingestion_date
        doc.is_active = is_active
        doc.updated_at = datetime.utcnow()
    
    session.commit()
    session.refresh(doc)
    return doc


def update_document_metadata(filename: str, updates: Dict[str, Any], session: Optional[Session] = None) -> Document:
    """
    Update metadata for a specific document.
    Maintains backward compatibility with the old API.
    
    Args:
        filename: Document filename
        updates: Dictionary of updates
        session: Optional SQLAlchemy session
    
    Returns:
        Updated Document object
    """
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True
    
    try:
        # Convert updates to upsert_document parameters
        machine_model = updates.get("machine_model")
        if machine_model is not None:
            # Normalize to list if needed
            if isinstance(machine_model, str):
                machine_model = [machine_model] if machine_model else None
            elif isinstance(machine_model, list):
                # Filter out empty strings
                machine_model = [m for m in machine_model if m and isinstance(m, str)]
                if len(machine_model) == 0:
                    machine_model = None
        
        last_ingestion_date = None
        if "last_ingestion_date" in updates:
            ingestion_date = updates["last_ingestion_date"]
            if ingestion_date is not None:
                if isinstance(ingestion_date, str):
                    try:
                        last_ingestion_date = datetime.fromisoformat(ingestion_date.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        last_ingestion_date = datetime.utcnow()
                elif isinstance(ingestion_date, datetime):
                    last_ingestion_date = ingestion_date
        
        return upsert_document(
            session=session,
            file_name=filename,
            machine_model=machine_model,
            category=updates.get("category"),
            product_family=updates.get("product_family"),
            is_active=updates.get("is_active", True),
            requires_admin_review=updates.get("requires_admin_review"),
            last_ingestion_date=last_ingestion_date
        )
    finally:
        if close_session:
            session.commit()
            session.close()


def set_document_active(filename: str, is_active: bool, session: Optional[Session] = None):
    """Set document active/inactive status."""
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True
    
    try:
        doc = get_document_by_filename(session, filename)
        if doc:
            doc.is_active = is_active
            doc.updated_at = datetime.utcnow()
            session.commit()
    finally:
        if close_session:
            session.close()


def update_ingestion_date(filename: str, session: Optional[Session] = None):
    """Update last ingestion date to current timestamp."""
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True
    
    try:
        doc = get_document_by_filename(session, filename)
        if doc:
            doc.last_ingestion_date = datetime.utcnow()
            doc.updated_at = datetime.utcnow()
            session.commit()
    finally:
        if close_session:
            session.close()


def delete_document_metadata(filename: str, session: Optional[Session] = None):
    """Remove metadata for a deleted document."""
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True
    
    try:
        doc = get_document_by_filename(session, filename)
        if doc:
            session.delete(doc)
            session.commit()
    finally:
        if close_session:
            session.close()


def is_document_active(filename: str, session: Optional[Session] = None) -> bool:
    """Check if document is active."""
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True
    
    try:
        doc = get_document_by_filename(session, filename)
        return doc.is_active if doc else True  # Default to active if not found
    finally:
        if close_session:
            session.close()


def infer_machine_model_from_filename(filename: str) -> Optional[list[str]]:
    """
    Infer machine model(s) from filename using pattern matching.
    ONLY returns values that are in ALLOWED_MACHINE_MODELS.
    
    This is a helper function for migration scripts.
    """
    import re
    from pathlib import Path
    
    allowed_models = [m for m in get_allowed_machine_models() if m != ANY_MACHINE]
    if not allowed_models:
        return None
    
    name_without_ext = Path(filename).stem
    name_lower = name_without_ext.lower()
    
    inferred_models = []
    
    for model in allowed_models:
        model_lower = model.lower()
        patterns = [
            r'^' + re.escape(model_lower) + r'(?:[_\s-]|$)',
            r'(?:[_\s-])' + re.escape(model_lower) + r'(?:[_\s-]|$)',
            r'\b' + re.escape(model_lower) + r'\b',
        ]
        for pattern in patterns:
            if re.search(pattern, name_lower):
                if model not in inferred_models:
                    inferred_models.append(model)
                break
    
    match = re.match(r'^([A-Za-z0-9]+?)(?:_|$)', name_without_ext)
    if match:
        candidate = match.group(1)
        for model in allowed_models:
            if candidate.lower() == model.lower():
                if model not in inferred_models:
                    inferred_models.append(model)
                break
    
    return inferred_models if inferred_models else None


def require_machine_model(filename: str, session: Optional[Session] = None) -> bool:
    """Returns True if document has a machine model set."""
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True
    
    try:
        doc = get_document_by_filename(session, filename)
        if doc is None:
            return False
        return doc.machine_model is not None and doc.machine_model != ""
    finally:
        if close_session:
            session.close()


def load_metadata(session: Optional[Session] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load all document metadata and return in the legacy format expected by orchestrator.
    
    This function provides backward compatibility for code that expects the old
    JSON-based metadata structure (dict mapping filename -> metadata dict).
    
    Returns:
        Dictionary mapping filename -> metadata dict with keys:
        - is_active: bool
        - machine_model: str | list[str] | None
        - category: str | None
        - product_family: str | None
        - last_ingestion_date: str | None (ISO format)
        - requires_admin_review: bool
    """
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True
    
    try:
        documents = get_all_documents(session=session, active_only=False)
        metadata_dict = {}
        
        for doc in documents:
            # Parse machine_model if it's a JSON string
            machine_model = doc.machine_model
            if machine_model and isinstance(machine_model, str):
                try:
                    # Try to parse as JSON array
                    machine_model = json.loads(machine_model)
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, treat as single string
                    machine_model = machine_model if machine_model else None
            
            metadata_dict[doc.file_name] = {
                "is_active": doc.is_active,
                "machine_model": machine_model,
                "category": doc.category,
                "product_family": doc.product_family,
                "last_ingestion_date": doc.last_ingestion_date.isoformat() if doc.last_ingestion_date else None,
                "requires_admin_review": doc.requires_admin_review
            }
        
        return metadata_dict
    finally:
        if close_session:
            session.close()


def ensure_metadata_entry(
    filename: str,
    machine_model: Optional[list[str] | str] = None,
    gcs_path: Optional[str] = None,
    session: Optional[Session] = None
) -> Document:
    """
    Ensure a metadata entry exists for a document.
    If machine_model is not provided, attempts automatic inference.
    
    Args:
        filename: Document filename
        machine_model: Optional machine model (can be string, list of strings, or None)
        gcs_path: Optional Cloud Storage path
        session: Optional SQLAlchemy session
    
    Returns:
        Document object
    """
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True
    
    try:
        doc = get_document_by_filename(session, filename)
        
        if doc is None:
            # Attempt to infer machine_model if not provided
            inferred_models = None
            requires_review = False
            
            if machine_model is None:
                inferred_models = infer_machine_model_from_filename(filename)
                if not inferred_models or not is_valid_machine_model_list(inferred_models):
                    requires_review = True
                    inferred_models = None
                    logger.warning(f"Could not infer valid machine_model for {filename}, marking for admin review")
            else:
                if isinstance(machine_model, str):
                    inferred_models = [machine_model] if is_valid_machine_model(machine_model) else None
                elif isinstance(machine_model, list):
                    inferred_models = machine_model if is_valid_machine_model_list(machine_model) else None
                
                if not inferred_models:
                    requires_review = True
            
            doc = upsert_document(
                session=session,
                file_name=filename,
                gcs_path=gcs_path,
                machine_model=inferred_models,
                is_active=True,
                requires_admin_review=requires_review,
                last_ingestion_date=datetime.utcnow()
            )
        else:
            # Update ingestion date if entry exists
            doc.last_ingestion_date = datetime.utcnow()
            doc.updated_at = datetime.utcnow()
            
            # If machine_model was provided and differs, update it
            if machine_model is not None:
                machine_models_list = None
                if isinstance(machine_model, str):
                    machine_models_list = [machine_model] if is_valid_machine_model(machine_model) else None
                elif isinstance(machine_model, list):
                    machine_models_list = machine_model if is_valid_machine_model_list(machine_model) else None
                
                if machine_models_list is not None:
                    doc.machine_model = json.dumps(machine_models_list) if len(machine_models_list) > 1 else machine_models_list[0]
                    doc.requires_admin_review = False
                else:
                    logger.warning(f"Invalid machine_model '{machine_model}' for {filename} - keeping existing value")
            
            if gcs_path is not None:
                doc.gcs_path = gcs_path
            
            session.commit()
            session.refresh(doc)
        
        return doc
    finally:
        if close_session:
            session.close()
