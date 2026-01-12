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
from sqlalchemy import or_, func

from .db import Document, MachineModel, SessionLocal
from .filenames import canonicalize_filename

# Special values historically used by orchestrator for "global" docs.
# These are not MachineModel table rows; treat them as reserved tokens.
ANY_MACHINE = "Any"
GENERAL_MACHINE = "GENERAL"


def get_allowed_machine_models(session: Optional[Session] = None) -> list[str]:
    """
    Return allowed machine model names from the database (source of truth),
    plus reserved tokens (GENERAL/Any) for backward compatibility.
    """
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True
    try:
        names = [row.name for row in session.query(MachineModel).order_by(MachineModel.name.asc()).all() if row.name]
        # Include reserved tokens at the end (kept for legacy filtering behavior)
        return names + [GENERAL_MACHINE, ANY_MACHINE]
    finally:
        if close_session:
            session.close()


def is_valid_machine_model(model: str | None, session: Optional[Session] = None) -> bool:
    """Validate a machine model name against machine_models table (case-insensitive), allowing reserved tokens."""
    if model is None:
        return False
    m = str(model).strip()
    if not m:
        return False
    if m in {ANY_MACHINE, GENERAL_MACHINE}:
        return True
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True
    try:
        from sqlalchemy import func
        return (
            session.query(MachineModel.id)
            .filter(func.upper(MachineModel.name) == " ".join(m.upper().split()))
            .first()
            is not None
        )
    finally:
        if close_session:
            session.close()


def is_valid_machine_model_list(models: list[str] | None, session: Optional[Session] = None) -> bool:
    if models is None:
        return False
    filtered = [m for m in models if m and isinstance(m, str) and m.strip()]
    if len(filtered) == 0:
        return False
    # Validate all
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True
    try:
        return all(is_valid_machine_model(m, session=session) for m in filtered)
    finally:
        if close_session:
            session.close()

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
    """
    Get a document by filename with multiple fallback strategies.
    
    Tries in order:
    1. Exact match on Document.file_name (is_active=True)
    2. Exact match on Document.display_name (is_active=True)
    3. Canonicalized match: canonicalize input and match against canonicalized file_name/display_name
    4. Case-insensitive match on file_name (is_active=True)
    5. Case-insensitive match on display_name (is_active=True)
    
    Logs which strategy succeeded for debugging.
    
    Args:
        session: SQLAlchemy session
        filename: Filename to search for (may be original or canonicalized)
    
    Returns:
        Document object if found, None otherwise
    """
    if not filename:
        return None
    
    # Strategy 1: Exact match on file_name (is_active=True)
    doc = session.query(Document).filter(
        Document.file_name == filename,
        Document.is_active == True
    ).first()
    if doc:
        return doc
    
    # Strategy 2: Exact match on display_name (is_active=True)
    doc = session.query(Document).filter(
        Document.display_name == filename,
        Document.is_active == True
    ).filter(Document.display_name.isnot(None)).first()
    if doc:
        logger.info(
            f"Document lookup fallback: display_name exact match succeeded. "
            f"Requested: '{filename}', Matched doc.id={doc.id}, doc.file_name='{doc.file_name}'"
        )
        return doc
    
    # Strategy 3: Canonicalized match
    canonical_input = canonicalize_filename(filename)
    if canonical_input:
        # Try matching canonicalized input against canonicalized file_name
        all_active_docs = session.query(Document).filter(Document.is_active == True).all()
        for doc in all_active_docs:
            canonical_db = canonicalize_filename(doc.file_name)
            if canonical_db == canonical_input:
                logger.info(
                    f"Document lookup fallback: canonicalized file_name match succeeded. "
                    f"Requested: '{filename}' (canonical: '{canonical_input}'), "
                    f"Matched doc.id={doc.id}, doc.file_name='{doc.file_name}'"
                )
                return doc
        
        # Also try against display_name
        for doc in all_active_docs:
            if doc.display_name:
                canonical_display = canonicalize_filename(doc.display_name)
                if canonical_display == canonical_input:
                    logger.info(
                        f"Document lookup fallback: canonicalized display_name match succeeded. "
                        f"Requested: '{filename}' (canonical: '{canonical_input}'), "
                        f"Matched doc.id={doc.id}, doc.file_name='{doc.file_name}', doc.display_name='{doc.display_name}'"
                    )
                    return doc
    
    # Strategy 4: Case-insensitive match on file_name (is_active=True)
    doc = session.query(Document).filter(
        func.lower(Document.file_name) == func.lower(filename),
        Document.is_active == True
    ).first()
    if doc:
        logger.info(
            f"Document lookup fallback: case-insensitive file_name match succeeded. "
            f"Requested: '{filename}', Matched doc.id={doc.id}, doc.file_name='{doc.file_name}'"
        )
        return doc
    
    # Strategy 5: Case-insensitive match on display_name (is_active=True)
    doc = session.query(Document).filter(
        func.lower(Document.display_name) == func.lower(filename),
        Document.is_active == True
    ).filter(Document.display_name.isnot(None)).first()
    if doc:
        logger.info(
            f"Document lookup fallback: case-insensitive display_name match succeeded. "
            f"Requested: '{filename}', Matched doc.id={doc.id}, doc.file_name='{doc.file_name}', doc.display_name='{doc.display_name}'"
        )
        return doc
    
    return None


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
        
        # Prefer many-to-many mapping; fallback to legacy string field
        machine_model = None
        try:
            if hasattr(doc, "machine_models") and doc.machine_models:
                machine_model = [m.name for m in doc.machine_models if getattr(m, "name", None)]
        except Exception:
            machine_model = None

        if not machine_model:
            # Parse legacy machine_model if it's a JSON string
            machine_model = doc.machine_model
            if machine_model and isinstance(machine_model, str):
                try:
                    machine_model = json.loads(machine_model)
                except (json.JSONDecodeError, TypeError):
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
    
    IMPORTANT: file_name is canonicalized for consistent document identity.
    display_name preserves the original filename for UI display.
    
    Args:
        session: SQLAlchemy session
        file_name: Filename (will be canonicalized for Document.file_name)
        gcs_path: Cloud Storage path
        display_name: Display name (defaults to original file_name before canonicalization)
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
    # Canonicalize filename for consistent identity
    canonical_file_name = canonicalize_filename(file_name)
    original_display_name = display_name or file_name
    # Normalize machine_model to JSON string (legacy column) and update M2M join table.
    machine_model_str = None
    if machine_model is not None:
        if isinstance(machine_model, list):
            # Validate all models
            if is_valid_machine_model_list(machine_model, session=session):
                machine_model_str = json.dumps(machine_model)
            else:
                invalid_models = [m for m in machine_model if not is_valid_machine_model(m)]
                logger.warning(f"Invalid machine_model(s) {invalid_models} for {file_name}")
                machine_model_str = None
                if requires_admin_review is None:
                    requires_admin_review = True
        elif isinstance(machine_model, str):
            if is_valid_machine_model(machine_model, session=session):
                machine_model_str = machine_model
            else:
                logger.warning(f"Invalid machine_model '{machine_model}' for {file_name}")
                machine_model_str = None
                if requires_admin_review is None:
                    requires_admin_review = True
    
    # Check if document exists (by canonical filename)
    doc = get_document_by_filename(session, canonical_file_name)
    
    if doc is None:
        # Create new document with canonical file_name
        doc = Document(
            file_name=canonical_file_name,  # Canonical for identity
            gcs_path=gcs_path,
            display_name=original_display_name,  # Original for display
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
        # Note: file_name should remain canonical (don't change it)
        if gcs_path is not None:
            doc.gcs_path = gcs_path
        if original_display_name is not None:
            doc.display_name = original_display_name
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
    
    # Update many-to-many join table if machine_model provided and valid
    if machine_model is not None and machine_model_str is not None:
        # Convert to list of names
        names: list[str]
        if isinstance(machine_model, list):
            names = [m.strip() for m in machine_model if isinstance(m, str) and m.strip()]
        else:
            names = [str(machine_model).strip()]

        # Resolve to MachineModel rows (excluding reserved tokens)
        from sqlalchemy import func
        normalized = [" ".join(n.upper().split()) for n in names if n not in {ANY_MACHINE, GENERAL_MACHINE}]
        mm_rows = []
        if normalized:
            mm_rows = session.query(MachineModel).filter(func.upper(MachineModel.name).in_(normalized)).all()
        # Attach relationship (canonical). Reserved tokens remain only in legacy string column.
        try:
            doc.machine_models = mm_rows
        except Exception as e:
            logger.warning(f"Failed to set document machine_models relationship for {file_name}: {e}")

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
            # Prefer many-to-many mapping; fallback to legacy string field
            machine_model = None
            try:
                if hasattr(doc, "machine_models") and doc.machine_models:
                    machine_model = [m.name for m in doc.machine_models if getattr(m, "name", None)]
            except Exception:
                machine_model = None

            if not machine_model:
                machine_model = doc.machine_model
                if machine_model and isinstance(machine_model, str):
                    try:
                        machine_model = json.loads(machine_model)
                    except (json.JSONDecodeError, TypeError):
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
