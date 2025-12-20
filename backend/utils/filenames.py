"""
Canonical filename utilities for consistent document identity across DB and index.

This module provides a canonical filename policy that ensures:
- Database Document.file_name uses canonical format
- Index node metadata file_name matches DB canonical format
- Query-time filtering can reliably match DB records to index chunks
"""

import os
import re
from typing import Optional, Any, Tuple


def canonicalize_filename(name: str) -> str:
    """
    Convert a filename to canonical form for consistent document identity.
    
    Canonical form:
    - Stripped of whitespace
    - Spaces replaced with underscores
    - Special characters normalized/removed
    - Repeated underscores collapsed
    - Lowercase (optional, but recommended for consistency)
    
    This ensures that "User Manual.pdf", "user_manual.pdf", and "User_Manual.pdf"
    all map to the same canonical identifier.
    
    Args:
        name: Original filename (can include path, will extract basename)
        
    Returns:
        Canonical filename string (basename only, normalized)
        
    Example:
        >>> canonicalize_filename("User Manual (2024).pdf")
        'user_manual_2024.pdf'
        >>> canonicalize_filename("data/Some Document.pdf")
        'some_document.pdf'
    """
    if not name:
        return ""
    
    # Extract basename (handle paths)
    basename = os.path.basename(name)
    
    # Strip whitespace
    canonical = basename.strip()
    
    # Try using werkzeug's secure_filename if available (better handling)
    try:
        from werkzeug.utils import secure_filename
        canonical = secure_filename(canonical)
        # secure_filename already handles most cases, but we'll do additional normalization
    except ImportError:
        # Fallback: manual normalization
        # Replace spaces with underscores
        canonical = canonical.replace(" ", "_")
        
        # Remove or normalize special characters
        # Keep: letters, numbers, dots, hyphens, underscores
        # Remove: ()[]{}, and other special chars
        canonical = re.sub(r'[()\[\]{}]', '', canonical)
        
        # Normalize other problematic characters
        canonical = canonical.replace(",", "_")
        canonical = canonical.replace(";", "_")
        canonical = canonical.replace(":", "_")
        canonical = canonical.replace("'", "")
        canonical = canonical.replace('"', "")
    
    # Collapse repeated underscores and hyphens
    canonical = re.sub(r'[_-]+', '_', canonical)
    
    # Strip trailing underscores/dashes before file extension
    # Pattern: remove trailing _- before .ext
    if '.' in canonical:
        name_part, ext = canonical.rsplit('.', 1)
        # Remove trailing underscores/dashes from name part
        name_part = name_part.rstrip('_-')
        canonical = f"{name_part}.{ext}"
    
    # Remove leading/trailing underscores
    canonical = canonical.strip('_')
    
    # Convert to lowercase for consistency (optional but recommended)
    # Note: This preserves file extension case (usually .pdf, .PDF, etc.)
    # We'll lowercase everything except the extension
    if '.' in canonical:
        name_part, ext = canonical.rsplit('.', 1)
        canonical = f"{name_part.lower()}.{ext.lower()}"
    else:
        canonical = canonical.lower()
    
    # Final cleanup: remove any remaining problematic characters
    # Keep only: alphanumeric, dots, hyphens, underscores
    canonical = re.sub(r'[^a-zA-Z0-9._-]', '', canonical)
    
    # Ensure we don't end up with empty string
    if not canonical:
        # Fallback: use a sanitized version of original
        original_basename = os.path.basename(name)
        canonical = re.sub(r'[^a-zA-Z0-9._-]', '_', original_basename.lower())
        canonical = re.sub(r'_+', '_', canonical).strip('_')
        if not canonical:
            canonical = "unnamed_file"
    
    return canonical


def normalize_filename_for_comparison(name: Optional[str]) -> str:
    """
    Normalize a filename for comparison purposes (more lenient than canonicalize).
    
    This is used when comparing filenames that might be in different formats.
    It's more tolerant and handles edge cases during migration.
    
    Args:
        name: Filename to normalize (can be None, empty, or include path)
        
    Returns:
        Normalized filename for comparison
    """
    if not name:
        return ""
    
    # Extract basename
    basename = os.path.basename(name)
    
    # Strip and lowercase
    normalized = basename.strip().lower()
    
    # Normalize spaces and underscores (treat as equivalent)
    normalized = normalized.replace(" ", "_")
    
    # Collapse repeated underscores
    normalized = re.sub(r'_+', '_', normalized)
    
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    
    return normalized


def get_canonical_basename(path_or_name: str) -> str:
    """
    Extract and canonicalize the basename from a path or filename.
    
    Convenience function that combines os.path.basename with canonicalize_filename.
    
    Args:
        path_or_name: Full path or just filename
        
    Returns:
        Canonical basename
    """
    return canonicalize_filename(path_or_name)


def ensure_node_has_filename(node: Any, strict: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Ensure a node has a canonical file_name in metadata.
    
    Attempts repair if file_name is missing or empty by extracting from:
    - gcs_path (basename)
    - source_path (basename)
    - file_path (basename)
    - filename (direct key)
    
    Args:
        node: Node object (TextNode, ImageNode, or dict-like with metadata)
        strict: If True, return False if cannot repair. If False, return True with fallback.
        
    Returns:
        (success: bool, file_name: Optional[str])
        - success: True if node now has file_name (was present or repaired)
        - file_name: The canonical filename that was set/found (None if failed)
    """
    # Extract metadata
    metadata = None
    if hasattr(node, 'metadata'):
        metadata = node.metadata
    elif isinstance(node, dict):
        metadata = node.get('metadata', {})
    else:
        # Try to get from __data__ wrapper
        if isinstance(node, dict) and '__data__' in node:
            inner = node['__data__']
            if isinstance(inner, dict):
                metadata = inner.get('metadata', {})
    
    if not metadata or not isinstance(metadata, dict):
        return (False, None)
    
    # Check if file_name already exists and is non-empty
    existing_file_name = metadata.get('file_name', '') or metadata.get('filename', '')
    if existing_file_name and str(existing_file_name).strip():
        # Already has filename, just canonicalize it
        canonical = canonicalize_filename(str(existing_file_name))
        if canonical != existing_file_name:
            metadata['file_name'] = canonical
        return (True, canonical)
    
    # Attempt repair from other metadata keys
    repair_keys = ['gcs_path', 'source_path', 'file_path', 'filename']
    for key in repair_keys:
        value = metadata.get(key)
        if value and isinstance(value, str) and value.strip():
            # Extract basename and canonicalize
            basename = os.path.basename(value)
            if basename:
                canonical = canonicalize_filename(basename)
                if canonical:
                    metadata['file_name'] = canonical
                    return (True, canonical)
    
    # Last resort: check if there's a ref_doc_id or document_id that we can map
    # (This would require access to document mapping, which we don't have here)
    # For now, if strict mode, fail
    if strict:
        return (False, None)
    
    # Non-strict: use fallback
    fallback = "unnamed_document"
    metadata['file_name'] = fallback
    return (True, fallback)


def ensure_node_has_filename(node: Any, strict: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Ensure a node has a canonical file_name in metadata.
    
    Attempts repair if file_name is missing or empty by extracting from:
    - gcs_path (basename)
    - source_path (basename)
    - file_path (basename)
    - filename (direct key)
    
    Args:
        node: Node object (TextNode, ImageNode, or dict-like with metadata)
        strict: If True, return False if cannot repair. If False, return True with fallback.
        
    Returns:
        (success: bool, file_name: Optional[str])
        - success: True if node now has file_name (was present or repaired)
        - file_name: The canonical filename that was set/found (None if failed)
    """
    # Extract metadata
    metadata = None
    if hasattr(node, 'metadata'):
        metadata = node.metadata
    elif isinstance(node, dict):
        metadata = node.get('metadata', {})
    else:
        # Try to get from __data__ wrapper
        if isinstance(node, dict) and '__data__' in node:
            inner = node['__data__']
            if isinstance(inner, dict):
                metadata = inner.get('metadata', {})
    
    if not metadata or not isinstance(metadata, dict):
        return (False, None)
    
    # Check if file_name already exists and is non-empty
    existing_file_name = metadata.get('file_name', '') or metadata.get('filename', '')
    if existing_file_name and str(existing_file_name).strip():
        # Already has filename, just canonicalize it
        canonical = canonicalize_filename(str(existing_file_name))
        if canonical != existing_file_name:
            metadata['file_name'] = canonical
        return (True, canonical)
    
    # Attempt repair from other metadata keys
    repair_keys = ['gcs_path', 'source_path', 'file_path', 'filename']
    for key in repair_keys:
        value = metadata.get(key)
        if value and isinstance(value, str) and value.strip():
            # Extract basename and canonicalize
            basename = os.path.basename(value)
            if basename:
                canonical = canonicalize_filename(basename)
                if canonical:
                    metadata['file_name'] = canonical
                    return (True, canonical)
    
    # Last resort: check if there's a ref_doc_id or document_id that we can map
    # (This would require access to document mapping, which we don't have here)
    # For now, if strict mode, fail
    if strict:
        return (False, None)
    
    # Non-strict: use fallback
    fallback = "unnamed_document"
    metadata['file_name'] = fallback
    return (True, fallback)

