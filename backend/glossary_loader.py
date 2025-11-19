"""
Glossary loaders: Database (preferred) and CSV/PDF fallback.

Phase 1: Migrated to use glossary_terms table in PostgreSQL.
Falls back to CSV/PDF file loading if database is not available or in test mode.
"""

import os
import csv
import logging
from typing import List, Dict, Optional

from llama_index.core.schema import TextNode

logger = logging.getLogger(__name__)


def load_glossary_from_db(session=None) -> List[TextNode]:
    """
    Load glossary terms from the database.
    
    Args:
        session: Optional SQLAlchemy session. If None, creates a new one.
    
    Returns:
        List of TextNode objects for glossary terms
    """
    try:
        from backend.utils.db import SessionLocal, GlossaryTerm
        
        close_session = False
        if session is None:
            session = SessionLocal()
            close_session = True
        
        try:
            terms = session.query(GlossaryTerm).all()
            nodes: List[TextNode] = []
            
            for term_record in terms:
                term = term_record.term
                definition = term_record.definition
                aliases = term_record.aliases or []
                
                if not term or not definition:
                    continue
                
                text = f"{term}: {definition}"
                node = TextNode(
                    text=text,
                    metadata={
                        'type': 'glossary',
                        'term': term,
                        'aliases': aliases if isinstance(aliases, list) else [],
                        'source': 'database',
                    }
                )
                nodes.append(node)
            
            logger.info(f"Loaded {len(nodes)} glossary terms from database")
            return nodes
            
        finally:
            if close_session:
                session.close()
                
    except Exception as e:
        logger.warning(f"Failed to load glossary from database: {e}")
        return []


def load_glossary_csv(path: str) -> List[TextNode]:
    """
    Load glossary from CSV file (fallback method).
    
    CSV columns: term, definition, aliases (pipe-separated)
    """
    nodes: List[TextNode] = []
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                term = (row.get('term') or '').strip()
                definition = (row.get('definition') or '').strip()
                aliases_raw = (row.get('aliases') or '').strip()
                if not term or not definition:
                    continue
                aliases: List[str] = [a.strip() for a in aliases_raw.split('|') if a.strip()]
                text = f"{term}: {definition}"
                node = TextNode(
                    text=text,
                    metadata={
                        'type': 'glossary',
                        'term': term,
                        'aliases': aliases,
                        'file_name': os.path.basename(path),
                        'source': 'csv',
                    }
                )
                nodes.append(node)
        logger.info(f"Loaded {len(nodes)} glossary terms from CSV: {path}")
    except Exception as e:
        logger.error(f"Failed to load glossary CSV from {path}: {e}")
    return nodes


def load_glossary_pdf(path: str) -> List[TextNode]:
    """
    Simple heuristic PDF parser for term-definition lines.
    Looks for "Term: Definition" or "Term - Definition".
    
    This is a fallback method and may not be used in production.
    """
    try:
        import fitz  # PyMuPDF
    except Exception:
        logger.warning("PyMuPDF not available for PDF glossary parsing")
        return []

    doc = fitz.open(path)
    nodes: List[TextNode] = []
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text() or ''
            for raw_line in text.split('\n'):
                line = raw_line.strip()
                if not line:
                    continue
                # Prefer colon, fallback to hyphen delimiter
                sep = ':' if ':' in line else (' - ' if ' - ' in line else None)
                if not sep:
                    continue
                parts = [p.strip() for p in line.split(sep, 1)]
                if len(parts) != 2:
                    continue
                term, definition = parts
                if len(term) > 1 and len(definition) > 1:
                    node = TextNode(
                        text=f"{term}: {definition}",
                        metadata={
                            'type': 'glossary',
                            'term': term,
                            'aliases': [],
                            'file_name': os.path.basename(path),
                            'page_label': str(page_num + 1),
                            'source': 'pdf',
                        }
                    )
                    nodes.append(node)
    finally:
        doc.close()
    
    logger.info(f"Loaded {len(nodes)} glossary terms from PDF: {path}")
    return nodes


def load_glossary_any(path: Optional[str] = None) -> List[TextNode]:
    """
    Load glossary from database (preferred) or fallback to file.
    
    Phase 1: Tries database first, then falls back to file if path is provided.
    
    Args:
        path: Optional path to CSV/PDF file (fallback only)
    
    Returns:
        List of TextNode objects for glossary terms
    """
    # Try database first (Phase 1 migration)
    nodes = load_glossary_from_db()
    
    if nodes:
        logger.info("Using glossary from database")
        return nodes
    
    # Fallback to file if database load failed and path is provided
    if path and os.path.exists(path):
        logger.info(f"Falling back to glossary file: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext == '.csv':
            return load_glossary_csv(path)
        elif ext == '.pdf':
            return load_glossary_pdf(path)
        else:
            logger.warning(f"Unknown glossary file type: {ext}")
            return []
    
    logger.warning("No glossary data available (database empty and no file path)")
    return []
