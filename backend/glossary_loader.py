"""
Glossary loaders: CSV (preferred) and simple PDF fallback.

Conventions:
- CSV columns: term, definition, aliases (pipe-separated)
- PDF fallback expects lines like "Term: Definition" or "Term - Definition"
"""

import os
import csv
from typing import List, Dict

from llama_index.core.schema import TextNode


def load_glossary_csv(path: str) -> List[TextNode]:
    nodes: List[TextNode] = []
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
                }
            )
            nodes.append(node)
    return nodes


def load_glossary_pdf(path: str) -> List[TextNode]:
    """
    Simple heuristic PDF parser for term-definition lines.
    Looks for "Term: Definition" or "Term - Definition".
    """
    try:
        import fitz  # PyMuPDF
    except Exception:
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
                        }
                    )
                    nodes.append(node)
    finally:
        doc.close()
    return nodes


def load_glossary_any(path: str) -> List[TextNode]:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return load_glossary_csv(path)
    if ext == '.pdf':
        return load_glossary_pdf(path)
    return []


