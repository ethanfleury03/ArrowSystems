from __future__ import annotations

from pathlib import PurePosixPath
from typing import Callable


def choose_docs_upload_object_name(
    *,
    docs_prefix: str,
    sanitized_filename: str,
    metadata_id: str,
    object_exists: Callable[[str], bool],
) -> str:
    """
    Choose a GCS object name for a document upload.

    Requirements:
    - If docs_prefix == "" (bucket root), default to "<sanitized_filename>" (no "<metadata_id>/").
    - If docs_prefix != "", default to "<docs_prefix><sanitized_filename>".
    - Do NOT overwrite an existing object: on collision, deterministically rename to:
        "<stem>__<metadata_id><ext>"
      (keeping it at the same prefix/root).
    - If that also collides (rare), append a numeric suffix.

    Notes:
    - `docs_prefix` is expected to be normalized already ("" or endswith "/"; no leading "/").
    - `sanitized_filename` should not contain path separators.
    """
    prefix = docs_prefix or ""

    # Defensive normalization (don't allow leading slash)
    if prefix.startswith("/"):
        prefix = prefix.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix = f"{prefix}/"

    # Ensure filename is a basename (no directories)
    fname = sanitized_filename.replace("\\", "/").split("/")[-1]

    base = f"{prefix}{fname}" if prefix else fname
    if not object_exists(base):
        return base

    p = PurePosixPath(fname)
    stem = p.stem
    ext = p.suffix  # includes leading dot or ""

    collision = f"{stem}__{metadata_id}{ext}"
    candidate = f"{prefix}{collision}" if prefix else collision
    if not object_exists(candidate):
        return candidate

    # Extremely rare: keep deterministic but avoid overwrite.
    i = 2
    while True:
        alt = f"{stem}__{metadata_id}__{i}{ext}"
        alt_name = f"{prefix}{alt}" if prefix else alt
        if not object_exists(alt_name):
            return alt_name
        i += 1


