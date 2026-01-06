#!/usr/bin/env python3
"""
Count images inside documents stored in a GCS bucket.

Supports:
- PDF: counts embedded images per page using PyMuPDF (fitz).
- DOCX: counts embedded media files under word/media/ in the docx zip.

Default behavior is designed to match a common ingestion pattern:
- PDF counts each page image occurrence (page.get_images()).
- Optional: apply the same "RGB/GRAY only" filter used in many pipelines
  (pix.n - pix.alpha < 4), which excludes CMYK/etc.

Requirements:
- pip install pymupdf google-cloud-storage
- GCP authentication (gcloud auth application-default login)

Usage examples:
  python count_image.py gs://my-bucket/documents --csv out.csv
  python count_image.py gs://my-bucket --prefix some/folder/ --fast
  python count_image.py gs://my-bucket --no-pdf-filter
"""

import argparse
import csv
import hashlib
import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Set, Dict
from urllib.parse import urlparse

try:
    import fitz  # PyMuPDF
    from PIL import Image
except Exception as e:
    fitz = None
    Image = None

try:
    from google.cloud import storage
except ImportError:
    storage = None


DOCX_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".wmf", ".emf"}


@dataclass
class FileCount:
    uri: str
    ext: str
    image_count: int
    note: str = ""
    images_found: int = 0
    images_kept: int = 0
    images_skipped: Dict[str, int] = None


def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    """
    Parse a gs:// URI into (bucket_name, prefix).
    Returns (bucket_name, "") if no prefix.
    """
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {uri}. Must start with gs://")
    
    parsed = urlparse(uri)
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip("/")
    
    return bucket_name, prefix


def gcs_ls_recursive(bucket_name: str, prefix: str = "") -> List[str]:
    """
    Returns a list of object URIs (gs://...) under bucket_name/prefix recursively.
    """
    if storage is None:
        raise RuntimeError(
            "google-cloud-storage not installed. Run: pip install google-cloud-storage\n"
            "Also ensure you're authenticated: gcloud auth application-default login"
        )
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        uris = []
        # List all blobs with the given prefix
        blobs = bucket.list_blobs(prefix=prefix)
        
        for blob in blobs:
            # Skip "directory" markers (blobs ending with /)
            if blob.name.endswith("/"):
                continue
            uri = f"gs://{bucket_name}/{blob.name}"
            uris.append(uri)
        
        return uris
    except Exception as e:
        error_msg = str(e)
        if "DefaultCredentialsError" in error_msg or "credentials" in error_msg.lower():
            raise RuntimeError(
                f"Authentication failed. Please set up Application Default Credentials:\n"
                f"  1. Run: gcloud auth application-default login\n"
                f"  2. Or set the GOOGLE_APPLICATION_CREDENTIALS environment variable to a service account key file\n"
                f"  3. Or use: gcloud auth login (for user credentials)\n\n"
                f"Original error: {error_msg}"
            )
        raise


def gcs_download_to_tmp(uri: str, suffix: str) -> str:
    """
    Downloads a single GCS object to a temp file and returns the local path.
    """
    if storage is None:
        raise RuntimeError(
            "google-cloud-storage not installed. Run: pip install google-cloud-storage"
        )
    
    bucket_name, blob_name = parse_gcs_uri(uri)
    
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(tmp_path)
    except Exception as e:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        error_msg = str(e)
        if "DefaultCredentialsError" in error_msg or "credentials" in error_msg.lower():
            raise RuntimeError(
                f"Authentication failed while downloading {uri}. "
                f"Please run: gcloud auth application-default login"
            )
        raise RuntimeError(f"Failed to download {uri}: {e}")
    
    return tmp_path


def should_keep_image(
    width: int,
    height: int,
    img_data: bytes,
    xref: int,
    min_side_px: int = 20,
    min_area_px: int = 10000,
    max_aspect_ratio: float = 20.0,
    doc_xrefs_seen: Optional[Set[int]] = None,
    global_hashes_seen: Optional[Set[str]] = None,
    dedupe_in_doc: bool = True,
    dedupe_global: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Determine if an image should be kept based on filtering criteria.
    
    Returns:
        (should_keep, skip_reason) where skip_reason is None if keeping the image
    """
    # A) Missing dimensions -> skip
    if width is None or height is None or width <= 0 or height <= 0:
        return False, "missing_dimensions"
    
    # B) Min side filter
    min_side = min(width, height)
    if min_side < min_side_px:
        return False, "min_side_too_small"
    
    # C) Area filter
    area = width * height
    if area < min_area_px:
        return False, "area_too_small"
    
    # D) Aspect ratio filter (catch 1px rules)
    max_side = max(width, height)
    aspect_ratio = max_side / max(1, min_side)
    if aspect_ratio > max_aspect_ratio:
        return False, "aspect_ratio_extreme"
    
    # E) Deduplicate within document (xref-based for PDFs)
    if dedupe_in_doc and doc_xrefs_seen is not None:
        if xref in doc_xrefs_seen:
            return False, "duplicate_in_doc"
        doc_xrefs_seen.add(xref)
    
    # F) Global deduplication (hash-based)
    if dedupe_global and global_hashes_seen is not None and img_data:
        img_hash = hashlib.sha256(img_data).hexdigest()
        if img_hash in global_hashes_seen:
            return False, "global_duplicate"
        global_hashes_seen.add(img_hash)
    
    return True, None


def count_images_in_pdf(
    pdf_path: str,
    apply_rgb_gray_filter: bool = True,
    fast: bool = False,
    apply_image_filters: bool = False,
    min_side_px: int = 20,
    min_area_px: int = 10000,
    max_aspect_ratio: float = 20.0,
    dedupe_in_doc: bool = True,
    dedupe_global: bool = False,
    global_hashes_seen: Optional[Set[str]] = None,
) -> Tuple[int, Dict[str, int]]:
    """
    Counts images in a PDF.

    fast=True counts len(page.get_images()) only (no pixmap creation).
    apply_rgb_gray_filter matches common ingest logic: pix.n - pix.alpha < 4.
    apply_image_filters enables filtering to reduce noise from logos/tiny icons/1px rules.
    
    Returns:
        (count, skip_stats) where skip_stats is a dict of skip reasons -> counts
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed. Run: pip install pymupdf")
    
    if apply_image_filters and Image is None:
        raise RuntimeError("PIL/Pillow not installed. Run: pip install pillow")

    doc = fitz.open(pdf_path)
    total = 0
    images_found = 0
    skip_stats = {
        "missing_dimensions": 0,
        "min_side_too_small": 0,
        "area_too_small": 0,
        "aspect_ratio_extreme": 0,
        "duplicate_in_doc": 0,
        "global_duplicate": 0,
    }
    
    # Track xrefs seen within this document for deduplication
    doc_xrefs_seen: Set[int] = set()
    
    try:
        for page in doc:
            imgs = page.get_images(full=True)
            images_found += len(imgs)
            
            if fast:
                # Fast mode: just count all images
                total += len(imgs)
                continue
            
            if not apply_rgb_gray_filter and not apply_image_filters:
                # No filters: count all images
                total += len(imgs)
                continue

            # Process each image with filters
            for img in imgs:
                xref = img[0]
                pix = None
                try:
                    pix = fitz.Pixmap(doc, xref)
                    
                    # RGB/GRAY filter (always applied if not fast)
                    if apply_rgb_gray_filter and (pix.n - pix.alpha) >= 4:
                        continue  # Skip CMYK/etc
                    
                    # If image filters are enabled, apply them
                    if apply_image_filters:
                        width = pix.width
                        height = pix.height
                        img_data = None
                        
                        # Get image bytes for global deduplication if needed
                        if dedupe_global:
                            img_data = pix.tobytes("png")
                        
                        should_keep, skip_reason = should_keep_image(
                            width=width,
                            height=height,
                            img_data=img_data,
                            xref=xref,
                            min_side_px=min_side_px,
                            min_area_px=min_area_px,
                            max_aspect_ratio=max_aspect_ratio,
                            doc_xrefs_seen=doc_xrefs_seen if dedupe_in_doc else None,
                            global_hashes_seen=global_hashes_seen if dedupe_global else None,
                            dedupe_in_doc=dedupe_in_doc,
                            dedupe_global=dedupe_global,
                        )
                        
                        if not should_keep:
                            skip_stats[skip_reason] = skip_stats.get(skip_reason, 0) + 1
                            continue
                    
                    # Image passed all filters
                    total += 1
                finally:
                    if pix is not None:
                        pix = None
    finally:
        doc.close()
    
    return total, skip_stats


def count_images_in_docx(docx_path: str) -> int:
    """
    Counts embedded images in a DOCX by counting files in word/media/.
    This counts unique embedded image *files* (not how many times an image is referenced).
    """
    total = 0
    with zipfile.ZipFile(docx_path, "r") as z:
        for name in z.namelist():
            if not name.startswith("word/media/"):
                continue
            ext = Path(name).suffix.lower()
            if ext in DOCX_IMAGE_EXTS:
                total += 1
    return total


def main():
    ap = argparse.ArgumentParser(
        description="Count images inside documents stored in a GCS bucket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python count_image.py gs://my-bucket/documents
  python count_image.py gs://my-bucket --csv output.csv
  python count_image.py gs://my-bucket --prefix some/folder/ --fast
  python count_image.py gs://my-bucket --no-pdf-filter --max-files 10
        """
    )
    ap.add_argument("bucket_uri", help="Root gs:// URI, e.g. gs://my-bucket or gs://my-bucket/prefix")
    ap.add_argument("--prefix", default="", help="Optional additional prefix under bucket_uri")
    ap.add_argument("--max-files", type=int, default=0, help="Process at most N files (0 = all)")
    ap.add_argument("--csv", default="", help="Write per-file counts to CSV")
    ap.add_argument("--fast", action="store_true", help="Faster PDF counting (no pixmap filter), counts raw get_images()")
    ap.add_argument("--no-pdf-filter", action="store_true", help="Do not apply RGB/GRAY filter when counting PDFs")
    ap.add_argument("--include-ext", default="", help="Comma-separated allowlist of extensions (e.g. pdf,docx)")
    ap.add_argument("--apply-image-filters", action="store_true", help="Apply image filtering to reduce noise (logos/tiny icons/1px rules)")
    ap.add_argument("--min-side-px", type=int, default=20, help="Minimum dimension (width or height) in pixels (default: 20)")
    ap.add_argument("--min-area-px", type=int, default=10000, help="Minimum area (width * height) in pixels (default: 10000)")
    ap.add_argument("--max-aspect-ratio", type=float, default=20.0, help="Maximum aspect ratio (max/min) to catch 1px rules (default: 20.0)")
    ap.add_argument("--dedupe-in-doc", action="store_true", default=True, help="Deduplicate images within the same document (default: True)")
    ap.add_argument("--no-dedupe-in-doc", action="store_false", dest="dedupe_in_doc", help="Disable document-level deduplication")
    args = ap.parse_args()
    
    # Ensure dedupe_in_doc defaults to True if not explicitly set
    if not hasattr(args, 'dedupe_in_doc'):
        args.dedupe_in_doc = True
    ap.add_argument("--dedupe-global", action="store_true", help="Deduplicate images across all documents in this run")
    ap.add_argument("--show-skip-stats", action="store_true", help="Show detailed breakdown of skipped images")
    args = ap.parse_args()

    # Parse the bucket URI
    try:
        bucket_name, prefix = parse_gcs_uri(args.bucket_uri)
        # Combine prefix from URI and --prefix argument if provided
        if args.prefix:
            prefix = f"{prefix.rstrip('/')}/{args.prefix.strip('/')}" if prefix else args.prefix.strip('/')
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    include_ext = None
    if args.include_ext.strip():
        include_ext = {("." + x.strip().lower().lstrip(".")) for x in args.include_ext.split(",") if x.strip()}

    print(f"Listing files in gs://{bucket_name}/{prefix}...")
    uris = gcs_ls_recursive(bucket_name, prefix)
    print(f"Found {len(uris)} files")
    
    if include_ext is not None:
        uris = [u for u in uris if Path(u).suffix.lower() in include_ext]

    # Default: only pdf/docx (since that's what you asked)
    if include_ext is None:
        uris = [u for u in uris if Path(u).suffix.lower() in {".pdf", ".docx"}]

    if args.max_files and args.max_files > 0:
        uris = uris[: args.max_files]

    print(f"Processing {len(uris)} files...\n")

    results: List[FileCount] = []
    total_images = 0
    total_images_found = 0
    total_pdfs = 0
    total_docx = 0
    total_skip_stats = {
        "missing_dimensions": 0,
        "min_side_too_small": 0,
        "area_too_small": 0,
        "aspect_ratio_extreme": 0,
        "duplicate_in_doc": 0,
        "global_duplicate": 0,
    }

    apply_pdf_filter = not args.no_pdf_filter
    global_hashes_seen: Set[str] = set() if args.dedupe_global else None
    
    for i, uri in enumerate(uris, 1):
        ext = Path(uri).suffix.lower()
        tmp = None
        try:
            tmp = gcs_download_to_tmp(uri, suffix=ext)

            if ext == ".pdf":
                if args.apply_image_filters:
                    c, skip_stats = count_images_in_pdf(
                        tmp,
                        apply_rgb_gray_filter=apply_pdf_filter,
                        fast=args.fast,
                        apply_image_filters=True,
                        min_side_px=args.min_side_px,
                        min_area_px=args.min_area_px,
                        max_aspect_ratio=args.max_aspect_ratio,
                        dedupe_in_doc=args.dedupe_in_doc,
                        dedupe_global=args.dedupe_global,
                        global_hashes_seen=global_hashes_seen,
                    )
                    # Aggregate skip stats
                    for reason, count in skip_stats.items():
                        total_skip_stats[reason] += count
                else:
                    c, skip_stats = count_images_in_pdf(
                        tmp,
                        apply_rgb_gray_filter=apply_pdf_filter,
                        fast=args.fast,
                        apply_image_filters=False
                    )
                    skip_stats = {}
                
                total_pdfs += 1
                note = "pdf_fast" if args.fast else ("pdf_filtered" if apply_pdf_filter else "pdf_raw")
                if args.apply_image_filters:
                    note += "+filters"
            elif ext == ".docx":
                c = count_images_in_docx(tmp)
                total_docx += 1
                note = "docx_word_media_files"
                skip_stats = {}
            else:
                continue

            total_images += c
            results.append(FileCount(
                uri=uri,
                ext=ext,
                image_count=c,
                note=note,
                images_skipped=skip_stats if args.apply_image_filters else None
            ))
            
            if args.apply_image_filters and args.show_skip_stats and skip_stats:
                skip_summary = ", ".join([f"{k}:{v}" for k, v in skip_stats.items() if v > 0])
                print(f"[{i}/{len(uris)}] {uri} -> {c} images kept (skipped: {skip_summary})")
            else:
                print(f"[{i}/{len(uris)}] {uri} -> {c} images")
        except Exception as e:
            results.append(FileCount(uri=uri, ext=ext, image_count=0, note=f"ERROR: {e}"))
            print(f"[{i}/{len(uris)}] {uri} -> ERROR: {e}", file=sys.stderr)
        finally:
            if tmp:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    print("\n=== SUMMARY ===")
    print(f"Files processed: {len(uris)} (pdf={total_pdfs}, docx={total_docx})")
    print(f"Total images counted: {total_images}")
    
    mode_parts = []
    if args.fast:
        mode_parts.append("FAST(raw get_images)")
    else:
        if apply_pdf_filter:
            mode_parts.append("RGB/GRAY FILTERED")
        else:
            mode_parts.append("RAW(no filter)")
    if args.apply_image_filters:
        mode_parts.append("IMAGE FILTERS ENABLED")
        print(f"Filter thresholds: min_side={args.min_side_px}px, min_area={args.min_area_px}px, max_aspect={args.max_aspect_ratio}")
        print(f"Deduplication: in_doc={args.dedupe_in_doc}, global={args.dedupe_global}")
    
    print(f"PDF mode: {' + '.join(mode_parts)}")
    
    if args.apply_image_filters and any(total_skip_stats.values()):
        print("\n=== IMAGE FILTER STATS ===")
        total_skipped = sum(total_skip_stats.values())
        print(f"Total images skipped: {total_skipped}")
        for reason, count in sorted(total_skip_stats.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {reason}: {count}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            headers = ["uri", "ext", "image_count", "note"]
            if args.apply_image_filters and args.show_skip_stats:
                headers.extend(["skipped_missing_dimensions", "skipped_min_side", "skipped_area", 
                              "skipped_aspect_ratio", "skipped_duplicate_in_doc", "skipped_global_duplicate"])
            w.writerow(headers)
            for r in results:
                row = [r.uri, r.ext, r.image_count, r.note]
                if args.apply_image_filters and args.show_skip_stats and r.images_skipped:
                    row.extend([
                        r.images_skipped.get("missing_dimensions", 0),
                        r.images_skipped.get("min_side_too_small", 0),
                        r.images_skipped.get("area_too_small", 0),
                        r.images_skipped.get("aspect_ratio_extreme", 0),
                        r.images_skipped.get("duplicate_in_doc", 0),
                        r.images_skipped.get("global_duplicate", 0),
                    ])
                w.writerow(row)
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
