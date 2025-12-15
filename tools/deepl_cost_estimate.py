#!/usr/bin/env python3
"""
DeepL API Cost Estimator for PDF Document Translation

This script estimates the cost of translating PDF documents using DeepL API Pro
before performing the actual translation. It extracts text from PDFs locally
to calculate character counts and applies DeepL's billing rules.

Installation:
    pip install pymupdf

Usage:
    python tools/deepl_cost_estimate.py --input manuals/en --langs PL,DE,ES --out deepl_cost_estimate.csv
    
    python tools/deepl_cost_estimate.py --input manuals/en --langs PL,DE,ES --price-per-mchar 25 --monthly-base 5.49 --recursive --out deepl_cost_estimate.csv

DeepL API Pro Pricing:
    - $25 per 1,000,000 characters (pay-as-you-go)
    - Minimum 50,000 characters billed per document per target language
    - Maximum 30 MB file size limit
    - Maximum 1,000,000 characters per document (flagged if exceeded)

Author: Auto-generated
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Install with: pip install pymupdf")
    sys.exit(1)


def extract_text_from_pdf(pdf_path: Path) -> Tuple[int, Optional[str]]:
    """
    Extract text from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Tuple of (extracted_char_count, error_message)
        If successful, error_message is None
    """
    try:
        doc = fitz.open(str(pdf_path))
        total_chars = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            total_chars += len(text)
        
        doc.close()
        return total_chars, None
    
    except Exception as e:
        return 0, f"Error extracting text: {str(e)}"


def process_pdf(
    pdf_path: Path,
    langs: List[str],
    price_per_mchar: float,
    pdf_mb_limit: float,
    doc_char_limit: int,
    min_billed_chars: int,
) -> Dict:
    """
    Process a single PDF file and calculate cost estimates.
    
    Args:
        pdf_path: Path to the PDF file
        langs: List of target language codes
        price_per_mchar: Price per million characters
        pdf_mb_limit: Maximum PDF size in MB
        doc_char_limit: Maximum characters per document (for flagging)
        min_billed_chars: Minimum characters billed per document per language
        
    Returns:
        Dictionary with file information and cost estimates
    """
    file_name = pdf_path.name
    file_path = str(pdf_path)
    
    # Get file size in MB
    size_bytes = pdf_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    
    # Extract text
    extracted_chars, error_msg = extract_text_from_pdf(pdf_path)
    
    # Initialize result dictionary
    result = {
        "file_path": file_path,
        "file_name": file_name,
        "size_mb": round(size_mb, 2),
        "extracted_chars": extracted_chars,
        "billed_chars_per_lang": 0,
        "langs": ",".join(langs),
        "billed_chars_total": 0,
        "est_cost_usd": 0.0,
        "status": "OK",
        "notes": "",
    }
    
    # Check for extraction errors
    if error_msg:
        result["status"] = "ERROR"
        result["notes"] = error_msg
        return result
    
    # Check if file exceeds size limit
    if size_mb > pdf_mb_limit:
        result["status"] = "SKIP_OVERSIZE"
        result["notes"] = f"File size {size_mb:.2f} MB exceeds limit of {pdf_mb_limit} MB"
        return result
    
    # Check if extracted characters exceed document limit
    if extracted_chars > doc_char_limit:
        result["status"] = "FLAG_OVER_CHAR_LIMIT"
        result["notes"] = f"Extracted {extracted_chars:,} characters exceeds limit of {doc_char_limit:,}"
        # Still calculate cost for flagged files
        billed_chars_per_lang = max(extracted_chars, min_billed_chars)
    else:
        # Apply minimum billing rule
        billed_chars_per_lang = max(extracted_chars, min_billed_chars)
    
    # Calculate total billed characters (per language * number of languages)
    num_langs = len(langs)
    billed_chars_total = billed_chars_per_lang * num_langs
    
    # Calculate estimated cost
    est_cost_usd = (billed_chars_total / 1_000_000) * price_per_mchar
    
    result["billed_chars_per_lang"] = billed_chars_per_lang
    result["billed_chars_total"] = billed_chars_total
    result["est_cost_usd"] = round(est_cost_usd, 2)
    
    return result


def find_pdf_files(input_dir: Path, recursive: bool = False) -> List[Path]:
    """
    Find all PDF files in the input directory.
    
    Args:
        input_dir: Directory to search
        recursive: If True, search subdirectories
        
    Returns:
        List of PDF file paths
    """
    pdf_files = []
    
    if recursive:
        pattern = "**/*.pdf"
    else:
        pattern = "*.pdf"
    
    for pdf_path in input_dir.glob(pattern):
        if pdf_path.is_file():
            pdf_files.append(pdf_path)
    
    return sorted(pdf_files)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate DeepL API costs for PDF document translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/deepl_cost_estimate.py --input manuals/en --langs PL,DE,ES --out deepl_cost_estimate.csv
  python tools/deepl_cost_estimate.py --input manuals/en --langs PL,DE,ES --recursive --workers 4 --out deepl_cost_estimate.csv
        """,
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing PDF files",
    )
    
    parser.add_argument(
        "--langs",
        type=str,
        required=True,
        help="Comma-separated list of target language codes (e.g., PL,DE,ES)",
    )
    
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output CSV file path",
    )
    
    parser.add_argument(
        "--price-per-mchar",
        type=float,
        default=25.0,
        help="Price per million characters (default: 25.0)",
    )
    
    parser.add_argument(
        "--monthly-base",
        type=float,
        default=5.49,
        help="Monthly base subscription cost (default: 5.49)",
    )
    
    parser.add_argument(
        "--pdf-mb-limit",
        type=float,
        default=30.0,
        help="Maximum PDF file size in MB (default: 30.0)",
    )
    
    parser.add_argument(
        "--doc-char-limit",
        type=int,
        default=1_000_000,
        help="Maximum characters per document for flagging (default: 1000000)",
    )
    
    parser.add_argument(
        "--min-billed-chars",
        type=int,
        default=50_000,
        help="Minimum characters billed per document per language (default: 50000)",
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers for PDF processing (default: 2)",
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories",
    )
    
    args = parser.parse_args()
    
    # Parse languages
    langs = [lang.strip().upper() for lang in args.langs.split(",") if lang.strip()]
    if not langs:
        print("ERROR: No valid language codes provided")
        sys.exit(1)
    
    # Validate input directory
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        print(f"\nTip: Use an existing directory. Common options:")
        # Suggest common directories if they exist
        common_dirs = ["data", "manuals", "docs"]
        existing = [d for d in common_dirs if Path(d).exists() and Path(d).is_dir()]
        if existing:
            print(f"  - {', '.join(existing)}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"ERROR: Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    # Find PDF files
    print(f"Scanning for PDF files in: {input_dir}")
    if args.recursive:
        print("  (recursive mode enabled)")
    
    pdf_files = find_pdf_files(input_dir, recursive=args.recursive)
    
    if not pdf_files:
        print(f"WARNING: No PDF files found in {input_dir}")
        sys.exit(0)
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    print(f"Processing with {args.workers} worker(s)...")
    print()
    
    # Process PDFs in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_pdf = {
            executor.submit(
                process_pdf,
                pdf_path,
                langs,
                args.price_per_mchar,
                args.pdf_mb_limit,
                args.doc_char_limit,
                args.min_billed_chars,
            ): pdf_path
            for pdf_path in pdf_files
        }
        
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                result = future.result()
                results.append(result)
                status_icon = "✓" if result["status"] == "OK" else "⚠" if result["status"] == "FLAG_OVER_CHAR_LIMIT" else "✗"
                print(f"{status_icon} {pdf_path.name}: {result['status']} ({result['extracted_chars']:,} chars)")
            except Exception as e:
                error_result = {
                    "file_path": str(pdf_path),
                    "file_name": pdf_path.name,
                    "size_mb": 0.0,
                    "extracted_chars": 0,
                    "billed_chars_per_lang": 0,
                    "langs": ",".join(langs),
                    "billed_chars_total": 0,
                    "est_cost_usd": 0.0,
                    "status": "ERROR",
                    "notes": f"Processing error: {str(e)}",
                }
                results.append(error_result)
                print(f"✗ {pdf_path.name}: ERROR - {str(e)}")
    
    # Sort results: by status (OK first), then by estimated cost descending
    status_order = {"OK": 0, "FLAG_OVER_CHAR_LIMIT": 1, "SKIP_OVERSIZE": 2, "ERROR": 3}
    results.sort(
        key=lambda x: (status_order.get(x["status"], 99), -x["est_cost_usd"])
    )
    
    # Write CSV output
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "file_path",
        "file_name",
        "size_mb",
        "extracted_chars",
        "billed_chars_per_lang",
        "langs",
        "billed_chars_total",
        "est_cost_usd",
        "status",
        "notes",
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print()
    print(f"Results written to: {output_path}")
    print()
    
    # Calculate summary statistics
    total_files = len(results)
    ok_files = sum(1 for r in results if r["status"] == "OK")
    skipped_oversize = sum(1 for r in results if r["status"] == "SKIP_OVERSIZE")
    flagged_over_char_limit = sum(1 for r in results if r["status"] == "FLAG_OVER_CHAR_LIMIT")
    error_files = sum(1 for r in results if r["status"] == "ERROR")
    
    # Calculate totals for OK files only
    total_billed_chars_ok = sum(
        r["billed_chars_total"] for r in results if r["status"] == "OK"
    )
    estimated_total_cost_ok = (
        (total_billed_chars_ok / 1_000_000) * args.price_per_mchar
    ) + args.monthly_base
    
    # Calculate per-language breakdown
    lang_breakdown = {}
    for lang in langs:
        lang_billed_chars = sum(
            r["billed_chars_per_lang"] for r in results if r["status"] == "OK"
        )
        lang_cost = (lang_billed_chars / 1_000_000) * args.price_per_mchar
        lang_breakdown[lang] = {
            "chars": lang_billed_chars,
            "cost": lang_cost,
        }
    
    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files processed:     {total_files}")
    print(f"  ✓ OK files:             {ok_files}")
    print(f"  ⚠ Flagged (over limit): {flagged_over_char_limit}")
    print(f"  ✗ Skipped (oversize):   {skipped_oversize}")
    print(f"  ✗ Errors:               {error_files}")
    print()
    
    if ok_files > 0:
        print(f"Total billed characters (OK files): {total_billed_chars_ok:,}")
        print(f"Estimated total cost (OK files):    ${estimated_total_cost_ok:.2f}")
        print(f"  (includes ${args.monthly_base:.2f} monthly base)")
        print()
        
        print("Per-language breakdown (OK files):")
        for lang, data in lang_breakdown.items():
            print(f"  {lang}: {data['chars']:,} chars = ${data['cost']:.2f}")
        print()
    
    if flagged_over_char_limit > 0:
        flagged_cost = sum(
            r["est_cost_usd"] for r in results if r["status"] == "FLAG_OVER_CHAR_LIMIT"
        )
        print(f"⚠ WARNING: {flagged_over_char_limit} file(s) exceed character limit")
        print(f"  Additional cost for flagged files: ${flagged_cost:.2f}")
        print()
    
    if skipped_oversize > 0:
        print(f"✗ {skipped_oversize} file(s) skipped due to size limit ({args.pdf_mb_limit} MB)")
        print()
    
    print("=" * 70)


if __name__ == "__main__":
    main()

