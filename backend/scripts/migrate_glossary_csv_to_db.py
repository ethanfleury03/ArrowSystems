"""
Migration Helper Script: Move glossary.csv to Database

This is a ONE-OFF migration script to move data from data/glossary.csv
into the glossary_terms table in PostgreSQL.

Usage:
    python -m backend.scripts.migrate_glossary_csv_to_db [--csv-file data/glossary.csv]

This script should ONLY be run once during the migration to GCP.
It is NOT used at runtime.

DO NOT import or use this script in production code.
"""

import os
import sys
import csv
import logging
from pathlib import Path
from typing import List, Dict

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.utils.db import SessionLocal, GlossaryTerm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CSV_FILE = Path("data/glossary.csv")


def migrate_csv_to_db(
    csv_file: Path = DEFAULT_CSV_FILE,
    dry_run: bool = False
) -> Dict[str, any]:
    """
    Migrate glossary terms from CSV file to database.
    
    Args:
        csv_file: Path to glossary.csv file
        dry_run: If True, don't actually write to database
    
    Returns:
        Dictionary with migration statistics
    """
    if not csv_file.exists():
        logger.warning(f"CSV file not found: {csv_file}")
        logger.info("No existing glossary to migrate. This is OK if starting fresh.")
        return {
            "success": True,
            "migrated": 0,
            "skipped": 0,
            "errors": 0,
            "message": "No CSV file found - starting fresh"
        }
    
    logger.info(f"Loading glossary from {csv_file}")
    
    session = SessionLocal()
    stats = {
        "migrated": 0,
        "skipped": 0,
        "errors": 0,
        "error_details": []
    }
    
    try:
        with open(csv_file, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    term = (row.get('term') or '').strip()
                    definition = (row.get('definition') or '').strip()
                    aliases_raw = (row.get('aliases') or '').strip()
                    
                    if not term or not definition:
                        logger.debug(f"Skipping row with missing term or definition: {row}")
                        stats["skipped"] += 1
                        continue
                    
                    # Parse aliases (pipe-separated)
                    aliases = [a.strip() for a in aliases_raw.split('|') if a.strip()] if aliases_raw else []
                    
                    # Check if term already exists
                    existing = session.query(GlossaryTerm).filter(
                        GlossaryTerm.term == term
                    ).first()
                    
                    if existing:
                        logger.debug(f"Skipping {term} - already exists in database")
                        stats["skipped"] += 1
                        continue
                    
                    # Create glossary term record
                    glossary_term = GlossaryTerm(
                        term=term,
                        definition=definition,
                        aliases=aliases if aliases else None  # Store as JSON array or None
                    )
                    
                    if not dry_run:
                        session.add(glossary_term)
                        session.commit()
                        logger.info(f"Migrated: {term}")
                    else:
                        logger.info(f"[DRY RUN] Would migrate: {term}")
                    
                    stats["migrated"] += 1
                    
                except Exception as e:
                    logger.error(f"Error migrating term from row {row}: {e}", exc_info=True)
                    stats["errors"] += 1
                    stats["error_details"].append({
                        "row": row,
                        "error": str(e)
                    })
                    session.rollback()
        
        if not dry_run:
            session.commit()
        
        logger.info(f"Migration complete: {stats['migrated']} migrated, {stats['skipped']} skipped, {stats['errors']} errors")
        
        return {
            "success": stats["errors"] == 0,
            **stats
        }
        
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}", exc_info=True)
        return {
            "success": False,
            "migrated": 0,
            "skipped": 0,
            "errors": 1,
            "message": f"Failed to read CSV: {e}"
        }
    finally:
        session.close()


def main():
    """Main entry point for migration script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate glossary.csv to PostgreSQL glossary_terms table"
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        default=DEFAULT_CSV_FILE,
        help=f"Path to glossary.csv (default: {DEFAULT_CSV_FILE})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually write to database, just show what would be migrated"
    )
    
    args = parser.parse_args()
    
    if not args.dry_run:
        logger.info("Starting glossary migration from CSV to database...")
    else:
        logger.info("[DRY RUN MODE] No changes will be made to the database")
    
    result = migrate_csv_to_db(
        csv_file=args.csv_file,
        dry_run=args.dry_run
    )
    
    if result["success"]:
        logger.info("✅ Migration completed successfully")
        sys.exit(0)
    else:
        logger.error(f"❌ Migration completed with {result['errors']} errors")
        if result.get("error_details"):
            for detail in result["error_details"][:10]:  # Show first 10 errors
                logger.error(f"  - {detail.get('row', {}).get('term', 'unknown')}: {detail['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()











