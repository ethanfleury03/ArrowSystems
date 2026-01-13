#!/usr/bin/env python3
"""
Safe cleanup script for local ticket migration artifacts.

Removes logs and archives SQLite databases after successful migration to Cloud SQL.

Usage:
    # Dry-run (default) - show what would be cleaned
    python scripts/cleanup_local_tickets_artifacts.py --dry-run
    
    # Apply cleanup
    python scripts/cleanup_local_tickets_artifacts.py --apply
    
    # Force apply (skip git status check)
    python scripts/cleanup_local_tickets_artifacts.py --apply --force
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Get project root (where this script lives)
project_root = Path(__file__).parent.parent.resolve()

# Change to project root
os.chdir(project_root)

# Add project root to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def get_file_size(path: Path) -> int:
    """Get file size in bytes."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def check_git_status() -> Tuple[bool, str]:
    """Check if git repo is clean. Returns (is_clean, message)."""
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return False, "Git status check failed (not a git repo?)"
        
        if result.stdout.strip():
            return False, "Git repo has uncommitted changes"
        
        return True, "Git repo is clean"
    except FileNotFoundError:
        return False, "Git not found (skipping git status check)"


def find_artifacts() -> Dict[str, List[Tuple[Path, int]]]:
    """
    Find all ticket migration artifacts.
    
    Returns:
        Dict with keys: 'delete', 'archive', 'protect'
        Each value is a list of (path, size_bytes) tuples
    """
    artifacts = {
        'delete': [],  # Category A: logs, temp outputs
        'archive': [],  # Category B: SQLite DBs, dumps
        'protect': []  # Category C: source code, configs (for reference)
    }
    
    # Category A: Logs and temp outputs (DELETE)
    log_patterns = [
        'out/migrate*.log',
        'out/*.log',
        'out/migrate_dryrun.log',
    ]
    
    # Check out/ directory
    out_dir = project_root / 'out'
    if out_dir.exists():
        for log_file in out_dir.glob('*.log'):
            if log_file.is_file():
                size = get_file_size(log_file)
                artifacts['delete'].append((log_file, size))
    
    # Check for __pycache__ in scripts (optional cleanup)
    scripts_pycache = project_root / 'scripts' / '__pycache__'
    if scripts_pycache.exists() and scripts_pycache.is_dir():
        total_size = sum(
            f.stat().st_size
            for f in scripts_pycache.rglob('*')
            if f.is_file()
        )
        if total_size > 0:
            artifacts['delete'].append((scripts_pycache, total_size))
    
    # Category B: SQLite databases (ARCHIVE, do NOT delete)
    db_patterns = [
        'Scraper/data/tickets.db',
        'Scraper/data/*.db',
        '*.db',  # Any .db files in root (but we'll filter)
    ]
    
    # Check Scraper/data/tickets.db specifically
    tickets_db = project_root / 'Scraper' / 'data' / 'tickets.db'
    if tickets_db.exists() and tickets_db.is_file():
        size = get_file_size(tickets_db)
        artifacts['archive'].append((tickets_db, size))
    
    # Check for other .db files in Scraper/data/
    scraper_data = project_root / 'Scraper' / 'data'
    if scraper_data.exists():
        for db_file in scraper_data.glob('*.db'):
            if db_file.is_file() and db_file != tickets_db:
                size = get_file_size(db_file)
                artifacts['archive'].append((db_file, size))
    
    # Check for Postgres dumps (if any)
    for dump_pattern in ['*.sql', '*.dump', '*.pg_dump']:
        for dump_file in project_root.glob(dump_pattern):
            # Only archive if it looks like a ticket-related dump
            if dump_file.is_file() and 'ticket' in dump_file.name.lower():
                size = get_file_size(dump_file)
                artifacts['archive'].append((dump_file, size))
    
    # Category C: Protected files (for reference, don't touch)
    # These are listed so user knows they're protected
    protected_patterns = [
        'backend/scripts/migrate_tickets_sqlite_to_postgres.py',
        'backend/scripts/verify_tickets_parity.py',
        'backend/scripts/smoke_ticket_reads.py',
        'scripts/validate_tickets_pipeline.py',
        'scripts/cleanup_local_tickets_artifacts.py',
        'docs/TICKETS_VALIDATION.md',
        'backend/migrations/',
        'backend/.env',
    ]
    
    for pattern in protected_patterns:
        path = project_root / pattern
        if path.exists():
            if path.is_file():
                size = get_file_size(path)
                artifacts['protect'].append((path, size))
            elif path.is_dir():
                # Count total size of directory
                total_size = sum(
                    f.stat().st_size
                    for f in path.rglob('*')
                    if f.is_file()
                )
                if total_size > 0:
                    artifacts['protect'].append((path, total_size))
    
    return artifacts


def create_archive_dir(base_dir: Path = None) -> Path:
    """Create archive directory with timestamp."""
    if base_dir is None:
        base_dir = project_root / '.archive'
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_dir = base_dir / 'tickets_migration' / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


def print_plan(artifacts: Dict[str, List[Tuple[Path, int]]], archive_dir: Path = None):
    """Print cleanup plan."""
    print("=" * 70)
    print("TICKET MIGRATION ARTIFACTS CLEANUP PLAN")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print()
    
    # Category A: Delete
    if artifacts['delete']:
        print("[CATEGORY A] DELETE (logs, temp outputs)")
        print("-" * 70)
        total_delete_size = 0
        for path, size in artifacts['delete']:
            rel_path = path.relative_to(project_root)
            total_delete_size += size
            print(f"  DELETE: {rel_path} ({format_size(size)})")
        print(f"  Total to delete: {format_size(total_delete_size)}")
        print()
    else:
        print("[CATEGORY A] DELETE - No files found")
        print()
    
    # Category B: Archive
    if artifacts['archive']:
        print("[CATEGORY B] ARCHIVE (SQLite DBs, dumps)")
        print("-" * 70)
        total_archive_size = 0
        for path, size in artifacts['archive']:
            rel_path = path.relative_to(project_root)
            total_archive_size += size
            if archive_dir:
                archive_path = archive_dir / rel_path.name
                print(f"  ARCHIVE: {rel_path} ({format_size(size)})")
                print(f"           -> {archive_path.relative_to(project_root)}")
            else:
                print(f"  ARCHIVE: {rel_path} ({format_size(size)})")
        print(f"  Total to archive: {format_size(total_archive_size)}")
        print()
    else:
        print("[CATEGORY B] ARCHIVE - No files found")
        print()
    
    # Category C: Protect (informational)
    if artifacts['protect']:
        print("[CATEGORY C] PROTECTED (source code, configs)")
        print("-" * 70)
        print("  These files will NOT be touched:")
        for path, size in artifacts['protect'][:10]:  # Show first 10
            rel_path = path.relative_to(project_root)
            print(f"    {rel_path} ({format_size(size)})")
        if len(artifacts['protect']) > 10:
            print(f"    ... and {len(artifacts['protect']) - 10} more protected files")
        print()
    
    # Summary
    total_delete = sum(size for _, size in artifacts['delete'])
    total_archive = sum(size for _, size in artifacts['archive'])
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files to DELETE: {len(artifacts['delete'])} ({format_size(total_delete)})")
    print(f"Files to ARCHIVE: {len(artifacts['archive'])} ({format_size(total_archive)})")
    print(f"Protected files: {len(artifacts['protect'])}")
    print("=" * 70)


def execute_cleanup(
    artifacts: Dict[str, List[Tuple[Path, int]]],
    archive_dir: Path,
    dry_run: bool = True
):
    """Execute cleanup operations."""
    if dry_run:
        print("\n[DRY-RUN MODE] No files will be modified")
        print("   Run with --apply to execute cleanup")
        return
    
    print("\n[EXECUTING CLEANUP]...")
    print()
    
    # Delete Category A files
    deleted_count = 0
    deleted_size = 0
    for path, size in artifacts['delete']:
        try:
            if path.is_file():
                path.unlink()
                deleted_count += 1
                deleted_size += size
                rel_path = path.relative_to(project_root)
                print(f"  [OK] Deleted: {rel_path}")
            elif path.is_dir():
                shutil.rmtree(path)
                deleted_count += 1
                deleted_size += size
                rel_path = path.relative_to(project_root)
                print(f"  [OK] Deleted directory: {rel_path}")
        except Exception as e:
            rel_path = path.relative_to(project_root)
            print(f"  [ERROR] Error deleting {rel_path}: {e}")
    
    # Archive Category B files
    archived_count = 0
    archived_size = 0
    for path, size in artifacts['archive']:
        try:
            # Create subdirectory structure in archive if needed
            rel_path = path.relative_to(project_root)
            archive_path = archive_dir / rel_path.name
            
            # Ensure parent directory exists
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy to archive
            shutil.copy2(path, archive_path)
            
            # Delete original
            path.unlink()
            
            archived_count += 1
            archived_size += size
            print(f"  [OK] Archived: {rel_path} -> {archive_path.relative_to(project_root)}")
        except Exception as e:
            rel_path = path.relative_to(project_root)
            print(f"  [ERROR] Error archiving {rel_path}: {e}")
    
    # Summary
    print()
    print("=" * 70)
    print("CLEANUP COMPLETE")
    print("=" * 70)
    print(f"Deleted: {deleted_count} files ({format_size(deleted_size)})")
    print(f"Archived: {archived_count} files ({format_size(archived_size)})")
    print(f"Archive location: {archive_dir.relative_to(project_root)}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Clean up local ticket migration artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry-run mode: show what would be cleaned (default)"
    )
    
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply cleanup (default is dry-run)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force apply even if git repo has uncommitted changes"
    )
    
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="Archive directory (default: .archive/tickets_migration/YYYYMMDD_HHMMSS/)"
    )
    
    args = parser.parse_args()
    
    # Determine if this is dry-run or apply
    dry_run = not args.apply
    
    # Check git status if applying
    if not dry_run and not args.force:
        is_clean, message = check_git_status()
        if not is_clean:
            print("❌ ERROR: Git repo has uncommitted changes")
            print(f"   {message}")
            print("   Commit or stash changes first, or use --force to override")
            sys.exit(1)
    
    # Find artifacts
    artifacts = find_artifacts()
    
    # Create archive directory if needed
    archive_dir = None
    if not dry_run or artifacts['archive']:
        archive_dir = args.archive_dir or create_archive_dir()
    
    # Print plan
    print_plan(artifacts, archive_dir)
    
    # Execute cleanup
    if not dry_run:
        print("\n[WARNING] This will DELETE and ARCHIVE files!")
        print("   Press Ctrl+C to cancel, or wait 3 seconds to continue...")
        try:
            import time
            time.sleep(3)
        except KeyboardInterrupt:
            print("\n[INFO] Cleanup cancelled by user")
            sys.exit(1)
    
    execute_cleanup(artifacts, archive_dir, dry_run=dry_run)
    
    if dry_run:
        print("\n[INFO] To apply cleanup, run:")
        print("   python scripts/cleanup_local_tickets_artifacts.py --apply")


if __name__ == "__main__":
    main()
