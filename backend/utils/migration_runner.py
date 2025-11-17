"""
Migration runner utility for managing Alembic database migrations.

Handles dev vs prod behavior:
- Dev: Auto-runs migrations on startup
- Prod: Fails fast if migrations are pending (must run manually)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text
import time

from .db import get_engine, _is_sqlite, DATABASE_URL
from ..config.env import settings

logger = logging.getLogger(__name__)

# Cache for migration status to avoid repeated Alembic context creation
_migration_status_cache: Optional[dict] = None
_migration_status_cache_time: float = 0
_migration_status_cache_ttl: float = 60.0  # Cache for 60 seconds


def get_alembic_config() -> Config:
    """Get Alembic configuration."""
    # Get the migrations directory path
    # migration_runner.py is in backend/utils/
    # migrations/ is in backend/migrations/
    backend_dir = Path(__file__).resolve().parent.parent
    migrations_dir = backend_dir / "migrations"
    alembic_ini_path = migrations_dir / "alembic.ini"
    
    # Try alternative paths (for Docker or different working directories)
    if not alembic_ini_path.exists():
        # Try from current working directory (Docker case: /app/backend/migrations/)
        alt_paths = [
            Path("backend/migrations/alembic.ini"),
            Path("./backend/migrations/alembic.ini"),
            Path.cwd() / "backend" / "migrations" / "alembic.ini",
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                alembic_ini_path = alt_path.resolve()
                break
        else:
            raise RuntimeError(
                f"Alembic configuration not found. Tried: {alembic_ini_path}, {alt_paths}. "
                f"Current working directory: {Path.cwd()}"
            )
    else:
        alembic_ini_path = alembic_ini_path.resolve()
    
    config = Config(str(alembic_ini_path))
    # Set the database URL
    config.set_main_option("sqlalchemy.url", DATABASE_URL)
    # Set script_location to absolute path of migrations directory
    # This ensures Alembic can find env.py regardless of working directory
    migrations_dir_abs = alembic_ini_path.parent.resolve()
    config.set_main_option("script_location", str(migrations_dir_abs))
    
    return config


def get_current_revision() -> Optional[str]:
    """
    Get the current database revision.
    
    Returns:
        Current revision string, or None if no migrations have been run
    """
    try:
        engine = get_engine()
        with engine.connect() as connection:
            # Suppress Alembic's verbose logging for routine checks
            import logging as std_logging
            alembic_logger = std_logging.getLogger('alembic.runtime.migration')
            original_level = alembic_logger.level
            alembic_logger.setLevel(std_logging.WARNING)  # Suppress INFO messages
            
            try:
                context = MigrationContext.configure(connection)
                current_rev = context.get_current_revision()
                return current_rev
            finally:
                alembic_logger.setLevel(original_level)
    except Exception as e:
        logger.warning(f"Could not get current revision: {e}")
        return None


def get_head_revision() -> Optional[str]:
    """
    Get the head (latest) revision from migration scripts.
    
    Returns:
        Head revision string, or None if no migrations exist
    """
    try:
        config = get_alembic_config()
        script = ScriptDirectory.from_config(config)
        head = script.get_current_head()
        return head
    except Exception as e:
        logger.warning(f"Could not get head revision: {e}")
        return None


def check_pending_migrations() -> bool:
    """
    Check if there are pending migrations.
    
    Returns:
        True if migrations are pending, False otherwise
    """
    current = get_current_revision()
    head = get_head_revision()
    
    if head is None:
        # No migrations exist
        return False
    
    if current is None:
        # Database has no migrations applied
        return True
    
    return current != head


def _has_existing_tables() -> bool:
    """Check if database has existing tables (but no Alembic version tracking)."""
    try:
        engine = get_engine()
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        # Check if we have our main tables but no alembic_version table
        has_main_tables = any(table in tables for table in ['users', 'query_history', 'feedback'])
        has_alembic_version = 'alembic_version' in tables
        return has_main_tables and not has_alembic_version
    except Exception:
        return False


def _database_has_schema_fixes() -> bool:
    """Check if database already has the schema fixes from 002_schema_fixes applied."""
    try:
        engine = get_engine()
        inspector = inspect(engine)
        
        # Check if updated_at columns exist (from 002_schema_fixes)
        feedback_columns = {col["name"] for col in inspector.get_columns("feedback")} if "feedback" in inspector.get_table_names() else set()
        query_history_columns = {col["name"] for col in inspector.get_columns("query_history")} if "query_history" in inspector.get_table_names() else set()
        
        has_updated_at = "updated_at" in feedback_columns and "updated_at" in query_history_columns
        
        # Check if indexes exist (from 002_schema_fixes)
        indexes = set()
        for table_name in inspector.get_table_names():
            try:
                table_indexes = inspector.get_indexes(table_name)
                indexes.update({idx["name"] for idx in table_indexes})
            except Exception:
                continue
        
        has_indexes = "ix_query_history_query_text" in indexes
        
        return has_updated_at and has_indexes
    except Exception:
        return False


def _stamp_existing_database() -> bool:
    """
    Stamp an existing database with the appropriate migration revision.
    If the database already has schema fixes applied, stamp at head.
    Otherwise, stamp at initial migration.
    
    Returns:
        True if stamping was successful, False otherwise
    """
    try:
        config = get_alembic_config()
        head_rev = get_head_revision()
        
        # If database already has schema fixes, stamp at head to skip migrations
        if _database_has_schema_fixes() and head_rev:
            logger.info(f"Database appears to have schema fixes already. Stamping at head: {head_rev}")
            command.stamp(config, head_rev)
            return True
        else:
            # Otherwise, stamp at initial migration
            initial_rev = '001_initial'
            logger.info(f"Stamping existing database with initial revision: {initial_rev}")
            command.stamp(config, initial_rev)
            return True
    except Exception as e:
        logger.warning(f"Failed to stamp database: {e}")
        return False


def run_migrations() -> Tuple[bool, str]:
    """
    Run pending migrations.
    
    Handles existing databases by stamping them first if needed.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        config = get_alembic_config()
        current_before = get_current_revision()
        
        # If database has existing tables but no Alembic version, stamp it first
        if current_before is None and _has_existing_tables():
            logger.info("Detected existing database without Alembic tracking. Stamping...")
            if _stamp_existing_database():
                current_before = get_current_revision()
                logger.info(f"Database stamped at revision: {current_before}")
            else:
                # If stamping fails, we'll try to run migrations anyway
                # The migration should handle existing tables gracefully
                logger.warning("Stamping failed, attempting migrations anyway...")
        
        # Quick check: if already at head, skip the upgrade (fast path)
        head_rev = get_head_revision()
        if current_before == head_rev and current_before is not None:
            logger.debug("Database is already at the latest migration, skipping upgrade")
            return True, "Database is up to date"
        
        # If database is at 001_initial but already has schema fixes, stamp at head to skip migration
        if current_before == '001_initial' and _database_has_schema_fixes() and head_rev:
            logger.info("Database is at 001_initial but already has schema fixes. Stamping at head to skip migration...")
            command.stamp(config, head_rev)
            return True, f"Database stamped at {head_rev} (schema fixes already applied)"
        
        # Only run upgrade if there are pending migrations
        logger.info("Running database migrations...")
        command.upgrade(config, "head")
        
        current_after = get_current_revision()
        
        if current_before != current_after:
            logger.info(
                f"Migration completed: {current_before or 'none'} -> {current_after}"
            )
            return True, f"Migrated from {current_before or 'none'} to {current_after}"
        else:
            logger.info("Database is already at the latest migration")
            return True, "Database is up to date"
            
    except Exception as e:
        error_msg = f"Migration failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg


def check_migration_status(use_cache: bool = True) -> dict:
    """
    Get detailed migration status information.
    
    Args:
        use_cache: If True, use cached result if available (default: True)
    
    Returns:
        Dictionary with migration status details
    """
    global _migration_status_cache, _migration_status_cache_time
    
    # Return cached result if available and fresh
    if use_cache and _migration_status_cache is not None:
        cache_age = time.time() - _migration_status_cache_time
        if cache_age < _migration_status_cache_ttl:
            return _migration_status_cache
    
    # Get fresh status
    current = get_current_revision()
    head = get_head_revision()
    pending = check_pending_migrations()
    
    status = {
        "current_revision": current,
        "head_revision": head,
        "pending_migrations": pending,
        "database_type": "sqlite" if _is_sqlite(DATABASE_URL) else "postgresql",
    }
    
    # Cache the result
    _migration_status_cache = status
    _migration_status_cache_time = time.time()
    
    return status


def stamp_database(revision: str = "head") -> Tuple[bool, str]:
    """
    Manually stamp the database with a specific revision.
    Useful for skipping migrations when database is already set up correctly.
    
    Args:
        revision: Revision to stamp (default: "head")
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        config = get_alembic_config()
        logger.info(f"Stamping database with revision: {revision}")
        command.stamp(config, revision)
        return True, f"Database stamped at {revision}"
    except Exception as e:
        error_msg = f"Failed to stamp database: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg


def main():
    """CLI entry point for migration commands."""
    if len(sys.argv) < 2:
        print("Usage: python -m backend.utils.migration_runner <command> [args]")
        print("Commands:")
        print("  upgrade          - Run pending migrations")
        print("  status           - Show migration status")
        print("  check            - Check if migrations are pending")
        print("  stamp [revision] - Manually stamp database (default: head)")
        sys.exit(1)
    
    command_name = sys.argv[1].lower()
    
    if command_name == "upgrade":
        success, message = run_migrations()
        if success:
            print(f"✅ {message}")
            sys.exit(0)
        else:
            print(f"❌ {message}")
            sys.exit(1)
    
    elif command_name == "status":
        status = check_migration_status()
        print(f"Current revision: {status['current_revision'] or 'none'}")
        print(f"Head revision: {status['head_revision'] or 'none'}")
        print(f"Pending migrations: {status['pending_migrations']}")
        print(f"Database type: {status['database_type']}")
        sys.exit(0)
    
    elif command_name == "check":
        pending = check_pending_migrations()
        if pending:
            print("⚠️  Pending migrations detected")
            sys.exit(1)
        else:
            print("✅ Database is up to date")
            sys.exit(0)
    
    elif command_name == "stamp":
        revision = sys.argv[2] if len(sys.argv) > 2 else "head"
        success, message = stamp_database(revision)
        if success:
            print(f"✅ {message}")
            sys.exit(0)
        else:
            print(f"❌ {message}")
            sys.exit(1)
    
    else:
        print(f"Unknown command: {command_name}")
        sys.exit(1)


if __name__ == "__main__":
    main()

