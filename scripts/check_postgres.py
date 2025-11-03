#!/usr/bin/env python3
"""
Database Connection Checker and Setup Helper
Validates PostgreSQL or SQLite connection and environment variables
Supports automatic fallback from PostgreSQL to SQLite
"""

import os
import sys
import logging
from typing import Dict, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_db_config() -> Dict[str, str]:
    """Get database configuration from environment variables"""
    # Support both POSTGRES_* and DB_* naming conventions
    config = {
        'host': os.getenv('POSTGRES_HOST') or os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT') or os.getenv('DB_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB') or os.getenv('DB_NAME', 'rag_app'),
        'user': os.getenv('POSTGRES_USER') or os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD') or os.getenv('DB_PASSWORD', ''),
    }
    return config

def check_environment_variables() -> tuple[bool, list[str]]:
    """Check if required environment variables are set"""
    issues = []
    
    # Check if PostgreSQL credentials are provided
    has_postgres_creds = any([
        os.getenv('POSTGRES_HOST'),
        os.getenv('POSTGRES_USER'),
        os.getenv('POSTGRES_PASSWORD'),
        os.getenv('DB_HOST'),
        os.getenv('DB_USER'),
        os.getenv('DB_PASSWORD')
    ])
    
    if has_postgres_creds:
        password = os.getenv('POSTGRES_PASSWORD') or os.getenv('DB_PASSWORD')
        if not password:
            issues.append("[WARN] PostgreSQL password is not set - will fall back to SQLite")
        elif password == 'password':
            issues.append("[WARN] Using default password 'password' - consider changing for production")
        
        config = get_db_config()
        logger.info("\n[INFO] PostgreSQL Configuration (if available):")
        logger.info(f"   Host: {config['host']}")
        logger.info(f"   Port: {config['port']}")
        logger.info(f"   Database: {config['database']}")
        logger.info(f"   User: {config['user']}")
        logger.info(f"   Password: {'*' * len(config['password']) if config['password'] else 'NOT SET'}")
    else:
        logger.info("\n[INFO] No PostgreSQL credentials found - will use SQLite")
        sqlite_path = os.getenv('SQLITE_DB_PATH', 'rag_app.db')
        logger.info(f"   SQLite database: {sqlite_path}")
    
    return True, issues

def test_connection() -> tuple[bool, Optional[str], Optional[str]]:
    """
    Test database connection - tries PostgreSQL first, falls back to SQLite
    Returns: (success, error_message, db_type)
    """
    # Try PostgreSQL first
    try:
        import psycopg2
        
        config = get_db_config()
        
        # Skip if no explicit PostgreSQL credentials
        has_creds = any([
            os.getenv('POSTGRES_HOST'),
            os.getenv('POSTGRES_USER'),
            os.getenv('POSTGRES_PASSWORD'),
            os.getenv('DB_HOST'),
            os.getenv('DB_USER'),
            os.getenv('DB_PASSWORD')
        ])
        
        if has_creds:
            logger.info("\n[TEST] Testing PostgreSQL connection...")
            
            try:
                conn = psycopg2.connect(
                    host=config['host'],
                    port=config['port'],
                    database=config['database'],
                    user=config['user'],
                    password=config['password'],
                    connect_timeout=5
                )
                
                cursor = conn.cursor()
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                
                cursor.execute("SELECT current_database();")
                db_name = cursor.fetchone()[0]
                
                cursor.close()
                conn.close()
                
                logger.info(f"[OK] PostgreSQL connection successful!")
                logger.info(f"   PostgreSQL Version: {version.split(',')[0]}")
                logger.info(f"   Connected to database: {db_name}")
                
                return True, None, 'postgres'
                
            except psycopg2.OperationalError as e:
                error_msg = str(e)
                logger.debug(f"PostgreSQL connection failed: {error_msg}")
                logger.info("   Will fall back to SQLite")
            except Exception as e:
                logger.debug(f"PostgreSQL connection failed: {e}")
                logger.info("   Will fall back to SQLite")
        else:
            logger.info("\n[TEST] No PostgreSQL credentials - will use SQLite")
    
    except ImportError:
        logger.debug("psycopg2 not installed - will use SQLite")
    
    # Fall back to SQLite
    try:
        import sqlite3
        
        logger.info("\n[TEST] Testing SQLite connection...")
        
        db_path = os.getenv('SQLITE_DB_PATH', 'rag_app.db')
        conn = sqlite3.connect(db_path, check_same_thread=False)
        
        cursor = conn.cursor()
        cursor.execute("SELECT sqlite_version();")
        version = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        logger.info(f"[OK] SQLite connection successful!")
        logger.info(f"   SQLite Version: {version}")
        logger.info(f"   Database file: {db_path}")
        
        return True, None, 'sqlite'
        
    except Exception as e:
        return False, f"[ERROR] SQLite connection error: {str(e)}", None

def check_tables() -> tuple[bool, list[str]]:
    """Check if required tables exist"""
    try:
        from utils.postgres_manager import PostgresManager
        
        logger.info("\n[CHECK] Checking database tables...")
        
        db = PostgresManager()
        if not (db.connection_pool or db.sqlite_conn):
            return False, ["Database connection not available"]
        
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Check for required tables
        required_tables = ['sessions', 'queries', 'feedback', 'validated_qna']
        existing_tables = []
        missing_tables = []
        
        if db.db_type == 'sqlite':
            cursor.execute("""
                SELECT name 
                FROM sqlite_master 
                WHERE type='table'
            """)
        else:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
        
        existing = {row[0] for row in cursor.fetchall()}
        
        for table in required_tables:
            if table in existing:
                existing_tables.append(table)
            else:
                missing_tables.append(table)
        
        cursor.close()
        if db.db_type == 'postgres':
            db.return_connection(conn)
        
        logger.info(f"   Found tables: {', '.join(existing_tables) if existing_tables else 'None'}")
        
        if missing_tables:
            logger.warning(f"   Missing tables: {', '.join(missing_tables)}")
            logger.info("   Run: python scripts/setup_postgres.py to create them")
            return False, [f"Missing tables: {', '.join(missing_tables)}"]
        
        logger.info("[OK] All required tables exist!")
        return True, []
        
    except Exception as e:
        return False, [f"Error checking tables: {str(e)}"]

def main():
    """Main function"""
    print("=" * 70)
    print("Database Setup and Connection Checker")
    print("Supports PostgreSQL and SQLite (auto-fallback)")
    print("=" * 70)
    print()
    
    # Check environment variables
    env_ok, env_issues = check_environment_variables()
    
    if env_issues:
        print("\n".join(env_issues))
        print()
    
    # Test connection
    conn_ok, conn_error, db_type = test_connection()
    
    if not conn_ok:
        print(f"\n{conn_error}")
        print("\nTroubleshooting:")
        if db_type is None:
            print("  - SQLite will be created automatically when first used")
            print("  - Or set PostgreSQL environment variables for PostgreSQL")
        sys.exit(1)
    
    # Check tables
    tables_ok, table_issues = check_tables()
    
    if table_issues:
        print("\n".join(table_issues))
    
    print()
    print("=" * 70)
    if conn_ok and tables_ok:
        db_name = "PostgreSQL" if db_type == 'postgres' else "SQLite"
        print(f"[OK] {db_name} is fully configured and ready!")
    elif conn_ok:
        db_name = "PostgreSQL" if db_type == 'postgres' else "SQLite"
        print(f"[WARN] {db_name} is connected but tables need to be created")
        print("   Run: python scripts/setup_postgres.py")
    print("=" * 70)

if __name__ == "__main__":
    main()

