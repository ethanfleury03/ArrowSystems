#!/usr/bin/env python3
"""
Quick diagnostic script to check database connection configuration.
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=" * 60)
    print("Database Connection Diagnostic")
    print("=" * 60)
    print()
    
    # Check if DATABASE_URL is set
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        print("❌ DATABASE_URL environment variable is NOT set!")
        print()
        print("To fix this, you need to set DATABASE_URL. Options:")
        print()
        print("Option 1: If using Cloud SQL Proxy (recommended for Cloud SQL):")
        print("  1. Start Cloud SQL Proxy first:")
        print("     cloud-sql-proxy arrow-rag-support-prod:us-central1:rag-postgres &")
        print("  2. Then set DATABASE_URL:")
        print("     export DATABASE_URL='postgresql://rag_user:YOUR_PASSWORD@127.0.0.1:5432/rag_app'")
        print()
        print("Option 2: Direct connection to Cloud SQL (if external IP enabled):")
        print("  export DATABASE_URL='postgresql://rag_user:YOUR_PASSWORD@<CLOUD_SQL_IP>:5432/rag_app'")
        print()
        print("Option 3: Use the setup script:")
        print("  source setup_runpod_ingestion.sh")
        print("  # Then edit it to set YOUR_PASSWORD")
        print()
        return 1
    
    # Mask password in output
    if "@" in database_url:
        parts = database_url.split("@")
        if len(parts) == 2:
            user_pass = parts[0]
            if ":" in user_pass:
                user, _ = user_pass.split(":", 1)
                masked_url = f"postgresql://{user}:***@{parts[1]}"
            else:
                masked_url = database_url
        else:
            masked_url = database_url
    else:
        masked_url = database_url
    
    print(f"✅ DATABASE_URL is set: {masked_url}")
    print()
    
    # Try to import and test connection
    try:
        from backend.utils.db import get_engine, DATABASE_URL
        print("✅ Database module imported successfully")
        print()
        
        print("Attempting to connect to database...")
        engine = get_engine()
        
        # Try a simple connection test
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            result.fetchone()
        
        print("✅ Database connection successful!")
        print()
        return 0
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print()
        
        # Provide specific guidance based on error
        error_str = str(e).lower()
        if "connection refused" in error_str or "127.0.0.1" in database_url:
            print("The connection is being refused. This usually means:")
            print("  1. PostgreSQL is not running on the target host")
            print("  2. Cloud SQL Proxy is not running (if using Cloud SQL)")
            print("  3. The host/port is incorrect")
            print()
            print("If using Cloud SQL Proxy, make sure it's running:")
            print("  ps aux | grep cloud-sql-proxy")
            print("  # If not running, start it:")
            print("  cloud-sql-proxy arrow-rag-support-prod:us-central1:rag-postgres &")
        elif "authentication failed" in error_str:
            print("Authentication failed. Check:")
            print("  1. Username is correct")
            print("  2. Password is correct")
            print("  3. Database user has proper permissions")
        elif "does not exist" in error_str:
            print("Database does not exist. Check the database name in DATABASE_URL")
        
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())

