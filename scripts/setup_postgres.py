#!/usr/bin/env python3
"""
PostgreSQL Setup Script for RAG Application
Creates database and tables for query history, feedback, and validated Q&A cache
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.postgres_manager import PostgresManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_postgres():
    """Setup PostgreSQL database and tables"""
    try:
        logger.info("Setting up PostgreSQL database...")
        
        # Initialize PostgreSQL manager
        pg_manager = PostgresManager()
        
        # Create tables
        pg_manager.create_tables()
        
        logger.info("PostgreSQL setup completed successfully!")
        
    except Exception as e:
        logger.error(f"PostgreSQL setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_postgres()
