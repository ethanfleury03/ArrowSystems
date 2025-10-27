#!/usr/bin/env python3
"""
PostgreSQL Setup Script for RAG Application
Creates database and tables for future migration from DynamoDB
"""

import os
import sys
import logging
from utils.postgres_manager import PostgreSQLManager

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
        pg_manager = PostgreSQLManager()
        
        # Create tables
        pg_manager.create_tables()
        
        logger.info("PostgreSQL setup completed successfully!")
        
    except Exception as e:
        logger.error(f"PostgreSQL setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_postgres()
