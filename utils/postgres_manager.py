"""
PostgreSQL Database Manager for RAG Application
Handles database operations for future migration from DynamoDB
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

logger = logging.getLogger(__name__)

class PostgresManager:
    """PostgreSQL database manager for RAG application"""
    
    def __init__(self):
        self.connection_pool = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize PostgreSQL connection pool"""
        try:
            # Database configuration from environment variables
            db_config = {
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': os.getenv('POSTGRES_PORT', '5432'),
                'database': os.getenv('POSTGRES_DB', 'rag_app'),
                'user': os.getenv('POSTGRES_USER', 'postgres'),
                'password': os.getenv('POSTGRES_PASSWORD', 'password'),
            }
            
            # Create connection pool
            self.connection_pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                **db_config
            )
            
            # Test the connection
            conn = self.connection_pool.getconn()
            self.connection_pool.putconn(conn)
            
            logger.info("PostgreSQL connection pool initialized")
            
        except Exception as e:
            logger.warning(f"PostgreSQL connection not available: {e}")
            logger.warning("Application will continue without database persistence")
            self.connection_pool = None
    
    def get_connection(self):
        """Get a connection from the pool"""
        if not self.connection_pool:
            raise Exception("PostgreSQL connection pool not initialized - database not available")
        return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        """Return a connection to the pool"""
        if self.connection_pool:
            self.connection_pool.putconn(conn)
    
    def create_tables(self):
        """Create necessary tables for the RAG application"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Create sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id VARCHAR(255) PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                )
            """)
            
            # Create queries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    query_id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) REFERENCES sessions(session_id),
                    query_text TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    response_text TEXT,
                    response_time_ms INTEGER,
                    metadata JSONB
                )
            """)
            
            # Create feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) REFERENCES sessions(session_id),
                    query_id INTEGER REFERENCES queries(query_id),
                    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                    feedback_text TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_queries_session_id ON queries(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_query_id ON feedback(query_id)")
            
            conn.commit()
            logger.info("PostgreSQL tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL tables: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self.return_connection(conn)
    
    def save_session(self, session_id: str, user_id: str, metadata: Dict = None) -> bool:
        """Save a new session"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO sessions (session_id, user_id, metadata)
                VALUES (%s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    last_activity = CURRENT_TIMESTAMP,
                    metadata = EXCLUDED.metadata
            """, (session_id, user_id, metadata))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self.return_connection(conn)
    
    def save_query(self, user: str = None, query_text: str = None, answer_text: str = None,
                   intent_type: str = None, intent_confidence: float = None, sources: List[str] = None,
                   confidence: float = None, response_time_ms: int = None, session_id: str = None,
                   **kwargs) -> Optional[int]:
        """Save a query and return query_id - supports multiple call signatures"""
        # Check if database is available
        if not self.connection_pool:
            logger.debug("Database not available, skipping query save")
            return None
            
        conn = None
        try:
            # Handle different call patterns
            if not session_id:
                session_id = kwargs.get('session_id', 'unknown')
            if not query_text:
                query_text = kwargs.get('query_text', '')
            
            response_text = answer_text or kwargs.get('response_text')
            
            # Build metadata from additional parameters
            metadata = {
                'user': user or kwargs.get('user'),
                'intent_type': intent_type,
                'intent_confidence': intent_confidence,
                'sources': sources or [],
                'confidence': confidence,
            }
            # Add any additional kwargs to metadata
            for k, v in kwargs.items():
                if k not in ['session_id', 'query_text', 'response_text', 'response_time_ms', 'user', 'answer_text']:
                    metadata[k] = v
            
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO queries (session_id, query_text, response_text, response_time_ms, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING query_id
            """, (session_id, query_text, response_text, response_time_ms, metadata))
            
            query_id = cursor.fetchone()[0]
            conn.commit()
            return query_id
            
        except Exception as e:
            logger.warning(f"Failed to save query to database: {e}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                self.return_connection(conn)
    
    def save_feedback(self, session_id: str, query_id: int, rating: int, 
                     feedback_text: str = None, metadata: Dict = None) -> bool:
        """Save user feedback"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO feedback (session_id, query_id, rating, feedback_text, metadata)
                VALUES (%s, %s, %s, %s, %s)
            """, (session_id, query_id, rating, feedback_text, metadata))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self.return_connection(conn)
    
    def get_session_queries(self, session_id: str) -> List[Dict]:
        """Get all queries for a session"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT q.*, f.rating, f.feedback_text
                FROM queries q
                LEFT JOIN feedback f ON q.query_id = f.query_id
                WHERE q.session_id = %s
                ORDER BY q.timestamp DESC
            """, (session_id,))
            
            return cursor.fetchall()
            
        except Exception as e:
            logger.error(f"Failed to get session queries: {e}")
            return []
        finally:
            if conn:
                self.return_connection(conn)
    
    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """Get all sessions for a user"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT s.*, COUNT(q.query_id) as query_count
                FROM sessions s
                LEFT JOIN queries q ON s.session_id = q.session_id
                WHERE s.user_id = %s
                GROUP BY s.session_id
                ORDER BY s.last_activity DESC
            """, (user_id,))
            
            return cursor.fetchall()
            
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []
        finally:
            if conn:
                self.return_connection(conn)
    
    def close(self):
        """Close the connection pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("PostgreSQL connection pool closed")
