"""
Database Manager for RAG Application
Supports both PostgreSQL and SQLite with automatic fallback
Handles all database operations for sessions, query history, feedback, and validated Q&A cache
"""

import os
import logging
import hashlib
import json
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import PostgreSQL libraries (optional)
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2.pool import SimpleConnectionPool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger.debug("psycopg2 not available - SQLite will be used")


class PostgresManager:
    """
    Unified database manager supporting both PostgreSQL and SQLite.
    Automatically tries PostgreSQL first, falls back to SQLite if unavailable.
    """
    
    def __init__(self):
        self.db_type = None  # 'postgres' or 'sqlite'
        self.connection_pool = None
        self.sqlite_conn = None
        self.param_style = 'postgres'  # 'postgres' uses %s, 'sqlite' uses ?
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize database connection - tries PostgreSQL first, falls back to SQLite"""
        # Try PostgreSQL first if credentials are provided
        if PSYCOPG2_AVAILABLE:
            postgres_ok = self._try_postgresql()
            if postgres_ok:
                return
        
        # Fall back to SQLite
        self._initialize_sqlite()
    
    def _try_postgresql(self) -> bool:
        """Try to initialize PostgreSQL connection"""
        try:
            # Database configuration from environment variables
            # Support both POSTGRES_* and DB_* naming conventions for compatibility
            db_config = {
                'host': os.getenv('POSTGRES_HOST') or os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('POSTGRES_PORT') or os.getenv('DB_PORT', '5432'),
                'database': os.getenv('POSTGRES_DB') or os.getenv('DB_NAME', 'rag_app'),
                'user': os.getenv('POSTGRES_USER') or os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('POSTGRES_PASSWORD') or os.getenv('DB_PASSWORD', 'password'),
            }
            
            # Skip if using defaults that likely won't work
            if db_config['host'] == 'localhost' and db_config['password'] == 'password':
                # Check if we have any explicit PostgreSQL env vars
                has_postgres_env = any([
                    os.getenv('POSTGRES_HOST'),
                    os.getenv('POSTGRES_USER'),
                    os.getenv('POSTGRES_PASSWORD'),
                    os.getenv('DB_HOST'),
                    os.getenv('DB_USER'),
                    os.getenv('DB_PASSWORD')
                ])
                if not has_postgres_env:
                    logger.debug("No PostgreSQL credentials found, will use SQLite")
                    return False
            
            # Create connection pool
            self.connection_pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                **db_config
            )
            
            # Test the connection
            conn = self.connection_pool.getconn()
            self.connection_pool.putconn(conn)
            
            self.db_type = 'postgres'
            self.param_style = 'postgres'
            logger.info(f"✅ PostgreSQL connection initialized: {db_config['host']}:{db_config['port']}/{db_config['database']}")
            return True
            
        except Exception as e:
            logger.debug(f"PostgreSQL connection failed: {e}")
            return False
    
    def _initialize_sqlite(self):
        """Initialize SQLite connection"""
        try:
            # Get database path from environment or use default
            db_path = os.getenv('SQLITE_DB_PATH', 'rag_app.db')
            
            # Create directory if needed
            db_file = Path(db_path)
            if db_file.parent != Path('.'):
                db_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to SQLite (will create file if doesn't exist)
            self.sqlite_conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self.sqlite_conn.row_factory = sqlite3.Row  # Enable dict-like access
            
            # Enable foreign keys
            self.sqlite_conn.execute("PRAGMA foreign_keys = ON")
            
            # Enable JSON support (SQLite 3.9+)
            self.sqlite_conn.execute("PRAGMA foreign_keys = ON")
            
            self.db_type = 'sqlite'
            self.param_style = 'sqlite'
            logger.info(f"✅ SQLite database initialized: {db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLite: {e}")
            self.sqlite_conn = None
    
    def get_connection(self):
        """Get a database connection"""
        if self.db_type == 'postgres':
            if not self.connection_pool:
                raise Exception("PostgreSQL connection pool not initialized")
            return self.connection_pool.getconn()
        elif self.db_type == 'sqlite':
            if not self.sqlite_conn:
                raise Exception("SQLite connection not initialized")
            return self.sqlite_conn
        else:
            raise Exception("No database connection available")
    
    def return_connection(self, conn):
        """Return a connection to the pool (PostgreSQL only)"""
        if self.db_type == 'postgres' and self.connection_pool:
            self.connection_pool.putconn(conn)
        # SQLite doesn't need connection pooling
    
    def _adapt_sql(self, sql: str) -> str:
        """Adapt SQL syntax for the current database backend"""
        if self.db_type == 'sqlite':
            # Replace PostgreSQL-specific syntax with SQLite equivalents
            sql = sql.replace('SERIAL PRIMARY KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT')
            sql = sql.replace('JSONB', 'TEXT')  # SQLite stores JSON as TEXT
            sql = sql.replace('TEXT[]', 'TEXT')  # SQLite doesn't have native arrays
            # Fix JSON extraction - PostgreSQL: metadata->>'key' -> SQLite: json_extract(metadata, '$.key')
            import re
            sql = re.sub(r"metadata->>'([^']+)'", r"json_extract(metadata, '$.\\1')", sql)
            sql = sql.replace("::float", "")  # SQLite doesn't need type casting
            sql = sql.replace("::integer", "")  # SQLite doesn't need type casting
        return sql
    
    def _adapt_params(self, params: tuple) -> tuple:
        """Adapt parameter placeholders for the current database backend"""
        if self.param_style == 'sqlite':
            # Convert %s to ? for SQLite
            return params  # Parameters are the same, just placeholder style differs
        return params
    
    def _execute(self, sql: str, params: tuple = None):
        """Execute SQL with proper adaptation for the database backend"""
        sql = self._adapt_sql(sql)
        
        # Convert parameter placeholders
        if self.param_style == 'sqlite':
            sql = sql.replace('%s', '?')
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            return cursor, conn
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            logger.error(f"SQL: {sql}")
            logger.error(f"Params: {params}")
            raise
    
    def create_tables(self):
        """Create necessary tables for the RAG application"""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Sessions table
            sessions_sql = """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id VARCHAR(255) PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """
            cursor.execute(self._adapt_sql(sessions_sql))
            
            # Queries table
            queries_sql = """
                CREATE TABLE IF NOT EXISTS queries (
                    query_id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255),
                    query_text TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    response_text TEXT,
                    response_time_ms INTEGER,
                    metadata TEXT
                )
            """
            cursor.execute(self._adapt_sql(queries_sql))
            
            # Foreign key constraint (may fail if table already exists)
            try:
                if self.db_type == 'sqlite':
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_queries_session_fk 
                        ON queries(session_id)
                    """)
            except:
                pass  # Index may already exist
            
            # Feedback table
            feedback_sql = """
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255),
                    query_id INTEGER,
                    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                    is_helpful BOOLEAN,
                    feedback_text TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """
            cursor.execute(self._adapt_sql(feedback_sql))
            
            # Validated Q&A table
            validated_sql = """
                CREATE TABLE IF NOT EXISTS validated_qna (
                    query_hash VARCHAR(255) PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    answer_text TEXT NOT NULL,
                    sources TEXT,
                    helpful_count INTEGER DEFAULT 0,
                    unhelpful_count INTEGER DEFAULT 0,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    first_validated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            cursor.execute(self._adapt_sql(validated_sql))
            
            # Create indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_queries_session_id ON queries(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_feedback_query_id ON feedback(query_id)",
                "CREATE INDEX IF NOT EXISTS idx_feedback_is_helpful ON feedback(is_helpful)",
                "CREATE INDEX IF NOT EXISTS idx_validated_qna_is_active ON validated_qna(is_active)",
                "CREATE INDEX IF NOT EXISTS idx_validated_qna_helpful_count ON validated_qna(helpful_count)"
            ]
            
            for idx_sql in indexes:
                try:
                    cursor.execute(self._adapt_sql(idx_sql))
                except Exception as e:
                    logger.debug(f"Index creation warning (may already exist): {e}")
            
            conn.commit()
            logger.info(f"✅ Database tables created successfully ({self.db_type})")
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            if conn:
                conn.rollback()
        finally:
            if cursor:
                cursor.close()
            if conn and self.db_type == 'postgres':
                self.return_connection(conn)
    
    def save_session(self, session_id: str, user_id: str, metadata: Dict = None) -> bool:
        """Save a new session"""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            metadata_json = json.dumps(metadata or {})
            
            if self.db_type == 'sqlite':
                cursor.execute("""
                    INSERT INTO sessions (session_id, user_id, metadata)
                    VALUES (?, ?, ?)
                    ON CONFLICT (session_id) DO UPDATE SET
                        last_activity = CURRENT_TIMESTAMP,
                        metadata = excluded.metadata
                """, (session_id, user_id, metadata_json))
            else:
                cursor.execute("""
                    INSERT INTO sessions (session_id, user_id, metadata)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (session_id) DO UPDATE SET
                        last_activity = CURRENT_TIMESTAMP,
                        metadata = EXCLUDED.metadata
                """, (session_id, user_id, metadata_json))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
            if conn and self.db_type == 'postgres':
                self.return_connection(conn)
    
    def save_query(self, user: str = None, query_text: str = None, answer_text: str = None,
                   intent_type: str = None, intent_confidence: float = None, sources: List[str] = None,
                   confidence: float = None, response_time_ms: int = None, session_id: str = None,
                   **kwargs) -> Optional[str]:
        """
        Save a query and return query_id (string format for compatibility)
        Returns: query_id string in format "username_timestamp"
        """
        if not (self.connection_pool or self.sqlite_conn):
            logger.debug("Database not available, skipping query save")
            return None
            
        conn = None
        cursor = None
        try:
            if not session_id:
                session_id = kwargs.get('session_id', 'unknown')
            if not query_text:
                query_text = kwargs.get('query_text', '')
            
            response_text = answer_text or kwargs.get('response_text')
            
            # Generate query_id string (DynamoDB-style format for compatibility)
            timestamp = datetime.utcnow().isoformat()
            user_str = user or kwargs.get('user', 'unknown')
            query_id_str = f"{user_str}_{timestamp}"
            
            # Build metadata from additional parameters
            metadata = {
                'user': user_str,
                'query_id': query_id_str,
                'intent_type': intent_type,
                'intent_confidence': intent_confidence,
                'sources': sources or [],
                'confidence': confidence,
            }
            for k, v in kwargs.items():
                if k not in ['session_id', 'query_text', 'response_text', 'response_time_ms', 'user', 'answer_text']:
                    metadata[k] = v
            
            conn = self.get_connection()
            cursor = conn.cursor()
            
            metadata_json = json.dumps(metadata)
            
            if self.db_type == 'sqlite':
                cursor.execute("""
                    INSERT INTO queries (session_id, query_text, response_text, response_time_ms, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (session_id, query_text, response_text, response_time_ms, metadata_json))
                query_id_int = cursor.lastrowid
            else:
                cursor.execute("""
                    INSERT INTO queries (session_id, query_text, response_text, response_time_ms, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING query_id
                """, (session_id, query_text, response_text, response_time_ms, metadata_json))
                query_id_int = cursor.fetchone()[0]
            
            conn.commit()
            
            # Return string query_id for compatibility
            return query_id_str
            
        except Exception as e:
            logger.warning(f"Failed to save query to database: {e}")
            if conn:
                conn.rollback()
            return None
        finally:
            if cursor:
                cursor.close()
            if conn and self.db_type == 'postgres':
                self.return_connection(conn)
    
    def save_feedback(self, query_id: str = None, session_id: str = None, user: str = None,
                     query_id_int: int = None, rating: int = None, is_helpful: bool = None,
                     feedback_text: str = None, metadata: Dict = None, **kwargs) -> bool:
        """
        Save user feedback - supports multiple call signatures
        """
        if not (self.connection_pool or self.sqlite_conn):
            logger.debug("Database not available, skipping feedback save")
            return False
            
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # If query_id (string) provided, look up query_id_int from queries table
            if query_id and not query_id_int:
                if self.db_type == 'sqlite':
                    cursor.execute("""
                        SELECT query_id FROM queries 
                        WHERE json_extract(metadata, '$.query_id') = ?
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    """, (query_id,))
                else:
                    cursor.execute("""
                        SELECT query_id FROM queries 
                        WHERE metadata->>'query_id' = %s
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    """, (query_id,))
                result = cursor.fetchone()
                if result:
                    query_id_int = result[0] if self.db_type == 'sqlite' else result[0]
                else:
                    # Fallback: try to find by user
                    if user:
                        if self.db_type == 'sqlite':
                            cursor.execute("""
                                SELECT query_id FROM queries 
                                WHERE json_extract(metadata, '$.user') = ? 
                                ORDER BY timestamp DESC 
                                LIMIT 1
                            """, (user,))
                        else:
                            cursor.execute("""
                                SELECT query_id FROM queries 
                                WHERE metadata->>'user' = %s 
                                ORDER BY timestamp DESC 
                                LIMIT 1
                            """, (user,))
                        result = cursor.fetchone()
                        if result:
                            query_id_int = result[0] if self.db_type == 'sqlite' else result[0]
            
            # Convert is_helpful to rating if needed
            if is_helpful is not None and rating is None:
                rating = 5 if is_helpful else 1
            elif rating is None:
                rating = None
            
            # Convert rating to is_helpful if needed
            if rating is not None and is_helpful is None:
                is_helpful = rating >= 4
            
            if not query_id_int:
                logger.warning("Could not determine query_id_int for feedback")
                return False
            
            metadata_json = json.dumps(metadata or {})
            
            if self.db_type == 'sqlite':
                cursor.execute("""
                    INSERT INTO feedback (session_id, query_id, rating, is_helpful, feedback_text, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (session_id or 'unknown', query_id_int, rating, is_helpful, feedback_text, metadata_json))
            else:
                cursor.execute("""
                    INSERT INTO feedback (session_id, query_id, rating, is_helpful, feedback_text, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (session_id or 'unknown', query_id_int, rating, is_helpful, feedback_text, metadata_json))
            
            conn.commit()
            
            # If helpful, update validated Q&A cache
            if is_helpful:
                self._update_validated_qna(query_id_int, is_helpful=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
            if conn and self.db_type == 'postgres':
                self.return_connection(conn)
    
    def get_query_feedback(self, query_id: str = None, query_id_int: int = None) -> List[Dict]:
        """Get all feedback for a specific query"""
        if not (self.connection_pool or self.sqlite_conn):
            return []
            
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            
            if self.db_type == 'sqlite':
                # SQLite uses Row factory which gives dict-like access
                cursor = conn.cursor()
                
                if query_id_int:
                    cursor.execute("""
                        SELECT * FROM feedback 
                        WHERE query_id = ? 
                        ORDER BY timestamp DESC
                    """, (query_id_int,))
                elif query_id:
                    cursor.execute("""
                        SELECT f.* FROM feedback f
                        JOIN queries q ON f.query_id = q.query_id
                        WHERE json_extract(q.metadata, '$.query_id') = ?
                        ORDER BY f.timestamp DESC
                    """, (query_id,))
                else:
                    return []
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            else:
                # PostgreSQL
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                if query_id_int:
                    cursor.execute("""
                        SELECT * FROM feedback 
                        WHERE query_id = %s 
                        ORDER BY timestamp DESC
                    """, (query_id_int,))
                elif query_id:
                    cursor.execute("""
                        SELECT f.* FROM feedback f
                        JOIN queries q ON f.query_id = q.query_id
                        WHERE q.metadata->>'query_id' = %s
                        ORDER BY f.timestamp DESC
                    """, (query_id,))
                else:
                    return []
                
                return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Failed to get query feedback: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn and self.db_type == 'postgres':
                self.return_connection(conn)
    
    def get_query_by_id(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific query by its ID (string format: username_timestamp)"""
        if not (self.connection_pool or self.sqlite_conn):
            return None
            
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            
            if self.db_type == 'sqlite':
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM queries 
                    WHERE json_extract(metadata, '$.query_id') = ?
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (query_id,))
            else:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                    SELECT * FROM queries 
                    WHERE metadata->>'query_id' = %s
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (query_id,))
            
            result = cursor.fetchone()
            if result:
                row = dict(result)
                # Parse metadata JSON
                if isinstance(row.get('metadata'), str):
                    row['metadata'] = json.loads(row['metadata'])
                return row
            return None
            
        except Exception as e:
            logger.error(f"Failed to get query by ID: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if conn and self.db_type == 'postgres':
                self.return_connection(conn)
    
    def get_user_query_history(self, user: str, limit: int = 20, start_date: datetime = None) -> List[Dict[str, Any]]:
        """Get query history for a specific user"""
        if not (self.connection_pool or self.sqlite_conn):
            return []
            
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            
            if self.db_type == 'sqlite':
                cursor = conn.cursor()
                if start_date:
                    cursor.execute("""
                        SELECT * FROM queries 
                        WHERE json_extract(metadata, '$.user') = ? 
                        AND timestamp >= ?
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (user, start_date, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM queries 
                        WHERE json_extract(metadata, '$.user') = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (user, limit))
            else:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                if start_date:
                    cursor.execute("""
                        SELECT * FROM queries 
                        WHERE metadata->>'user' = %s 
                        AND timestamp >= %s
                        ORDER BY timestamp DESC 
                        LIMIT %s
                    """, (user, start_date, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM queries 
                        WHERE metadata->>'user' = %s 
                        ORDER BY timestamp DESC 
                        LIMIT %s
                    """, (user, limit))
            
            results = cursor.fetchall()
            history = []
            for row in results:
                item = dict(row)
                if isinstance(item.get('metadata'), str):
                    item['metadata'] = json.loads(item['metadata'])
                history.append(item)
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get user query history: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn and self.db_type == 'postgres':
                self.return_connection(conn)
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get overall feedback statistics"""
        if not (self.connection_pool or self.sqlite_conn):
            return {'total': 0, 'helpful': 0, 'unhelpful': 0, 'helpful_percentage': 0}
            
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if self.db_type == 'sqlite':
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN is_helpful = 1 THEN 1 ELSE 0 END) as helpful,
                        SUM(CASE WHEN is_helpful = 0 THEN 1 ELSE 0 END) as unhelpful
                    FROM feedback
                """)
            else:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE is_helpful = TRUE) as helpful,
                        COUNT(*) FILTER (WHERE is_helpful = FALSE) as unhelpful
                    FROM feedback
                """)
            
            result = cursor.fetchone()
            total = result[0] or 0
            helpful = result[1] or 0
            unhelpful = result[2] or 0
            
            return {
                'total': total,
                'helpful': helpful,
                'unhelpful': unhelpful,
                'helpful_percentage': (helpful / total * 100) if total > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get feedback stats: {e}")
            return {'total': 0, 'helpful': 0, 'unhelpful': 0, 'helpful_percentage': 0}
        finally:
            if cursor:
                cursor.close()
            if conn and self.db_type == 'postgres':
                self.return_connection(conn)
    
    def get_intent_distribution(self, days: int = 30) -> Dict[str, int]:
        """Get distribution of intent types over the last N days"""
        if not (self.connection_pool or self.sqlite_conn):
            return {}
            
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            start_date = datetime.utcnow() - timedelta(days=days)
            
            if self.db_type == 'sqlite':
                cursor.execute("""
                    SELECT json_extract(metadata, '$.intent_type') as intent_type, COUNT(*) as count
                    FROM queries
                    WHERE timestamp >= ?
                    AND json_extract(metadata, '$.intent_type') IS NOT NULL
                    GROUP BY json_extract(metadata, '$.intent_type')
                """, (start_date,))
            else:
                cursor.execute("""
                    SELECT metadata->>'intent_type' as intent_type, COUNT(*) as count
                    FROM queries
                    WHERE timestamp >= %s
                    AND metadata->>'intent_type' IS NOT NULL
                    GROUP BY metadata->>'intent_type'
                """, (start_date,))
            
            results = cursor.fetchall()
            distribution = {}
            for row in results:
                intent = row[0] if self.db_type == 'sqlite' else row[0]
                count = row[1] if self.db_type == 'sqlite' else row[1]
                intent_name = intent or 'unknown'
                distribution[intent_name] = count
            
            return distribution
            
        except Exception as e:
            logger.error(f"Failed to get intent distribution: {e}")
            return {}
        finally:
            if cursor:
                cursor.close()
            if conn and self.db_type == 'postgres':
                self.return_connection(conn)
    
    def get_average_metrics(self, days: int = 30) -> Dict[str, float]:
        """Get average confidence and response time metrics"""
        if not (self.connection_pool or self.sqlite_conn):
            return {
                'avg_confidence': 0.0,
                'avg_response_time_ms': 0.0,
                'avg_intent_confidence': 0.0,
                'total_queries': 0
            }
            
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            start_date = datetime.utcnow() - timedelta(days=days)
            
            if self.db_type == 'sqlite':
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        AVG(CAST(json_extract(metadata, '$.confidence') AS REAL)) as avg_confidence,
                        AVG(response_time_ms) as avg_response_time,
                        AVG(CAST(json_extract(metadata, '$.intent_confidence') AS REAL)) as avg_intent_confidence
                    FROM queries
                    WHERE timestamp >= ?
                """, (start_date,))
            else:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        AVG((metadata->>'confidence')::float) as avg_confidence,
                        AVG(response_time_ms) as avg_response_time,
                        AVG((metadata->>'intent_confidence')::float) as avg_intent_confidence
                    FROM queries
                    WHERE timestamp >= %s
                """, (start_date,))
            
            result = cursor.fetchone()
            total = result[0] or 0
            
            if total == 0:
                return {
                    'avg_confidence': 0.0,
                    'avg_response_time_ms': 0.0,
                    'avg_intent_confidence': 0.0,
                    'total_queries': 0
                }
            
            return {
                'avg_confidence': float(result[1] or 0.0),
                'avg_response_time_ms': float(result[2] or 0.0),
                'avg_intent_confidence': float(result[3] or 0.0),
                'total_queries': total
            }
            
        except Exception as e:
            logger.error(f"Failed to get average metrics: {e}")
            return {
                'avg_confidence': 0.0,
                'avg_response_time_ms': 0.0,
                'avg_intent_confidence': 0.0,
                'total_queries': 0
            }
        finally:
            if cursor:
                cursor.close()
            if conn and self.db_type == 'postgres':
                self.return_connection(conn)
    
    def _update_validated_qna(self, query_id_int: int, is_helpful: bool):
        """
        Internal method to update validated Q&A based on feedback.
        Strategy: First validated answer wins (never overwrite).
        """
        if not (self.connection_pool or self.sqlite_conn):
            return
            
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            
            if self.db_type == 'sqlite':
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT query_text, response_text, metadata 
                    FROM queries 
                    WHERE query_id = ?
                """, (query_id_int,))
            else:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                    SELECT query_text, response_text, metadata 
                    FROM queries 
                    WHERE query_id = %s
                """, (query_id_int,))
            
            result = cursor.fetchone()
            if not result:
                return
            
            query = dict(result)
            metadata = query.get('metadata') or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            
            query_text = query['query_text']
            answer_text = query['response_text'] or ''
            sources = metadata.get('sources', [])
            
            # Create hash of query for deduplication
            query_hash = hashlib.md5(query_text.lower().encode()).hexdigest()
            
            # Store sources as JSON string for SQLite
            sources_json = json.dumps(sources) if isinstance(sources, list) else sources
            
            # Update or create validated QnA entry
            if self.db_type == 'sqlite':
                cursor.execute("""
                    INSERT INTO validated_qna (
                        query_hash, query_text, answer_text, sources, 
                        helpful_count, last_used, is_active, first_validated
                    )
                    VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP, 1, CURRENT_TIMESTAMP)
                    ON CONFLICT (query_hash) DO UPDATE SET
                        helpful_count = validated_qna.helpful_count + 1,
                        last_used = CURRENT_TIMESTAMP,
                        query_text = COALESCE(validated_qna.query_text, excluded.query_text),
                        answer_text = COALESCE(validated_qna.answer_text, excluded.answer_text),
                        sources = COALESCE(validated_qna.sources, excluded.sources),
                        first_validated = COALESCE(validated_qna.first_validated, excluded.first_validated)
                """, (query_hash, query_text, answer_text, sources_json))
            else:
                cursor.execute("""
                    INSERT INTO validated_qna (
                        query_hash, query_text, answer_text, sources, 
                        helpful_count, last_used, is_active, first_validated
                    )
                    VALUES (%s, %s, %s, %s, 1, CURRENT_TIMESTAMP, TRUE, CURRENT_TIMESTAMP)
                    ON CONFLICT (query_hash) DO UPDATE SET
                        helpful_count = validated_qna.helpful_count + 1,
                        last_used = CURRENT_TIMESTAMP,
                        query_text = COALESCE(validated_qna.query_text, EXCLUDED.query_text),
                        answer_text = COALESCE(validated_qna.answer_text, EXCLUDED.answer_text),
                        sources = COALESCE(validated_qna.sources, EXCLUDED.sources),
                        first_validated = COALESCE(validated_qna.first_validated, EXCLUDED.first_validated)
                """, (query_hash, query_text, answer_text, sources))
            
            conn.commit()
            logger.info(f"✅ Updated ValidatedQnA for query (helpful_count +1, answer preserved)")
            
        except Exception as e:
            logger.error(f"Error updating validated QnA: {e}")
            if conn:
                conn.rollback()
        finally:
            if cursor:
                cursor.close()
            if conn and self.db_type == 'postgres':
                self.return_connection(conn)
    
    def get_validated_answer(self, query_text: str) -> Optional[Dict[str, Any]]:
        """
        Get a validated answer for a similar query (if exists).
        Uses exact hash match for now.
        """
        if not (self.connection_pool or self.sqlite_conn):
            return None
            
        conn = None
        cursor = None
        try:
            query_hash = hashlib.md5(query_text.lower().encode()).hexdigest()
            
            conn = self.get_connection()
            
            if self.db_type == 'sqlite':
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM validated_qna
                    WHERE query_hash = ?
                    AND is_active = 1
                    AND helpful_count >= 2
                """, (query_hash,))
            else:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                    SELECT * FROM validated_qna
                    WHERE query_hash = %s
                    AND is_active = TRUE
                    AND helpful_count >= 2
                """, (query_hash,))
            
            result = cursor.fetchone()
            if result:
                row = dict(result)
                # Convert sources from JSON string to list if needed
                sources = row.get('sources')
                if isinstance(sources, str):
                    try:
                        row['sources'] = json.loads(sources) if sources.startswith('[') else [sources]
                    except:
                        row['sources'] = [sources] if sources else []
                return row
            return None
            
        except Exception as e:
            logger.error(f"Error getting validated answer: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if conn and self.db_type == 'postgres':
                self.return_connection(conn)
    
    def get_session_queries(self, session_id: str) -> List[Dict]:
        """Get all queries for a session"""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            
            if self.db_type == 'sqlite':
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT q.*, f.rating, f.feedback_text
                    FROM queries q
                    LEFT JOIN feedback f ON q.query_id = f.query_id
                    WHERE q.session_id = ?
                    ORDER BY q.timestamp DESC
                """, (session_id,))
            else:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                    SELECT q.*, f.rating, f.feedback_text
                    FROM queries q
                    LEFT JOIN feedback f ON q.query_id = f.query_id
                    WHERE q.session_id = %s
                    ORDER BY q.timestamp DESC
                """, (session_id,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get session queries: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn and self.db_type == 'postgres':
                self.return_connection(conn)
    
    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """Get all sessions for a user"""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            
            if self.db_type == 'sqlite':
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT s.*, COUNT(q.query_id) as query_count
                    FROM sessions s
                    LEFT JOIN queries q ON s.session_id = q.session_id
                    WHERE s.user_id = ?
                    GROUP BY s.session_id
                    ORDER BY s.last_activity DESC
                """, (user_id,))
            else:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                    SELECT s.*, COUNT(q.query_id) as query_count
                    FROM sessions s
                    LEFT JOIN queries q ON s.session_id = q.session_id
                    WHERE s.user_id = %s
                    GROUP BY s.session_id
                    ORDER BY s.last_activity DESC
                """, (user_id,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn and self.db_type == 'postgres':
                self.return_connection(conn)
    
    def close(self):
        """Close the database connection"""
        if self.db_type == 'postgres' and self.connection_pool:
            self.connection_pool.closeall()
            logger.info("PostgreSQL connection pool closed")
        elif self.db_type == 'sqlite' and self.sqlite_conn:
            self.sqlite_conn.close()
            logger.info("SQLite connection closed")
