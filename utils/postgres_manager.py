import logging
from typing import Any, Dict, List, Optional

from prisma import Prisma
from prisma.enums import UserRole

logger = logging.getLogger(__name__)


class PrismaManager:
    """
    Thin wrapper around Prisma client to keep backwards compatibility with the old PostgresManager API.
    The public methods are async and mirror the signatures used throughout the backend.
    """

    def __init__(self, client: Prisma):
        self.client = client

    async def _ensure_user(self, email: str) -> str:
        """
        Ensure we have a user for the incoming request.
        For now we upsert on email and mark technicians as the default role.
        """
        if not email:
            email = "api_user"

        user = await self.client.user.upsert(
            where={"email": email},
            update={},
            create={
                "email": email,
                "name": email,
                "role": UserRole.TECHNICIAN,
                "passwordHash": "",
            },
        )
        return user.id

    async def save_query(
        self,
        user: str,
        query_text: str,
        answer_text: str,
        intent_type: Optional[str] = None,
        intent_confidence: Optional[float] = None,
        sources: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        response_time_ms: Optional[int] = None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        """
        Persist a query/answer pair. Returns the query history id.
        """
        try:
            user_id = await self._ensure_user(user or "api_user")
            record = await self.client.queryhistory.create(
                data={
                    "userId": user_id,
                    "queryText": query_text,
                    "answerText": answer_text,
                    "responseTimeMs": response_time_ms,
                    "metadata": {
                        "sessionId": session_id,
                        "intentType": intent_type,
                        "intentConfidence": intent_confidence,
                        "confidence": confidence,
                        "sources": sources or [],
                        **{k: v for k, v in kwargs.items() if v is not None},
                    },
                }
            )
            return record.id
        except Exception as exc:  # pragma: no cover - external IO
            logger.warning("Failed to save query: %s", exc)
            return None

    async def get_query_history(self, user: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch recent query history for the given user email.
        """
        try:
            where_clause = {}
            if user:
                where_clause["user"] = {"email": user}

            records = await self.client.queryhistory.find_many(
                where=where_clause or None,
                order={"createdAt": "desc"},
                take=limit,
                include={"user": True},
            )

            history: List[Dict[str, Any]] = []
            for record in records:
                history.append(
                    {
                        "id": record.id,
                        "query": record.queryText,
                        "answer": record.answerText or "",
                        "timestamp": record.createdAt.isoformat(),
                        "intent_type": (record.metadata or {}).get("intentType"),
                        "confidence": (record.metadata or {}).get("confidence"),
                        "sources": (record.metadata or {}).get("sources") or [],
                        "response_time_ms": record.responseTimeMs,
                    }
                )

            return history
        except Exception as exc:  # pragma: no cover - external IO
            logger.error("Failed to fetch query history: %s", exc)
            return []

    def get_validated_answer(self, query_text: str) -> Optional[Dict[str, Any]]:
        """
        Placeholder to maintain compatibility with orchestrator fast-path.
        Validated answers are not yet implemented with Prisma.
        """
        return None

    async def save_feedback(
        self,
        query_id: str,
        user: str,
        is_helpful: bool,
        confidence: Optional[float] = None,
        intent_type: Optional[str] = None,
    ) -> bool:
        """
        Store thumbs up/down feedback for a query.
        query_id here refers to the original query text for backwards compatibility.
        """
        try:
            query_record = await self.client.queryhistory.find_first(
                where={"queryText": query_id},
                order={"createdAt": "desc"}
            )
            if not query_record:
                logger.warning("Query text '%s' not found for feedback", query_id)
                return False

            user_id = await self._ensure_user(user or "api_user")

            await self.client.feedback.create(
                data={
                    "queryId": query_record.id,
                    "userId": user_id,
                    "isHelpful": is_helpful,
                    "confidence": confidence,
                    "intentType": intent_type,
                }
            )

            return True
        except Exception as exc:  # pragma: no cover - external IO
            logger.error("Failed to save feedback: %s", exc)
            return False
    
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
                    AND helpful_count >= 1
                """, (query_hash,))
            else:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                    SELECT * FROM validated_qna
                    WHERE query_hash = %s
                    AND is_active = TRUE
                    AND helpful_count >= 1
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
    
    def get_all_validated_qna(self, limit: int = 50, min_helpful_count: int = 2) -> List[Dict[str, Any]]:
        """
        Get all validated Q&A entries that have been marked as helpful.
        
        Args:
            limit: Maximum number of entries to return
            min_helpful_count: Minimum helpful_count to include
        
        Returns:
            List of validated Q&A entries
        """
        if not (self.connection_pool or self.sqlite_conn):
            return []
            
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            
            if self.db_type == 'sqlite':
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM validated_qna
                    WHERE is_active = 1
                    AND helpful_count >= ?
                    ORDER BY helpful_count DESC, last_used DESC
                    LIMIT ?
                """, (min_helpful_count, limit))
            else:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                    SELECT * FROM validated_qna
                    WHERE is_active = TRUE
                    AND helpful_count >= %s
                    ORDER BY helpful_count DESC, last_used DESC
                    LIMIT %s
                """, (min_helpful_count, limit))
            
            results = cursor.fetchall()
            validated_entries = []
            for row in results:
                item = dict(row)
                # Convert sources from JSON string to list if needed
                sources = item.get('sources')
                if isinstance(sources, str):
                    try:
                        item['sources'] = json.loads(sources) if sources.startswith('[') else [sources] if sources else []
                    except:
                        item['sources'] = [sources] if sources else []
                else:
                    item['sources'] = sources or []
                validated_entries.append(item)
            
            return validated_entries
            
        except Exception as e:
            logger.error(f"Error getting all validated Q&A: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn and self.db_type == 'postgres':
                self.return_connection(conn)

    def get_query_history(self, user: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve recent query history for a user.
        """
        if not (self.connection_pool or self.sqlite_conn):
            logger.debug("Database not available, skipping query history fetch")
            return []

        conn = None
        cursor = None
        try:
            conn = self.get_connection()

            if self.db_type == 'sqlite':
                cursor = conn.cursor()
                if user:
                    cursor.execute(
                        """
                        SELECT query_id, query_text, response_text, timestamp, metadata
                        FROM queries
                        WHERE json_extract(metadata, '$.user') = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (user, limit),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT query_id, query_text, response_text, timestamp, metadata
                        FROM queries
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (limit,),
                    )
                rows = cursor.fetchall()
            else:
                cursor = conn.cursor()
                if user:
                    cursor.execute(
                        """
                        SELECT query_id, query_text, response_text, timestamp, metadata
                        FROM queries
                        WHERE metadata->>'user' = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                        """,
                        (user, limit),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT query_id, query_text, response_text, timestamp, metadata
                        FROM queries
                        ORDER BY timestamp DESC
                        LIMIT %s
                        """,
                        (limit,),
                    )
                rows = cursor.fetchall()

            history: List[Dict[str, Any]] = []
            for row in rows:
                query_id, query_text, response_text, timestamp_val, metadata_val = row
                metadata_obj: Dict[str, Any] = {}
                if metadata_val:
                    if isinstance(metadata_val, str):
                        try:
                            metadata_obj = json.loads(metadata_val)
                        except json.JSONDecodeError:
                            metadata_obj = {}
                    elif isinstance(metadata_val, dict):
                        metadata_obj = metadata_val

                entry = {
                    "id": metadata_obj.get("query_id") or str(query_id),
                    "query_text": query_text,
                    "answer_text": response_text,
                    "created_at": timestamp_val.isoformat() if hasattr(timestamp_val, "isoformat") else str(timestamp_val),
                    "intent_type": metadata_obj.get("intent_type"),
                    "confidence": metadata_obj.get("confidence"),
                    "sources": metadata_obj.get("sources", []),
                    "response_time_ms": metadata_obj.get("response_time_ms"),
                }
                history.append(entry)

            return history
        except Exception as e:
            logger.error(f"Failed to fetch query history: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn and self.db_type == 'postgres':
                self.return_connection(conn)

    def upsert_saved_response(
        self,
        query_text: str,
        answer_text: str,
        sources: Optional[List[str]] = None,
        helpful_count: int = 1,
    ) -> bool:
        """
        Ensure a response is present in the validated_qna table and marked active.
        """
        if not (self.connection_pool or self.sqlite_conn):
            logger.debug("Database not available, skipping saved response upsert")
            return False

        conn = None
        cursor = None
        try:
            query_hash = hashlib.md5(query_text.lower().encode()).hexdigest()
            conn = self.get_connection()

            sources_payload = json.dumps(sources or [])

            if self.db_type == 'sqlite':
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO validated_qna (
                        query_hash, query_text, answer_text, sources,
                        helpful_count, unhelpful_count, last_used, is_active, first_validated, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, 0, CURRENT_TIMESTAMP, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (query_hash) DO UPDATE SET
                        helpful_count = MAX(validated_qna.helpful_count, excluded.helpful_count),
                        answer_text = excluded.answer_text,
                        sources = excluded.sources,
                        is_active = 1,
                        last_used = CURRENT_TIMESTAMP
                    """,
                    (query_hash, query_text, answer_text, sources_payload, helpful_count),
                )
            else:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO validated_qna (
                        query_hash, query_text, answer_text, sources,
                        helpful_count, unhelpful_count, last_used, is_active, first_validated, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, 0, CURRENT_TIMESTAMP, TRUE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (query_hash) DO UPDATE SET
                        helpful_count = GREATEST(validated_qna.helpful_count, EXCLUDED.helpful_count),
                        answer_text = EXCLUDED.answer_text,
                        sources = EXCLUDED.sources,
                        is_active = TRUE,
                        last_used = CURRENT_TIMESTAMP
                    """,
                    (query_hash, query_text, answer_text, sources_payload),
                )

            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to upsert saved response: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
            if conn and self.db_type == 'postgres':
                self.return_connection(conn)

    def remove_saved_response(self, query_text: str) -> bool:
        """
        Mark a saved response as inactive in validated_qna.
        """
        if not (self.connection_pool or self.sqlite_conn):
            logger.debug("Database not available, skipping saved response removal")
            return False

        conn = None
        cursor = None
        try:
            query_hash = hashlib.md5(query_text.lower().encode()).hexdigest()
            conn = self.get_connection()

            if self.db_type == 'sqlite':
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE validated_qna
                    SET is_active = 0,
                        last_used = CURRENT_TIMESTAMP
                    WHERE query_hash = ?
                    """,
                    (query_hash,),
                )
            else:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE validated_qna
                    SET is_active = FALSE,
                        last_used = CURRENT_TIMESTAMP
                    WHERE query_hash = %s
                    """,
                    (query_hash,),
                )

            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to remove saved response: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
            if conn and self.db_type == 'postgres':
                self.return_connection(conn)

    def is_query_saved(self, query_text: str) -> bool:
        """
        Determine if a query has an active saved/validated entry.
        """
        if not (self.connection_pool or self.sqlite_conn):
            return False

        conn = None
        cursor = None
        try:
            query_hash = hashlib.md5(query_text.lower().encode()).hexdigest()
            conn = self.get_connection()

            if self.db_type == 'sqlite':
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT 1 FROM validated_qna
                    WHERE query_hash = ?
                    AND is_active = 1
                    LIMIT 1
                    """,
                    (query_hash,),
                )
            else:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT 1 FROM validated_qna
                    WHERE query_hash = %s
                    AND is_active = TRUE
                    LIMIT 1
                    """,
                    (query_hash,),
                )

            return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Failed to check saved response: {e}")
            return False
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
