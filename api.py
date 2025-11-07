"""
FastAPI Backend for DuraFlex Technical Assistant
Production-ready API server with RAG capabilities

This FastAPI application provides a REST API interface to the RAG system,
allowing frontend applications (like Next.js) to query the knowledge base.

Version: 1.0.0
Author: Arrow Systems Inc
"""

from __future__ import annotations

import os
import logging
import time
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from rag_pipeline import RAGPipeline, initialize_rag_pipeline, get_rag_pipeline
from utils.postgres_manager import PostgresManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables for RAG pipeline and database
rag_pipeline = None
db_manager = None


# =============================================================================
# Session Manager for Chat History (Concurrency-Safe & Modular)
# =============================================================================
class ChatMessage:
    """Represents a single chat message."""
    def __init__(self, role: str, content: str, timestamp: float = None):
        self.role = role  # "user" or "assistant"
        self.content = content
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }


class Trimmer(ABC):
    """Abstract base class for message trimming strategies."""
    
    @abstractmethod
    def trim(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """
        Trim messages according to the strategy.
        
        Args:
            messages: List of messages to trim
            
        Returns:
            Trimmed list of messages
        """
        pass


class MessageCountTrimmer(Trimmer):
    """Trims messages based on maximum count."""
    
    def __init__(self, max_messages: int):
        self.max_messages = max_messages
    
    def trim(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Keep only the last max_messages."""
        return messages[-self.max_messages:] if len(messages) > self.max_messages else messages


class SessionStore(ABC):
    """Abstract base class for session storage backends."""
    
    @abstractmethod
    async def get_messages(self, session_id: str) -> List[ChatMessage]:
        """Get all messages for a session."""
        pass
    
    @abstractmethod
    async def add_message(self, session_id: str, message: ChatMessage) -> None:
        """Add a message to a session."""
        pass
    
    @abstractmethod
    async def clear_session(self, session_id: str) -> None:
        """Clear all messages for a session."""
        pass
    
    @abstractmethod
    async def trim_session(self, session_id: str, trimmer: Trimmer) -> None:
        """Trim messages in a session using the provided trimmer."""
        pass


class InMemorySessionStore(SessionStore):
    """In-memory implementation of SessionStore."""
    
    def __init__(self, trimmer: Trimmer):
        self._sessions: Dict[str, List[ChatMessage]] = {}
        self._trimmer = trimmer
    
    async def get_messages(self, session_id: str) -> List[ChatMessage]:
        """Get all messages for a session."""
        return self._sessions.get(session_id, []).copy()
    
    async def add_message(self, session_id: str, message: ChatMessage) -> None:
        """Add a message to a session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        
        self._sessions[session_id].append(message)
        await self.trim_session(session_id, self._trimmer)
    
    async def clear_session(self, session_id: str) -> None:
        """Clear all messages for a session."""
        self._sessions.pop(session_id, None)
    
    async def trim_session(self, session_id: str, trimmer: Trimmer) -> None:
        """Trim messages in a session using the provided trimmer."""
        if session_id in self._sessions:
            self._sessions[session_id] = trimmer.trim(self._sessions[session_id])


class SessionManager:
    """
    Thread-safe session manager with per-session async locks.
    Uses a pluggable SessionStore backend for modularity.
    """
    
    def __init__(self, store: SessionStore, max_messages: int = 10):
        self._store = store
        self._locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()
        self.max_messages = max_messages
        logger.info(f"SessionManager initialized (max {max_messages} messages per session)")
    
    async def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a lock for a session."""
        async with self._locks_lock:
            if session_id not in self._locks:
                self._locks[session_id] = asyncio.Lock()
            return self._locks[session_id]
    
    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to the session history (thread-safe)."""
        if not session_id:
            return
        
        lock = await self._get_lock(session_id)
        async with lock:
            message = ChatMessage(role=role, content=content)
            await self._store.add_message(session_id, message)
            messages = await self._store.get_messages(session_id)
            logger.debug(f"Added {role} message to session {session_id} (total: {len(messages)})")
    
    async def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get chat history for a session (thread-safe)."""
        if not session_id:
            return []
        
        lock = await self._get_lock(session_id)
        async with lock:
            messages = await self._store.get_messages(session_id)
            return [msg.to_dict() for msg in messages]
    
    async def clear_session(self, session_id: str) -> None:
        """Clear chat history for a session (thread-safe)."""
        if not session_id:
            return
        
        lock = await self._get_lock(session_id)
        async with lock:
            await self._store.clear_session(session_id)
            # Clean up lock if session is cleared
            async with self._locks_lock:
                self._locks.pop(session_id, None)
            logger.info(f"Cleared session {session_id}")
    
    async def get_conversation_messages(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history in format suitable for Claude API (thread-safe).
        Returns list of {"role": "user"|"assistant", "content": "..."}
        """
        history = await self.get_history(session_id)
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in history
        ]


# Global session manager instance
_trimmer = MessageCountTrimmer(max_messages=10)
_store = InMemorySessionStore(trimmer=_trimmer)
session_manager = SessionManager(store=_store, max_messages=10)


def _extract_document_sources(sources: List[Dict[str, Any]]) -> List[DocumentSource]:
    """
    Extract unique document sources with page numbers and snippets from retrieved sources.
    
    Args:
        sources: List of source dictionaries from RAG response (includes snippets)
        
    Returns:
        List of DocumentSource objects with doc_id, pages_used, and snippet
    """
    doc_map: Dict[str, Dict[str, Any]] = {}
    
    for source in sources:
        doc_name = source.get('name', 'Unknown')
        pages_str = source.get('pages', 'N/A')
        snippet = source.get('snippet', '')
        
        # Parse page numbers from string like "3, 7, 12" or "N/A"
        pages = []
        if pages_str and pages_str != 'N/A':
            # Split by comma and convert to integers
            for page_str in pages_str.split(','):
                page_str = page_str.strip()
                try:
                    page_num = int(page_str)
                    pages.append(page_num)
                except ValueError:
                    # Skip non-numeric page labels
                    continue
        
        # Use filename as doc_id (remove path if present)
        doc_id = doc_name.split('/')[-1] if '/' in doc_name else doc_name
        
        # Initialize or update document entry
        if doc_id not in doc_map:
            doc_map[doc_id] = {
                'pages': set(),
                'snippets': []
            }
        
        # Add pages to the document's set
        doc_map[doc_id]['pages'].update(pages)
        
        # Collect snippets (keep first non-empty snippet, or best one)
        if snippet and snippet not in doc_map[doc_id]['snippets']:
            doc_map[doc_id]['snippets'].append(snippet)
    
    # Convert to DocumentSource objects with sorted page numbers and best snippet
    document_sources = []
    for doc_id, doc_data in doc_map.items():
        pages_set = doc_data['pages']
        snippets = doc_data['snippets']
        
        # Use first snippet (most relevant) or empty string
        best_snippet = snippets[0] if snippets else ""
        
        document_sources.append(DocumentSource(
            doc_id=doc_id,
            pages_used=sorted(list(pages_set)) if pages_set else [],
            snippet=best_snippet[:200]  # Ensure max 200 chars
        ))
    
    return document_sources


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    global rag_pipeline, db_manager
    
    # Startup
    logger.info("🚀 Starting FastAPI backend...")
    
    try:
        # Initialize database manager
        db_manager = PostgresManager()
        logger.info(f"✅ Database connection initialized (Google Cloud SQL)")
    except Exception as e:
        logger.warning(f"⚠️ Database initialization failed: {e}")
        db_manager = None
    
    # Initialize RAG pipeline
    try:
        # Determine storage path - check multiple locations
        possible_paths = [
            "latest_model",  # Current directory
            "../latest_model",  # Parent directory (for scripts/)
            "/workspace/latest_model",  # RunPod workspace
            "/workspace/ArrowSystems/latest_model",  # RunPod with ArrowSystems
            "/workspace/storage",  # Old storage location
            "./storage"  # Local storage
        ]
        
        storage_path = None
        for path in possible_paths:
            if os.path.exists(path):
                storage_path = path
                break
        
        if not storage_path:
            raise FileNotFoundError(
                "Index not found. Please run 'python ingest.py' first, "
                "or ensure the latest_model directory exists. "
                f"Checked paths: {possible_paths}"
            )
        
        logger.info(f"Using storage path: {storage_path}")
        
        # Use environment variable for cache directory if set
        cache_dir = os.getenv('HF_HOME', '/app/.cache/huggingface/hub')
        if cache_dir.endswith('huggingface'):
            cache_dir = os.path.join(cache_dir, 'hub')
        
        # Initialize RAG pipeline
        rag_pipeline = initialize_rag_pipeline(
            storage_dir=storage_path,
            cache_dir=cache_dir,
            db_manager=db_manager
        )
        logger.info("✅ RAG pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG pipeline: {e}")
        raise
    
    # Set startup time for uptime calculation
    app.state.start_time = time.time()
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down FastAPI backend...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="DuraFlex Technical Assistant API",
    description="Production-ready RAG API for DuraFlex technical documentation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="User query", min_length=1, max_length=5000)
    session_id: Optional[str] = Field(None, description="Session ID for chat history (auto-generated if not provided)")
    top_k: int = Field(10, description="Number of chunks to retrieve", ge=1, le=50)
    alpha: float = Field(0.5, description="Hybrid search weight (0=BM25 only, 1=dense only)", ge=0.0, le=1.0)
    metadata_filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")
    dynamic_windowing: bool = Field(True, description="Enable dynamic context windowing")


class SourceInfo(BaseModel):
    """Source information model."""
    id: str
    name: str
    pages: str
    content_type: str


class DocumentSource(BaseModel):
    """Document source with page numbers and snippet."""
    doc_id: str  # File path or filename
    pages_used: List[int]  # List of page numbers
    snippet: str = ""  # Short extract/snippet (~200 chars) for quick relevance check


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str
    answer: str
    reasoning: str
    sources: List[SourceInfo]
    document_sources: List[DocumentSource]  # Document provenance with page numbers
    confidence: float
    intent_type: str
    intent_confidence: float
    response_time_ms: int
    session_id: Optional[str] = None
    cache_hit: bool = False


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    rag_pipeline_initialized: bool
    database_connected: bool
    uptime_seconds: float


class CacheStatsResponse(BaseModel):
    """Cache statistics response model."""
    query_cache: Dict[str, Any]
    semantic_cache: Dict[str, Any]
    document_evaluator: Dict[str, Any]
    answer_generator: Dict[str, Any]


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "DuraFlex Technical Assistant API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global rag_pipeline, db_manager
    
    return HealthResponse(
        status="healthy" if rag_pipeline and rag_pipeline.is_initialized() else "unhealthy",
        rag_pipeline_initialized=rag_pipeline is not None and rag_pipeline.is_initialized(),
        database_connected=db_manager is not None,
        uptime_seconds=time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    )


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base using RAG pipeline with session-based chat memory.
    
    This endpoint accepts a query and returns a structured response with
    answer, reasoning, sources, and metadata. If session_id is provided,
    chat history is maintained and included in the LLM context.
    """
    global rag_pipeline
    
    if not rag_pipeline or not rag_pipeline.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Please check server logs."
        )
    
    try:
        start_time = time.time()
        
        # Generate or use provided session_id
        session_id = request.session_id
        if not session_id:
            # Generate a simple session ID (in production, use proper UUID)
            session_id = f"session_{int(time.time() * 1000)}"
        
        # Get chat history for this session (last 10 messages)
        chat_history = await session_manager.get_conversation_messages(session_id)
        logger.info(f"Session {session_id}: {len(chat_history)} previous messages")
        
        # Execute RAG query with chat history
        # Note: Retrieval uses only current query, but LLM gets chat history
        response = rag_pipeline.query(
            query=request.query,
            top_k=request.top_k,
            alpha=request.alpha,
            metadata_filters=request.metadata_filters,
            dynamic_windowing=request.dynamic_windowing,
            chat_history=chat_history  # Pass chat history to pipeline
        )
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Store messages in session history
        await session_manager.add_message(session_id, "user", request.query)
        await session_manager.add_message(session_id, "assistant", response.answer)
        
        # Convert sources to response format
        sources = [
            SourceInfo(
                id=source['id'],
                name=source['name'],
                pages=source['pages'],
                content_type=source['content_type']
            )
            for source in response.sources
        ]
        
        # Extract document sources with page numbers for provenance
        document_sources = _extract_document_sources(response.sources)
        
        # Save to database if available
        if db_manager:
            try:
                query_id = db_manager.save_query(
                    user="api_user",  # Could be extracted from auth headers
                    query_text=request.query,
                    answer_text=response.answer,
                    intent_type=response.intent.intent_type,
                    intent_confidence=response.intent.confidence,
                    sources=[s['name'] for s in response.sources],
                    confidence=response.confidence,
                    response_time_ms=response_time_ms,
                    session_id=session_id
                )
                logger.info(f"Query saved to database: {query_id}")
            except Exception as e:
                logger.warning(f"Failed to save query to database: {e}")
        
        return QueryResponse(
            query=response.query,
            answer=response.answer,
            reasoning=response.reasoning,
            sources=sources,
            document_sources=document_sources,
            confidence=response.confidence,
            intent_type=response.intent.intent_type,
            intent_confidence=response.intent.confidence,
            response_time_ms=response_time_ms,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/saved")
async def get_saved_responses(limit: int = 50, min_helpful_count: int = 1):
    """
    Get saved/validated responses that have been marked as helpful.
    
    Args:
        limit: Maximum number of saved responses to return (default: 50)
        min_helpful_count: Minimum helpful_count to include (default: 2)
    
    Returns:
        List of saved responses with query, answer, sources, and metadata
    """
    global db_manager
    
    if not db_manager:
        return {
            "status": "no_database",
            "message": "Database not available",
            "saved": []
        }
    
    try:
        validated_entries = db_manager.get_all_validated_qna(limit=limit, min_helpful_count=min_helpful_count)
        
        # Format for frontend
        formatted_responses = []
        for entry in validated_entries:
            formatted_responses.append({
                "id": entry.get('query_hash', ''),
                "query": entry.get('query_text', ''),
                "answer": entry.get('answer_text', ''),
                "sources": entry.get('sources', []),
                "helpful_count": entry.get('helpful_count', 0),
                "unhelpful_count": entry.get('unhelpful_count', 0),
                "last_used": str(entry.get('last_used', '')),
                "first_validated": str(entry.get('first_validated', '')),
                "created_at": str(entry.get('created_at', ''))
            })
        
        return {
            "status": "success",
            "count": len(formatted_responses),
            "saved": formatted_responses
        }
        
    except Exception as e:
        logger.error(f"Error fetching saved responses: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch saved responses: {str(e)}"
        )


@app.get("/history")
async def get_chat_history(user: str = "api_user", limit: int = 50):
    """
    Get chat history for a user.
    
    Args:
        user: Username (default: "api_user")
        limit: Maximum number of queries to return (default: 50)
    
    Returns:
        List of query history items with query, answer, timestamp, and metadata
    """
    global db_manager
    
    if not db_manager:
        return {
            "status": "no_database",
            "message": "Database not available",
            "history": []
        }
    
    try:
        history = db_manager.get_user_query_history(user=user, limit=limit)
        
        # Format history for frontend
        formatted_history = []
        for item in history:
            metadata = item.get('metadata', {})
            if isinstance(metadata, str):
                import json
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            # Get query_id - prefer integer ID, fall back to string ID from metadata
            query_id = str(item.get('query_id', ''))
            if not query_id and isinstance(metadata, dict):
                query_id = str(metadata.get('query_id', ''))
            if not query_id:
                query_id = f"query_{item.get('id', 'unknown')}"
            
            formatted_history.append({
                "id": query_id,
                "query": str(item.get('query_text', '')) or '',
                "answer": str(item.get('response_text', '')) or '',
                "timestamp": str(item.get('timestamp', '')) or '',
                "intent_type": metadata.get('intent_type', 'unknown') if isinstance(metadata, dict) else 'unknown',
                "confidence": float(metadata.get('confidence', 0.0)) if isinstance(metadata, dict) else 0.0,
                "sources": metadata.get('sources', []) if isinstance(metadata, dict) else [],
                "response_time_ms": int(item.get('response_time_ms', 0)) or 0
            })
        
        return {
            "status": "success",
            "count": len(formatted_history),
            "history": formatted_history
        }
        
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch chat history: {str(e)}"
        )


@app.post("/session/{session_id}/clear")
async def clear_session(session_id: str):
    """
    Clear chat history for a specific session.
    
    Args:
        session_id: Session ID to clear
    """
    await session_manager.clear_session(session_id)
    return {
        "status": "success",
        "message": f"Session {session_id} cleared",
        "session_id": session_id
    }


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """
    Get chat history for a specific session.
    
    Args:
        session_id: Session ID
    """
    history = await session_manager.get_history(session_id)
    return {
        "status": "success",
        "session_id": session_id,
        "message_count": len(history),
        "history": history
    }


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Get cache statistics."""
    global rag_pipeline
    
    if not rag_pipeline or not rag_pipeline.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized"
        )
    
    try:
        stats = rag_pipeline.get_cache_stats()
        return CacheStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting cache stats: {str(e)}"
        )


@app.post("/cache/clear")
async def clear_caches():
    """Clear all caches."""
    global rag_pipeline
    
    if not rag_pipeline or not rag_pipeline.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized"
        )
    
    try:
        rag_pipeline.clear_caches()
        return {"message": "All caches cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing caches: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing caches: {str(e)}"
        )


@app.get("/documents/{filename:path}")
async def serve_document(filename: str):
    """
    Serve PDF documents from the data directory.
    
    Args:
        filename: PDF filename (e.g., "DuraFlex Installation Guide.pdf")
    
    Returns:
        PDF file content
    """
    import os
    from fastapi.responses import FileResponse
    
    # URL decode the filename (handles %20 for spaces, etc.)
    import urllib.parse
    filename = urllib.parse.unquote(filename)
    
    # Security: prevent directory traversal
    filename = os.path.basename(filename)
    
    # Find the file in data directory
    possible_paths = [
        os.path.join("data", filename),
        os.path.join("/app/data", filename),
        os.path.join("../data", filename),
        os.path.join("/workspace/data", filename),
        os.path.join("/workspace/ArrowSystems/data", filename),
        os.path.join("./data", filename)
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.isfile(path):
            file_path = path
            logger.info(f"📄 Found document at: {file_path}")
            break
    
    if not file_path:
        logger.warning(f"⚠️ Document not found: '{filename}'. Searched paths: {possible_paths}")
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found. Available files in data directory."
        )
    
    # Verify it's a PDF
    if not filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=filename
    )


@app.get("/models/info")
async def get_models_info():
    """Get information about loaded models."""
    global rag_pipeline
    
    if not rag_pipeline or not rag_pipeline.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized"
        )
    
    try:
        orchestrator = rag_pipeline.orchestrator
        
        return {
            "embedding_model": {
                "name": getattr(orchestrator.embed_model, 'model_name', 'Unknown'),
                "device": getattr(orchestrator.embed_model, 'device', 'Unknown')
            },
            "reranker": {
                "available": orchestrator.reranker is not None,
                "name": getattr(orchestrator.reranker, 'model_name', 'Unknown') if orchestrator.reranker else None
            },
            "llm_evaluation": {
                "enabled": orchestrator.document_evaluator is not None,
                "model": getattr(orchestrator.document_evaluator, 'model_name', 'Unknown') if orchestrator.document_evaluator else None
            },
            "llm_answers": {
                "enabled": orchestrator.answer_generator is not None,
                "model": getattr(orchestrator.answer_generator, 'model_name', 'Unknown') if orchestrator.answer_generator else None
            }
        }
    except Exception as e:
        logger.error(f"Error getting models info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting models info: {str(e)}"
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found. Check /docs for available endpoints."}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs for details."}
    )


# Startup time is now set in lifespan handler above


def main():
    """Main function to run the FastAPI server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DuraFlex Technical Assistant API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8501, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting FastAPI server on {args.host}:{args.port}")
    logger.info(f"📚 API documentation available at http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info"
    )


if __name__ == "__main__":
    main()
