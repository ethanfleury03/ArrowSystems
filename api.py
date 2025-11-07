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

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

from rag_pipeline import RAGPipeline, initialize_rag_pipeline, get_rag_pipeline
from utils.postgres_manager import PostgresManager
from utils.query_summarizer import QuerySummarizer

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
query_summarizer = None  # Query summarization utility


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
    global rag_pipeline, db_manager, query_summarizer
    
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
    
    # Initialize query summarizer
    try:
        query_summarizer = QuerySummarizer(
            enabled=True,  # Enable by default
            min_length=500  # Summarize queries >500 chars
        )
        logger.info("✅ Query summarizer initialized")
    except Exception as e:
        logger.warning(f"⚠️ Failed to initialize query summarizer: {e}")
        query_summarizer = None
    
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
    matched_machine_name: Optional[str] = None  # Machine name matched in query (if >=95% similarity)


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


# Admin API Models
class DocumentInfo(BaseModel):
    """Document information model."""
    filename: str
    size_bytes: int
    uploaded_date: Optional[str] = None
    chunk_count: int
    file_path: str


class ChunkInfo(BaseModel):
    """Chunk information model."""
    chunk_id: str
    doc_title: str
    chunk_text: str  # Trimmed to 200 chars
    summary_exists: bool
    embedding_exists: bool
    page_label: Optional[str] = None
    content_type: Optional[str] = None


class ChunkDetail(BaseModel):
    """Detailed chunk information."""
    chunk_id: str
    doc_title: str
    chunk_text: str  # Full text
    summary: Optional[str] = None
    metadata: Dict[str, Any]
    page_label: Optional[str] = None
    content_type: Optional[str] = None


class SearchSandboxRequest(BaseModel):
    """Search sandbox request model."""
    query: str
    top_k: int = 10
    alpha: float = 0.5


class SearchSandboxResponse(BaseModel):
    """Search sandbox response model."""
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    machine_detection_fired: bool
    matched_machine_name: Optional[str] = None
    document_ids: List[str]
    total_chunks: int


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


@app.post("/summarize-query")
async def summarize_query_endpoint(request: Dict[str, Any]):
    """
    Summarize a long query before sending to RAG pipeline.
    Used by frontend to preprocess long user inputs (emails, error logs, etc.).
    """
    global query_summarizer
    
    if not query_summarizer:
        return JSONResponse(
            status_code=503,
            content={"detail": "Query summarization not available"}
        )
    
    query = request.get("query", "")
    if not query:
        return JSONResponse(
            status_code=400,
            content={"detail": "Query is required"}
        )
    
    try:
        summary, was_summarized, content_type = query_summarizer.summarize(query)
        
        return JSONResponse(content={
            "summary": summary,
            "was_summarized": was_summarized,
            "content_type": content_type,
            "original_length": len(query),
            "summarized_length": len(summary)
        })
    except Exception as e:
        logger.error(f"Error summarizing query: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to summarize query: {str(e)}"}
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
        
        # Log incoming query
        logger.info(f"📥 Received query: {request.query[:200]}{'...' if len(request.query) > 200 else ''}")
        
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
        
        # Track query for analytics
        try:
            from utils.query_tracker import log_query
            documents_retrieved = [s['name'] for s in response.sources]
            log_query(
                query_text=request.query,
                session_id=session_id,
                answer_text=response.answer,
                documents_retrieved=documents_retrieved,
                relevance_score=response.confidence,  # Using confidence as relevance score
                confidence=response.confidence,
                response_time_ms=response_time_ms,
                matched_machine_name=response.matched_machine_name,
                sources=response.sources
            )
        except Exception as e:
            logger.warning(f"Failed to log query for analytics: {e}")
        
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
            session_id=session_id,
            matched_machine_name=response.matched_machine_name
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
                "last_used": entry.get('last_used', ''),
                "first_validated": entry.get('first_validated', ''),
                "created_at": entry.get('created_at', '')
            })
        
        return {
            "status": "success",
            "count": len(formatted_responses),
            "saved": formatted_responses
        }
    except Exception as e:
        logger.error(f"Error fetching saved responses: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "saved": []
        }


@app.get("/history")
async def get_chat_history(user: str = "api_user", limit: int = 50):
    """
    Get chat history for a user.
    
    Args:
        user: User identifier (default: "api_user")
        limit: Maximum number of entries to return (default: 50)
    
    Returns:
        List of chat history entries
    """
    global db_manager
    
    if not db_manager:
        return {
            "status": "no_database",
            "message": "Database not available",
            "history": []
        }
    
    try:
        history = db_manager.get_query_history(user=user, limit=limit)
        
        # Format for frontend
        formatted_history = []
        for entry in history:
            formatted_history.append({
                "id": entry.get('id', ''),
                "query": entry.get('query_text', ''),
                "answer": entry.get('answer_text', ''),
                "timestamp": entry.get('created_at', ''),
                "intent_type": entry.get('intent_type', ''),
                "confidence": entry.get('confidence', 0.0),
                "sources": entry.get('sources', []),
                "response_time_ms": entry.get('response_time_ms', 0)
            })
        
        return {
            "status": "success",
            "count": len(formatted_history),
            "history": formatted_history
        }
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "history": []
        }


@app.get("/documents/{filename:path}")
async def serve_document(filename: str):
    """
    Serve document files (PDFs) from the data directory.
    Used by frontend to display source documents.
    """
    import urllib.parse
    
    # URL decode the filename
    filename = urllib.parse.unquote(filename)
    
    # Security: prevent directory traversal
    if '..' in filename or filename.startswith('/'):
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Try multiple possible data directory locations
    possible_paths = [
        "data",
        "../data",
        "/app/data",
        "/workspace/data",
        "/workspace/ArrowSystems/data"
    ]
    
    for base_path in possible_paths:
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            # Check file extension for security
            if not filename.lower().endswith(('.pdf', '.docx', '.md', '.markdown')):
                raise HTTPException(status_code=400, detail="Invalid file type")
            
            return FileResponse(
                file_path,
                media_type="application/pdf" if filename.lower().endswith('.pdf') else "application/octet-stream",
                filename=filename
            )
    
    raise HTTPException(status_code=404, detail=f"Document not found: {filename}")


@app.post("/session/{session_id}/clear")
async def clear_session(session_id: str):
    """Clear chat history for a session."""
    await session_manager.clear_session(session_id)
    return {"status": "success", "message": f"Session {session_id} cleared"}


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get chat history for a session."""
    history = await session_manager.get_history(session_id)
    return {"status": "success", "history": history}


# =============================================================================
# Admin API Endpoints
# =============================================================================

@app.get("/admin/documents")
async def get_all_documents():
    """
    Get all documents in the index with enhanced metadata.
    Returns list of documents with metadata including status, machine_model, etc.
    """
    global rag_pipeline
    
    if not rag_pipeline or not rag_pipeline.is_initialized():
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        from utils.document_metadata import get_document_metadata
        
        documents = []
        
        # Group chunks by document filename and count pages
        # Primary source: corpus_nodes (most reliable)
        doc_chunks = {}
        doc_pages = {}
        seen_filenames = set()
        
        if hasattr(rag_pipeline.orchestrator, 'retriever') and rag_pipeline.orchestrator.retriever:
            retriever = rag_pipeline.orchestrator.retriever
            if hasattr(retriever, 'corpus_nodes') and retriever.corpus_nodes:
                logger.info(f"Found {len(retriever.corpus_nodes)} nodes in corpus_nodes")
                for node_wrapper in retriever.corpus_nodes:
                    node = node_wrapper.node if hasattr(node_wrapper, 'node') else node_wrapper
                    if hasattr(node, 'metadata') and node.metadata:
                        filename = node.metadata.get('file_name', 'Unknown')
                        if filename and filename != 'Unknown':
                            seen_filenames.add(filename)
                            if filename not in doc_chunks:
                                doc_chunks[filename] = []
                                doc_pages[filename] = set()
                            doc_chunks[filename].append(node)
                            # Track unique pages
                            page_label = node.metadata.get('page_label')
                            if page_label:
                                try:
                                    page_num = int(str(page_label).split('.')[0])  # Get page number
                                    doc_pages[filename].add(page_num)
                                except:
                                    pass
        
        # Get document IDs from ALL sources (corpus_nodes, docstore, AND filesystem)
        # Combine all sources to ensure we get all documents
        doc_ids = []
        docstore = None
        all_filenames = set(seen_filenames)  # Start with corpus_nodes filenames
        
        # Source 1: Get from docstore
        if rag_pipeline.orchestrator.index and hasattr(rag_pipeline.orchestrator.index, 'docstore') and rag_pipeline.orchestrator.index.docstore:
            docstore = rag_pipeline.orchestrator.index.docstore
            docstore_ids = list(docstore.docs.keys())
            logger.info(f"Found {len(docstore_ids)} documents in docstore")
            
            # Extract filenames from docstore documents
            for doc_id in docstore_ids[:1000]:  # Limit to prevent memory issues
                try:
                    doc = docstore.get_document(doc_id)
                    if hasattr(doc, 'metadata') and doc.metadata:
                        filename = doc.metadata.get('file_name', doc_id)
                        if filename and filename != doc_id:
                            all_filenames.add(filename)
                except:
                    pass
        
        # Source 2: Get from filesystem (most comprehensive)
        logger.info("Scanning filesystem for documents...")
        data_dir = "data"
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.lower().endswith(('.pdf', '.docx', '.md', '.markdown')):
                    file_path = os.path.join(data_dir, filename)
                    if os.path.isfile(file_path):
                        all_filenames.add(filename)
            logger.info(f"Found {len(all_filenames)} total unique documents across all sources")
        
        # Convert to list for processing
        doc_ids = list(all_filenames)
        
        if seen_filenames:
            logger.info(f"Found {len(seen_filenames)} documents via corpus_nodes")
        if docstore:
            logger.info(f"Found documents in docstore")
        logger.info(f"Total unique documents: {len(doc_ids)}")
        
        if not doc_ids:
            logger.warning("No documents found in corpus_nodes, docstore, or filesystem")
            return {"documents": [], "total": 0}
        
        # Build document list
        for filename in doc_ids[:1000]:  # Limit to first 1000
            try:
                # Try to get document from docstore if available
                doc = None
                if docstore:
                    # Try to find doc_id that matches this filename
                    for doc_id in docstore.docs.keys():
                        try:
                            temp_doc = docstore.get_document(doc_id)
                            if hasattr(temp_doc, 'metadata') and temp_doc.metadata:
                                if temp_doc.metadata.get('file_name') == filename:
                                    doc = temp_doc
                                    break
                        except:
                            continue
                
                # Get file path and size
                file_path = os.path.join("data", filename)
                size_bytes = 0
                if os.path.exists(file_path):
                    size_bytes = os.path.getsize(file_path)
                
                # Count chunks for this document
                chunk_count = len(doc_chunks.get(filename, []))
                
                # Get page count
                page_count = len(doc_pages.get(filename, set()))
                if page_count == 0:
                    # Fallback: estimate from chunk count or use 1
                    page_count = max(1, chunk_count // 5)  # Rough estimate
                
                # Get file type
                file_ext = os.path.splitext(filename)[1].lower()
                file_type = file_ext[1:] if file_ext else 'pdf'  # Remove dot
                
                # Get metadata from document_metadata.json
                doc_metadata = get_document_metadata(filename)
                
                documents.append({
                    "filename": filename,
                    "size_bytes": size_bytes,
                    "uploaded_date": doc_metadata.get("last_ingestion_date"),
                    "chunk_count": chunk_count,
                    "page_count": page_count,
                    "file_path": file_path,
                    "file_type": file_type,
                    "is_active": doc_metadata.get("is_active", True),
                    "machine_model": doc_metadata.get("machine_model"),
                    "category": doc_metadata.get("category"),
                    "product_family": doc_metadata.get("product_family")
                })
            except Exception as e:
                logger.debug(f"Error processing document {doc_id}: {e}")
                continue
        
        # Remove duplicates by filename
        seen = set()
        unique_docs = []
        for doc in documents:
            if doc['filename'] not in seen:
                seen.add(doc['filename'])
                unique_docs.append(doc)
        
        return {"documents": unique_docs, "total": len(unique_docs)}
        
    except Exception as e:
        logger.error(f"Error fetching documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")


@app.get("/admin/chunks")
async def get_all_chunks(page: int = 1, page_size: int = 50):
    """
    Get paginated list of all chunks.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of chunks per page
    """
    global rag_pipeline
    
    if not rag_pipeline or not rag_pipeline.is_initialized():
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        retriever = rag_pipeline.orchestrator.retriever
        if not retriever or not hasattr(retriever, 'corpus_nodes') or not retriever.corpus_nodes:
            return {"chunks": [], "total": 0, "page": page, "page_size": page_size, "total_pages": 0}
        
        all_chunks = []
        
        for node_wrapper in retriever.corpus_nodes:
            node = node_wrapper.node if hasattr(node_wrapper, 'node') else node_wrapper
            
            # Get node metadata
            metadata = node.metadata if hasattr(node, 'metadata') else {}
            filename = metadata.get('file_name', 'Unknown')
            chunk_text = node.text if hasattr(node, 'text') else str(node)
            
            # Check if summary exists (from query summarizer cache)
            summary_exists = False
            if query_summarizer:
                # Check cache for this chunk
                import hashlib
                chunk_hash = hashlib.md5(chunk_text.encode('utf-8')).hexdigest()
                cache_path = query_summarizer._get_cache_path(chunk_hash)
                summary_exists = cache_path.exists()
            
            # Check if embedding exists (node has embedding)
            embedding_exists = hasattr(node, 'embedding') and node.embedding is not None
            
            chunk_id = node.node_id if hasattr(node, 'node_id') else str(id(node))
            
            all_chunks.append({
                "chunk_id": chunk_id,
                "doc_title": filename,
                "chunk_text": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                "summary_exists": summary_exists,
                "embedding_exists": embedding_exists,
                "page_label": metadata.get('page_label'),
                "content_type": metadata.get('content_type', 'text')
            })
        
        # Paginate
        total = len(all_chunks)
        total_pages = (total + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_chunks = all_chunks[start_idx:end_idx]
        
        return {
            "chunks": paginated_chunks,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages
        }
        
    except Exception as e:
        logger.error(f"Error fetching chunks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch chunks: {str(e)}")


@app.get("/admin/chunks/{chunk_id}")
async def get_chunk_detail(chunk_id: str):
    """Get detailed information for a specific chunk."""
    global rag_pipeline
    
    if not rag_pipeline or not rag_pipeline.is_initialized():
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        retriever = rag_pipeline.orchestrator.retriever
        if not retriever or not hasattr(retriever, 'corpus_nodes'):
            raise HTTPException(status_code=404, detail="Chunk not found")
        
        # Find the chunk
        for node_wrapper in retriever.corpus_nodes:
            node = node_wrapper.node if hasattr(node_wrapper, 'node') else node_wrapper
            current_id = node.node_id if hasattr(node, 'node_id') else str(id(node))
            
            if current_id == chunk_id:
                metadata = node.metadata if hasattr(node, 'metadata') else {}
                chunk_text = node.text if hasattr(node, 'text') else str(node)
                
                # Get summary if exists
                summary = None
                if query_summarizer:
                    import hashlib
                    chunk_hash = hashlib.md5(chunk_text.encode('utf-8')).hexdigest()
                    cached_summary = query_summarizer._load_from_cache(chunk_hash)
                    if cached_summary:
                        summary = cached_summary
                
                return {
                    "chunk_id": chunk_id,
                    "doc_title": metadata.get('file_name', 'Unknown'),
                    "chunk_text": chunk_text,
                    "summary": summary,
                    "metadata": metadata,
                    "page_label": metadata.get('page_label'),
                    "content_type": metadata.get('content_type', 'text')
                }
        
        raise HTTPException(status_code=404, detail="Chunk not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching chunk detail: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch chunk: {str(e)}")


@app.get("/admin/documents/{filename}/chunks")
async def get_document_chunks(filename: str):
    """Get all chunks for a specific document."""
    global rag_pipeline
    
    if not rag_pipeline or not rag_pipeline.is_initialized():
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        import urllib.parse
        filename = urllib.parse.unquote(filename)
        
        retriever = rag_pipeline.orchestrator.retriever
        if not retriever or not hasattr(retriever, 'corpus_nodes'):
            return {"chunks": [], "total": 0}
        
        document_chunks = []
        
        for node_wrapper in retriever.corpus_nodes:
            node = node_wrapper.node if hasattr(node_wrapper, 'node') else node_wrapper
            metadata = node.metadata if hasattr(node, 'metadata') else {}
            
            if metadata.get('file_name') == filename:
                chunk_id = node.node_id if hasattr(node, 'node_id') else str(id(node))
                chunk_text = node.text if hasattr(node, 'text') else str(node)
                
                document_chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                    "page_label": metadata.get('page_label'),
                    "content_type": metadata.get('content_type', 'text')
                })
        
        return {"chunks": document_chunks, "total": len(document_chunks)}
        
    except Exception as e:
        logger.error(f"Error fetching document chunks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch document chunks: {str(e)}")


@app.post("/admin/documents/{filename}/toggle")
async def toggle_document_status(filename: str, request: Dict[str, Any]):
    """
    Enable or disable a document.
    Inactive documents are excluded from search retrieval.
    """
    global rag_pipeline
    
    if not rag_pipeline or not rag_pipeline.is_initialized():
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        import urllib.parse
        filename = urllib.parse.unquote(filename)
        
        # Security check
        if '..' in filename or filename.startswith('/'):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        is_active = request.get("is_active", True)
        
        from utils.document_metadata import set_document_active
        set_document_active(filename, is_active)
        
        status = "enabled" if is_active else "disabled"
        logger.info(f"Document {filename} {status}")
        
        return {
            "status": "success",
            "message": f"Document {filename} {status}",
            "is_active": is_active
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling document status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to toggle document status: {str(e)}")


@app.post("/admin/documents/{filename}/metadata")
async def update_document_metadata_endpoint(filename: str, request: Dict[str, Any]):
    """
    Update document metadata (machine_model, category, product_family).
    """
    global rag_pipeline
    
    if not rag_pipeline or not rag_pipeline.is_initialized():
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        import urllib.parse
        filename = urllib.parse.unquote(filename)
        
        # Security check
        if '..' in filename or filename.startswith('/'):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Extract allowed metadata fields
        updates = {}
        allowed_fields = ["machine_model", "category", "product_family"]
        for field in allowed_fields:
            if field in request:
                updates[field] = request[field]
        
        if not updates:
            raise HTTPException(status_code=400, detail="No valid metadata fields provided")
        
        from utils.document_metadata import update_document_metadata
        update_document_metadata(filename, updates)
        
        logger.info(f"Updated metadata for {filename}: {updates}")
        
        return {
            "status": "success",
            "message": f"Metadata updated for {filename}",
            "metadata": updates
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating metadata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update metadata: {str(e)}")


@app.delete("/admin/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete a document completely from:
    - data storage (data/ and data/original_pdfs/)
    - vector store entries
    - docstore index
    - metadata file
    Then reload RAG pipeline.
    """
    global rag_pipeline
    
    if not rag_pipeline or not rag_pipeline.is_initialized():
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        import urllib.parse
        filename = urllib.parse.unquote(filename)
        
        # Security check
        if '..' in filename or filename.startswith('/'):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Delete file from data directories
        data_path = os.path.join("data", filename)
        original_path = os.path.join("data/original_pdfs", filename)
        
        deleted_files = []
        if os.path.exists(data_path):
            os.remove(data_path)
            deleted_files.append(data_path)
        if os.path.exists(original_path):
            os.remove(original_path)
            deleted_files.append(original_path)
        
        # Delete metadata
        from utils.document_metadata import delete_document_metadata
        delete_document_metadata(filename)
        
        # Remove from vector store and docstore
        # Note: LlamaIndex doesn't support direct deletion, so we need to rebuild
        # For now, mark as deleted and filter in retrieval
        # In production, you'd want to trigger a re-index
        
        # Reload RAG pipeline
        if rag_pipeline:
            try:
                # Determine storage path
                possible_paths = [
                    "latest_model",
                    "../latest_model",
                    "/workspace/latest_model",
                    "/workspace/ArrowSystems/latest_model",
                    "/workspace/storage",
                    "./storage"
                ]
                
                storage_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        storage_path = path
                        break
                
                if storage_path:
                    logger.info("Reloading RAG pipeline after document deletion...")
                    rag_pipeline.orchestrator.load_index(storage_dir=storage_path)
                    logger.info("✅ RAG pipeline reloaded")
            except Exception as e:
                logger.warning(f"Failed to reload RAG pipeline: {e}")
        
        logger.info(f"Deleted document: {filename} (files: {deleted_files})")
        
        return {
            "status": "success",
            "message": f"Document {filename} deleted completely. RAG pipeline reloaded.",
            "deleted_files": deleted_files
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


# =============================================================================
# Admin Query Analytics Endpoints
# =============================================================================

@app.get("/admin/queries")
async def get_all_queries_admin(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    machine_type: Optional[str] = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
    limit: int = 100,
    offset: int = 0,
    sort_by: str = "timestamp",
    sort_order: str = "desc"
):
    """
    Get all queries with filtering and sorting for admin analytics.
    """
    try:
        from utils.query_tracker import get_all_queries
        
        result = get_all_queries(
            start_date=start_date,
            end_date=end_date,
            machine_type=machine_type,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching queries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch queries: {str(e)}")


@app.get("/admin/queries/failed")
async def get_failed_queries_admin(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    machine_type: Optional[str] = None,
    include_resolved: bool = False,
    limit: int = 100,
    offset: int = 0
):
    """
    Get failed queries (low confidence or no documents retrieved).
    """
    try:
        from utils.query_tracker import get_failed_queries
        
        result = get_failed_queries(
            start_date=start_date,
            end_date=end_date,
            machine_type=machine_type,
            include_resolved=include_resolved,
            limit=limit,
            offset=offset
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching failed queries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch failed queries: {str(e)}")


@app.post("/admin/queries/mark_resolved")
async def mark_query_resolved(request: Dict[str, Any]):
    """
    Mark a failed query as resolved.
    """
    try:
        from utils.query_tracker import mark_query_resolved
        
        query_id = request.get("query_id")
        if not query_id:
            raise HTTPException(status_code=400, detail="query_id is required")
        
        success = mark_query_resolved(query_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Query not found")
        
        return {
            "status": "success",
            "message": f"Query {query_id} marked as resolved"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking query as resolved: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to mark query as resolved: {str(e)}")


@app.get("/admin/queries/stats")
async def get_query_stats():
    """
    Get aggregate statistics about queries.
    """
    try:
        from utils.query_tracker import get_query_stats
        
        stats = get_query_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error fetching query stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch query stats: {str(e)}")


@app.post("/admin/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF or DOCX) and trigger single-file ingestion.
    Saves file to data/original_pdfs/ and ingests it into the existing index.
    """
    # Security check
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    allowed_extensions = ['.pdf', '.docx', '.md', '.markdown']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}")
    
    try:
        # Save file to data/original_pdfs/ directory
        original_pdfs_dir = "data/original_pdfs"
        os.makedirs(original_pdfs_dir, exist_ok=True)
        
        # Also save to data/ for compatibility
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Read file content
        content = await file.read()
        
        # Save to both locations
        original_path = os.path.join(original_pdfs_dir, file.filename)
        data_path = os.path.join(data_dir, file.filename)
        
        with open(original_path, "wb") as f:
            f.write(content)
        
        with open(data_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Uploaded file: {original_path} ({len(content)} bytes)")
        
        # Import single-file ingestion utility
        from utils.single_file_ingestion import ingest_single_file
        
        # Determine storage directory (same logic as main initialization)
        possible_paths = [
            "latest_model",
            "../latest_model",
            "/workspace/latest_model",
            "/workspace/ArrowSystems/latest_model",
            "/workspace/storage",
            "./storage"
        ]
        
        storage_path = None
        for path in possible_paths:
            if os.path.exists(path):
                storage_path = path
                break
        
        if not storage_path:
            raise HTTPException(
                status_code=503,
                detail="Index not found. Please ensure latest_model directory exists."
            )
        
        # Use environment variable for cache directory if set
        cache_dir = os.getenv('HF_HOME', '/app/.cache/huggingface/hub')
        if cache_dir.endswith('huggingface'):
            cache_dir = os.path.join(cache_dir, 'hub')
        
        # Ingest the single file
        logger.info(f"Starting ingestion for {file.filename}...")
        result = ingest_single_file(
            file_path=data_path,  # Use data/ path for ingestion
            storage_dir=storage_path,
            cache_dir=cache_dir,
            enable_rewriting=False  # Can be made configurable
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed: {result.get('error', 'Unknown error')}"
            )
        
        # Reload RAG pipeline to pick up new document
        global rag_pipeline
        if rag_pipeline and rag_pipeline.is_initialized():
            logger.info("Reloading RAG pipeline to include new document...")
            try:
                rag_pipeline.orchestrator.load_index(storage_dir=storage_path)
                logger.info("✅ RAG pipeline reloaded successfully")
            except Exception as e:
                logger.warning(f"Failed to reload RAG pipeline: {e}. New document may not be immediately searchable.")
        
        return {
            "status": "success",
            "message": f"File {file.filename} uploaded and ingested successfully.",
            "file_path": original_path,
            "size_bytes": len(content),
            "doc_id": result["doc_id"],
            "filename": result["filename"],
            "page_count": result["page_count"],
            "chunk_count": result["chunk_count"],
            "text_chunks": result.get("text_chunks", 0),
            "non_text_chunks": result.get("non_text_chunks", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading/ingesting document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload/ingest document: {str(e)}")


@app.post("/admin/chunks/{chunk_id}/regenerate-summary")
async def regenerate_chunk_summary(chunk_id: str):
    """Regenerate summary for a specific chunk."""
    global rag_pipeline, query_summarizer
    
    if not query_summarizer:
        raise HTTPException(status_code=503, detail="Query summarizer not available")
    
    if not rag_pipeline or not rag_pipeline.is_initialized():
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        # Find the chunk
        retriever = rag_pipeline.orchestrator.retriever
        if not retriever or not hasattr(retriever, 'corpus_nodes'):
            raise HTTPException(status_code=404, detail="Chunk not found")
        
        chunk_text = None
        for node_wrapper in retriever.corpus_nodes:
            node = node_wrapper.node if hasattr(node_wrapper, 'node') else node_wrapper
            current_id = node.node_id if hasattr(node, 'node_id') else str(id(node))
            
            if current_id == chunk_id:
                chunk_text = node.text if hasattr(node, 'text') else str(node)
                break
        
        if not chunk_text:
            raise HTTPException(status_code=404, detail="Chunk not found")
        
        # Generate summary
        summary, was_summarized, _ = query_summarizer.summarize(chunk_text)
        
        return {
            "status": "success",
            "chunk_id": chunk_id,
            "summary": summary,
            "was_summarized": was_summarized
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error regenerating summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to regenerate summary: {str(e)}")


@app.delete("/admin/chunks/{chunk_id}")
async def delete_chunk(chunk_id: str):
    """
    Delete a chunk from the index.
    Note: This requires re-indexing to take effect.
    """
    # Note: LlamaIndex doesn't support direct chunk deletion
    # This would require re-indexing without that chunk
    # For now, return a message indicating re-index is needed
    
    return {
        "status": "info",
        "message": "Chunk deletion requires re-indexing. Please remove the source document and re-run ingest.py"
    }


@app.get("/admin/summaries/missing")
async def get_missing_summaries():
    """Get all chunks that are missing summaries."""
    global rag_pipeline, query_summarizer
    
    if not rag_pipeline or not rag_pipeline.is_initialized():
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    if not query_summarizer:
        return {"chunks": [], "total": 0}
    
    try:
        retriever = rag_pipeline.orchestrator.retriever
        if not retriever or not hasattr(retriever, 'corpus_nodes'):
            return {"chunks": [], "total": 0}
        
        missing_summaries = []
        
        for node_wrapper in retriever.corpus_nodes:
            node = node_wrapper.node if hasattr(node_wrapper, 'node') else node_wrapper
            metadata = node.metadata if hasattr(node, 'metadata') else {}
            chunk_text = node.text if hasattr(node, 'text') else str(node)
            
            # Check if summary exists
            import hashlib
            chunk_hash = hashlib.md5(chunk_text.encode('utf-8')).hexdigest()
            cache_path = query_summarizer._get_cache_path(chunk_hash)
            summary_exists = cache_path.exists()
            
            if not summary_exists:
                chunk_id = node.node_id if hasattr(node, 'node_id') else str(id(node))
                missing_summaries.append({
                    "chunk_id": chunk_id,
                    "doc_title": metadata.get('file_name', 'Unknown'),
                    "chunk_text": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                    "page_label": metadata.get('page_label'),
                    "content_type": metadata.get('content_type', 'text')
                })
        
        return {"chunks": missing_summaries, "total": len(missing_summaries)}
        
    except Exception as e:
        logger.error(f"Error fetching missing summaries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch missing summaries: {str(e)}")


@app.post("/admin/summaries/generate-batch")
async def generate_batch_summaries():
    """Generate summaries for all chunks missing summaries."""
    global rag_pipeline, query_summarizer
    
    if not query_summarizer:
        raise HTTPException(status_code=503, detail="Query summarizer not available")
    
    if not rag_pipeline or not rag_pipeline.is_initialized():
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        retriever = rag_pipeline.orchestrator.retriever
        if not retriever or not hasattr(retriever, 'corpus_nodes'):
            return {"generated": 0, "total": 0, "errors": []}
        
        generated = 0
        errors = []
        
        for node_wrapper in retriever.corpus_nodes:
            node = node_wrapper.node if hasattr(node_wrapper, 'node') else node_wrapper
            chunk_text = node.text if hasattr(node, 'text') else str(node)
            
            # Check if summary exists
            import hashlib
            chunk_hash = hashlib.md5(chunk_text.encode('utf-8')).hexdigest()
            cache_path = query_summarizer._get_cache_path(chunk_hash)
            
            if not cache_path.exists():
                try:
                    # Generate summary
                    summary, was_summarized, _ = query_summarizer.summarize(chunk_text)
                    if was_summarized:
                        generated += 1
                except Exception as e:
                    errors.append(str(e))
                    logger.warning(f"Failed to generate summary: {e}")
        
        return {
            "status": "success",
            "generated": generated,
            "total": len(retriever.corpus_nodes),
            "errors": errors
        }
        
    except Exception as e:
        logger.error(f"Error generating batch summaries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate batch summaries: {str(e)}")


@app.post("/admin/search-sandbox", response_model=SearchSandboxResponse)
async def search_sandbox(request: SearchSandboxRequest):
    """
    Search sandbox endpoint for admin testing.
    Returns detailed retrieval information for debugging.
    """
    global rag_pipeline
    
    if not rag_pipeline or not rag_pipeline.is_initialized():
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        # Execute search
        response = rag_pipeline.query(
            query=request.query,
            top_k=request.top_k,
            alpha=request.alpha
        )
        
        # Extract chunk details
        retrieved_chunks = []
        document_ids = set()
        
        for source in response.sources:
            doc_name = source.get('name', 'Unknown')
            document_ids.add(doc_name)
            
            retrieved_chunks.append({
                "doc_id": doc_name,
                "pages": source.get('pages', 'N/A'),
                "content_type": source.get('content_type', 'text'),
                "source_id": source.get('id', '')
            })
        
        # Check machine detection
        machine_detection_fired = response.matched_machine_name is not None
        
        return SearchSandboxResponse(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            machine_detection_fired=machine_detection_fired,
            matched_machine_name=response.matched_machine_name,
            document_ids=list(document_ids),
            total_chunks=len(retrieved_chunks)
        )
        
    except Exception as e:
        logger.error(f"Error in search sandbox: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Startup time is now set in lifespan handler above


def main():
    """Main function to run the FastAPI server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DuraFlex Technical Assistant API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8501, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
