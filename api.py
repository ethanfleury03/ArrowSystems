"""
FastAPI Backend for DuraFlex Technical Assistant
Production-ready API server with RAG capabilities

This FastAPI application provides a REST API interface to the RAG system,
allowing frontend applications (like Next.js) to query the knowledge base.

Version: 1.0.0
Author: Arrow Systems Inc
"""

import os
import logging
import time
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from rag_pipeline import RAGPipeline, initialize_rag_pipeline, get_rag_pipeline
from utils.dynamodb_manager import DynamoDBManager

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
        use_aws = bool(os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'))
        db_manager = DynamoDBManager(local_mode=not use_aws)
        logger.info(f"✅ Database connection initialized ({'AWS DynamoDB' if use_aws else 'Local DynamoDB'})")
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
        
        # Initialize RAG pipeline
        rag_pipeline = initialize_rag_pipeline(
            storage_dir=storage_path,
            db_manager=db_manager
        )
        logger.info("✅ RAG pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG pipeline: {e}")
        raise
    
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
    query: str = Field(..., description="User query", min_length=1, max_length=1000)
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


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str
    answer: str
    reasoning: str
    sources: List[SourceInfo]
    confidence: float
    intent_type: str
    intent_confidence: float
    response_time_ms: int
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
    Query the knowledge base using RAG pipeline.
    
    This endpoint accepts a query and returns a structured response with
    answer, reasoning, sources, and metadata.
    """
    global rag_pipeline
    
    if not rag_pipeline or not rag_pipeline.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Please check server logs."
        )
    
    try:
        start_time = time.time()
        
        # Execute RAG query
        response = rag_pipeline.query(
            query=request.query,
            top_k=request.top_k,
            alpha=request.alpha,
            metadata_filters=request.metadata_filters,
            dynamic_windowing=request.dynamic_windowing
        )
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
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
                    session_id="api_session"  # Could be extracted from request
                )
                logger.info(f"Query saved to database: {query_id}")
            except Exception as e:
                logger.warning(f"Failed to save query to database: {e}")
        
        return QueryResponse(
            query=response.query,
            answer=response.answer,
            reasoning=response.reasoning,
            sources=sources,
            confidence=response.confidence,
            intent_type=response.intent.intent_type,
            intent_confidence=response.intent.confidence,
            response_time_ms=response_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


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


# Startup event
@app.on_event("startup")
async def startup_event():
    """Set startup time for uptime calculation."""
    app.state.start_time = time.time()


def main():
    """Main function to run the FastAPI server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DuraFlex Technical Assistant API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
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
