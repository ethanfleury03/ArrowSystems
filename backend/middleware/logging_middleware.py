"""
Logging Middleware for FastAPI

Logs all HTTP requests with structured logging including:
- request_id
- user_id (if available)
- role (if available)
- path, method, status_code, latency_ms
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..logging_context import set_request_id, set_user_id, set_user_role, clear_context
from ..logging_config import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all HTTP requests with structured logging."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log with structured logging."""
        # Generate or read request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Set request ID in context
        set_request_id(request_id)
        
        # Try to extract user from JWT token if present
        user_id = None
        role = None
        
        try:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.replace("Bearer ", "")
                from ..security import decode_access_token
                try:
                    payload = decode_access_token(token)
                    user_id = payload.get("email") or payload.get("user_id") or payload.get("sub")
                    role = payload.get("role")
                    if user_id:
                        set_user_id(str(user_id))
                    if role:
                        set_user_role(role)
                except Exception:
                    # Token invalid or expired - ignore
                    pass
        except Exception:
            # Failed to extract user - ignore
            pass
        
        # Start timing
        start_time = time.time()
        
        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            exception = None
        except Exception as e:
            status_code = 500
            exception = e
            # Re-raise the exception
            raise
        finally:
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Log the request
            # Note: structlog takes event name as first positional arg, so don't include "event" in dict
            log_data = {
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 2),
                "request_id": request_id,
            }
            
            if user_id:
                log_data["user_id"] = user_id
            if role:
                log_data["role"] = role
            
            # Add query parameters if present
            if request.url.query:
                log_data["query"] = request.url.query
            
            # Log based on status code
            if status_code >= 500:
                logger.error("http_request", **log_data, exc_info=exception)
            elif status_code >= 400:
                logger.warning("http_request", **log_data)
            else:
                logger.info("http_request", **log_data)
            
            # Clear context after request
            clear_context()
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response

