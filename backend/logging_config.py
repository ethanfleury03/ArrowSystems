"""
Logging Configuration - Structured logging with structlog

Configures structlog for JSON output in production and pretty output in development.
Automatically includes request_id, user_id, and role from context variables.
"""

import os
import sys
import logging
from typing import Any, Dict, Optional
import structlog
from structlog.contextvars import merge_contextvars


def configure_logging(environment: Optional[str] = None) -> None:
    """
    Configure structlog for the application.
    
    Args:
        environment: Environment name ('prod', 'dev', 'local', etc.)
                    If None, tries to detect from ENV or NODE_ENV env vars.
    """
    if environment is None:
        environment = os.getenv('ENV', os.getenv('NODE_ENV', 'dev')).lower()
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    
    # Configure processors based on environment
    if environment in ('prod', 'production', 'cloud'):
        # Production: JSON output for Cloud Run
        processors = [
            merge_contextvars,  # Merge contextvars (request_id, user_id, role)
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()  # JSON output for Cloud Run
        ]
    else:
        # Development: Pretty console output with colors
        processors = [
            merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True)  # Pretty colored output
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Get logger and log startup
    logger = structlog.get_logger()
    logger.info(
        "logging_configured",
        environment=environment,
        output_format="json" if environment in ('prod', 'production', 'cloud') else "console",
    )


def get_logger(name: str = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structlog logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Bound logger that automatically includes context variables
    """
    return structlog.get_logger(name)

