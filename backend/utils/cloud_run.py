"""
Cloud Run detection and ingestion control utilities.
"""

import os


def is_cloud_run() -> bool:
    """
    Detect if running on Google Cloud Run.
    
    Cloud Run sets K_SERVICE and K_REVISION environment variables.
    """
    return os.getenv("K_SERVICE") is not None or os.getenv("K_REVISION") is not None


def is_ingestion_enabled() -> bool:
    """
    Check if ingestion is enabled via environment variable.
    
    Defaults to False for safety (especially on Cloud Run).
    """
    return os.getenv("ENABLE_INGESTION_ON_STARTUP", "false").lower() == "true"


def should_skip_ingestion() -> bool:
    """
    Determine if ingestion should be skipped.
    
    Returns True if:
    - We're on Cloud Run (ingestion must always be skipped on Cloud Run)
    - OR ENABLE_INGESTION_ON_STARTUP is not explicitly set to "true"
    """
    # Ingestion must always be skipped on Cloud Run
    if is_cloud_run():
        return True
    # For other environments, check the flag
    return not is_ingestion_enabled()

