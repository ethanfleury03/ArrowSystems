"""
Google Cloud Run Jobs API utilities.

Provides functions to trigger Cloud Run Jobs programmatically.
"""

import os
from typing import Optional, Dict, Any

from backend.config.env import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)


def run_ticket_reindex_job() -> Optional[str]:
    """
    Trigger the ticket cache reindex Cloud Run Job.
    
    Uses Cloud Run Jobs API v2 to execute the job.
    
    Returns:
        Execution name (e.g., "ticket-cache-reindex-12345") if successful, None otherwise
        
    Raises:
        Exception: If job trigger fails
    """
    try:
        from google.auth import default
        from google.auth.transport.requests import Request
        import requests
    except ImportError:
        logger.error("[TICKET_REINDEX] google-auth or requests not installed - cannot trigger job")
        raise RuntimeError("Required dependencies not installed: google-auth, requests")
    
    project_id = settings.GCP_PROJECT_ID
    region = settings.GCP_REGION
    job_name = settings.TICKET_REINDEX_JOB_NAME
    
    # Get default credentials
    credentials, _ = default()
    
    # Refresh credentials if needed
    if not credentials.valid:
        credentials.refresh(Request())
    
    # Build API URL
    api_url = f"https://run.googleapis.com/v2/projects/{project_id}/locations/{region}/jobs/{job_name}:run"
    
    # Prepare request
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
    }
    
    # Minimal overrides (can be extended if needed)
    payload: Dict[str, Any] = {}
    
    logger.info(
        "[TICKET_REINDEX] Triggering Cloud Run Job",
        project=project_id,
        region=region,
        job=job_name,
        api_url=api_url
    )
    
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        execution_name = result.get("name", "")
        
        logger.info(
            "[TICKET_REINDEX] Job triggered successfully",
            execution_name=execution_name,
            job=job_name
        )
        
        return execution_name
        
    except requests.exceptions.RequestException as e:
        logger.error(
            "[TICKET_REINDEX] Failed to trigger job",
            job=job_name,
            error=str(e),
            status_code=getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None,
            exc_info=True
        )
        raise RuntimeError(f"Failed to trigger Cloud Run Job {job_name}: {e}") from e
