"""
Pipeline runner for background ticket scraping.

This module orchestrates the Scraper pipeline stages:
1. Index requests (Stage 1)
2. Build detailed conversations for new solved tickets (Stage 2)
3. Judge cache eligibility for new tickets (Stage 3)

All stages update the scrape_runs table with progress.
"""

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add Scraper to path
project_root = Path(__file__).parent.parent.parent
scraper_path = project_root / "Scraper"
if str(scraper_path) not in sys.path:
    sys.path.insert(0, str(scraper_path))

import sqlite3

# Delta check is done by comparing DB state before/after Stage 1
from ..logging_config import get_logger

logger = get_logger(__name__)


def get_ticket_store():
    """
    Get ticket store instance (SQLite or Postgres based on TICKETS_STORAGE_BACKEND).
    
    Returns:
        TicketStore instance
    """
    from ticket_store import get_ticket_store as _get_ticket_store
    
    # Get db_path if using SQLite
    backend = os.getenv("TICKETS_STORAGE_BACKEND", "sqlite").lower()
    if backend == "sqlite":
        from ..utils.tickets_admin import get_tickets_db_path
        db_path = get_tickets_db_path()
        if not db_path:
            raise FileNotFoundError(
                "Tickets database not found. Ensure Scraper/data/tickets.db exists "
                "or set TICKETS_DB_PATH environment variable."
            )
        return _get_ticket_store(backend="sqlite", db_path=str(db_path))
    else:
        return _get_ticket_store(backend="postgres")


def check_cancellation(store, run_id: str) -> bool:
    """
    Check if a scrape run has been cancelled.
    
    Args:
        store: TicketStore instance
        run_id: Run ID to check
        
    Returns:
        True if cancelled, False otherwise
    """
    run = store.get_latest_scrape_run()
    if run and run["run_id"] == run_id:
        return run["status"] == "cancelled"
    return False


def update_run_status(
    store,
    run_id: str,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    error: Optional[str] = None,
    summary_json: Optional[dict] = None
) -> None:
    """
    Update scrape run status.
    
    Args:
        conn: SQLite connection
        run_id: Run ID
        status: Optional status
        stage: Optional stage
        error: Optional error message
        summary_json: Optional summary dict (will be JSON-encoded)
    """
    import db as scraper_db
    
    summary_str = None
    if summary_json is not None:
        summary_str = json.dumps(summary_json, ensure_ascii=False)
    
    completed_at = None
    if status in ("completed", "failed", "cancelled"):
        completed_at = datetime.now(timezone.utc).isoformat()
    
    store.update_scrape_run(
        run_id,
        status=status,
        stage=stage,
        error=error,
        summary_json=summary_str,
        completed_at=completed_at
    )


def run_scrape_pipeline(run_id: str, created_by: Optional[str] = None) -> None:
    """
    Run the full scrape pipeline in the background.
    
    This function:
    1. Updates scrape_runs status to 'running', stage='indexing'
    2. Runs Stage 1: Index all requests
    3. Determines new solved ticket IDs (delta check)
    4. Updates stage='building_details'
    5. Runs Stage 2: Build conversations for new solved tickets
    6. Updates stage='judging'
    7. Runs Stage 3: Judge cache eligibility for new tickets
    8. Updates status='completed' with summary
    
    On error, updates status='failed' with error message.
    
    Args:
        run_id: Unique run ID (UUID)
        created_by: Optional admin email/user ID
    """
    try:
        logger.info(f"[{run_id}] Starting scrape pipeline")
        
        # Get ticket store
        store = get_ticket_store()
        
        # Ensure scrape_runs table exists
        store.ensure_scrape_runs_table()
        
        # Update status to running
        update_run_status(store, run_id, status="running", stage="indexing")
        logger.info(f"[{run_id}] Stage: indexing")
        
        # Stage 1: Index requests
        try:
            # Check for cancellation before starting
            if check_cancellation(store, run_id):
                logger.info(f"[{run_id}] Scrape cancelled before Stage 1")
                update_run_status(store, run_id, status="cancelled", completed_at=datetime.now(timezone.utc).isoformat())
                return
            
            import index_requests
            
            # Get db_path if using SQLite
            backend = os.getenv("TICKETS_STORAGE_BACKEND", "sqlite").lower()
            db_path = None
            if backend == "sqlite":
                from ..utils.tickets_admin import get_tickets_db_path
                db_path_obj = get_tickets_db_path()
                if db_path_obj:
                    db_path = str(db_path_obj)
            
            index_summary = index_requests.run_index_requests(db_path=db_path, headless=True)
            
            logger.info(f"[{run_id}] Stage 1 complete: indexed {index_summary['indexed']} requests")
            
            # Check for cancellation after Stage 1
            if check_cancellation(store, run_id):
                logger.info(f"[{run_id}] Scrape cancelled after Stage 1")
                update_run_status(store, run_id, status="cancelled", completed_at=datetime.now(timezone.utc).isoformat())
                return
        except Exception as e:
            # Check if it was cancelled during execution
            if check_cancellation(store, run_id):
                logger.info(f"[{run_id}] Scrape cancelled during Stage 1")
                update_run_status(store, run_id, status="cancelled", completed_at=datetime.now(timezone.utc).isoformat())
                return
            
            error_msg = f"Stage 1 (indexing) failed: {str(e)}"
            logger.error(f"[{run_id}] {error_msg}", exc_info=True)
            update_run_status(store, run_id, status="failed", error=error_msg)
            return
        
        # Determine new solved ticket IDs for Stage 2 (tickets without detail)
        try:
            logger.info(f"[{run_id}] Checking for new solved tickets without detail...")
            
            tickets_for_stage2 = store.get_ticket_ids_without_detail()
            
            logger.info(f"[{run_id}] Found {len(tickets_for_stage2)} new solved tickets without detail")
        except Exception as e:
            error_msg = f"Delta check failed: {str(e)}"
            logger.error(f"[{run_id}] {error_msg}", exc_info=True)
            update_run_status(store, run_id, status="failed", error=error_msg)
            return
        
        # If no new tickets, complete early
        if not tickets_for_stage2:
            logger.info(f"[{run_id}] No new solved tickets to process")
            summary = {
                "tickets_indexed": index_summary.get("total_count", 0),
                "tickets_new": 0,
                "tickets_solved": 0,
                "tickets_detail_built": 0,
                "tickets_judged": 0
            }
            update_run_status(store, run_id, status="completed", stage=None, summary_json=summary)
            return
        
        # Stage 2: Build detailed conversations (only for tickets without detail)
        if tickets_for_stage2:
            # Check for cancellation before Stage 2
            if check_cancellation(store, run_id):
                logger.info(f"[{run_id}] Scrape cancelled before Stage 2")
                update_run_status(store, run_id, status="cancelled", completed_at=datetime.now(timezone.utc).isoformat())
                return
            
            update_run_status(store, run_id, stage="building_details")
            logger.info(f"[{run_id}] Stage: building_details ({len(tickets_for_stage2)} tickets)")
            
            try:
                import build_solved_tickets
                
                # Get db_path if using SQLite
                backend = os.getenv("TICKETS_STORAGE_BACKEND", "sqlite").lower()
                db_path = None
                if backend == "sqlite":
                    from ..utils.tickets_admin import get_tickets_db_path
                    db_path_obj = get_tickets_db_path()
                    if db_path_obj:
                        db_path = str(db_path_obj)
                
                build_summary = build_solved_tickets.run_build_solved_tickets(
                    ticket_ids=tickets_for_stage2,
                    db_path=db_path,
                    headless=True
                )
                
                logger.info(f"[{run_id}] Stage 2 complete: built {build_summary['built']} conversations")
                
                # Check for cancellation after Stage 2
                if check_cancellation(store, run_id):
                    logger.info(f"[{run_id}] Scrape cancelled after Stage 2")
                    update_run_status(store, run_id, status="cancelled", completed_at=datetime.now(timezone.utc).isoformat())
                    return
            except Exception as e:
                # Check if it was cancelled during execution
                if check_cancellation(store, run_id):
                    logger.info(f"[{run_id}] Scrape cancelled during Stage 2")
                    update_run_status(store, run_id, status="cancelled", completed_at=datetime.now(timezone.utc).isoformat())
                    return
                
                error_msg = f"Stage 2 (building_details) failed: {str(e)}"
                logger.error(f"[{run_id}] {error_msg}", exc_info=True)
                update_run_status(store, run_id, status="failed", error=error_msg)
                return
        else:
            build_summary = {"built": 0}
            logger.info(f"[{run_id}] Stage 2 skipped: no tickets need detail building")
        
        # Determine tickets for Stage 3 (tickets without judgements)
        try:
            tickets_for_stage3 = store.get_ticket_ids_needing_judgement(only_solved=True)
            logger.info(f"[{run_id}] Found {len(tickets_for_stage3)} tickets needing judgement")
        except Exception as e:
            logger.warning(f"[{run_id}] Error getting tickets for judgement: {e}")
            tickets_for_stage3 = []
        
        # Stage 3: Judge cache eligibility (only for tickets without judgements)
        if tickets_for_stage3:
            # Check for cancellation before Stage 3
            if check_cancellation(store, run_id):
                logger.info(f"[{run_id}] Scrape cancelled before Stage 3")
                update_run_status(store, run_id, status="cancelled", completed_at=datetime.now(timezone.utc).isoformat())
                return
            
            update_run_status(store, run_id, stage="judging")
            logger.info(f"[{run_id}] Stage: judging ({len(tickets_for_stage3)} tickets)")
            
            try:
                import judge_ticket_cache_eligibility
                
                # Get db_path if using SQLite
                backend = os.getenv("TICKETS_STORAGE_BACKEND", "sqlite").lower()
                db_path = None
                if backend == "sqlite":
                    from ..utils.tickets_admin import get_tickets_db_path
                    db_path_obj = get_tickets_db_path()
                    if db_path_obj:
                        db_path = str(db_path_obj)
                
                judge_summary = judge_ticket_cache_eligibility.run_judge_cache_eligibility(
                    ticket_ids=tickets_for_stage3,
                    db_path=db_path,
                    force=False,
                    require_requester_confirmation=True
                )
                
                logger.info(f"[{run_id}] Stage 3 complete: judged {judge_summary['processed']} tickets")
                
                # Check for cancellation after Stage 3
                if check_cancellation(store, run_id):
                    logger.info(f"[{run_id}] Scrape cancelled after Stage 3")
                    update_run_status(store, run_id, status="cancelled", completed_at=datetime.now(timezone.utc).isoformat())
                    return
            except Exception as e:
                # Check if it was cancelled during execution
                if check_cancellation(store, run_id):
                    logger.info(f"[{run_id}] Scrape cancelled during Stage 3")
                    update_run_status(store, run_id, status="cancelled", completed_at=datetime.now(timezone.utc).isoformat())
                    return
                
                error_msg = f"Stage 3 (judging) failed: {str(e)}"
                logger.error(f"[{run_id}] {error_msg}", exc_info=True)
                update_run_status(store, run_id, status="failed", error=error_msg)
                return
        else:
            judge_summary = {"processed": 0, "cache_eligible_count": 0}
            logger.info(f"[{run_id}] Stage 3 skipped: no tickets need judgement")
        
        # Complete successfully
        summary = {
            "tickets_indexed": index_summary.get("total_count", 0),
            "tickets_new": len(tickets_for_stage2),
            "tickets_solved": len(tickets_for_stage2),
            "tickets_detail_built": build_summary.get("built", 0),
            "tickets_judged": judge_summary.get("processed", 0),
            "cache_eligible_count": judge_summary.get("cache_eligible_count", 0)
        }
        
        update_run_status(store, run_id, status="completed", stage=None, summary_json=summary)
        logger.info(f"[{run_id}] Pipeline completed successfully")
        
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"[{run_id}] {error_msg}", exc_info=True)
        
        try:
            store = get_ticket_store()
            update_run_status(store, run_id, status="failed", error=str(e))
        except Exception:
            pass  # Best effort
