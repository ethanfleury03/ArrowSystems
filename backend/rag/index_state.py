"""
Centralized index state tracker for RAG index loading.

Tracks the state of index download and loading with structured progress information.
"""

import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from threading import Lock

# Thread-safe state tracker
_state_lock = Lock()

_index_state: Dict[str, Any] = {
    "ready": False,
    "phase": "idle",  # "idle" | "downloading" | "downloaded" | "loading" | "ready" | "error"
    "error": None,
    "started_at": None,
    "updated_at": None,
    "files": {},  # {filename: {size_bytes, downloaded, elapsed_s, status, error}}
    "total_bytes": 0,
    "bytes_downloaded": 0,
    "files_done": 0,
    "files_total": 0,
    "bucket": None,
    "prefix": None,
    "local_dir": None,
}


def get_index_state() -> Dict[str, Any]:
    """Get a snapshot of the current index state (thread-safe)."""
    with _state_lock:
        # Check for stuck states and auto-transition to error
        _check_stuck_state_internal()
        
        # Return a deep copy to avoid external mutations
        return {
            "ready": _index_state["ready"],
            "phase": _index_state["phase"],
            "error": _index_state["error"],
            "started_at": _index_state["started_at"],
            "updated_at": _index_state["updated_at"],
            "files": dict(_index_state["files"]),  # Shallow copy of files dict
            "total_bytes": _index_state["total_bytes"],
            "bytes_downloaded": _index_state["bytes_downloaded"],
            "files_done": _index_state["files_done"],
            "files_total": _index_state["files_total"],
            "bucket": _index_state["bucket"],
            "prefix": _index_state["prefix"],
            "local_dir": _index_state["local_dir"],
        }


def _check_stuck_state_internal() -> None:
    """
    Check if state has been stuck in downloading/loading for too long.
    Auto-transition to 'error' if stuck beyond timeout threshold.
    """
    phase = _index_state["phase"]
    if phase not in ("downloading", "loading"):
        return
    
    started_at = _index_state.get("started_at")
    if started_at is None:
        return
    
    # Get timeout from environment (default: 15 minutes)
    import os
    max_time = int(os.getenv("RAG_MAX_LOAD_TIME_SEC", "900"))  # 15 minutes default
    
    elapsed = time.time() - started_at
    if elapsed > max_time:
        error_msg = (
            f"Index {phase} timed out after {elapsed:.0f} seconds (max: {max_time}s). "
            f"This usually indicates a stuck download or load operation. "
            f"Check GCS permissions, network connectivity, and file sizes."
        )
        from backend.logging_config import get_logger
        logger = get_logger(__name__)
        logger.error(
            f"[RAG] State stuck timeout - auto-failing",
            phase=phase,
            elapsed_seconds=elapsed,
            max_time_seconds=max_time,
            message=error_msg,
        )
        _index_state["phase"] = "error"
        _index_state["error"] = error_msg
        _index_state["ready"] = False
        _index_state["updated_at"] = time.time()


def set_phase(phase: str, error: Optional[str] = None, bucket: Optional[str] = None, prefix: Optional[str] = None, local_dir: Optional[str] = None) -> None:
    """Set the current phase and optionally update error/bucket info."""
    with _state_lock:
        _index_state["phase"] = phase
        _index_state["updated_at"] = time.time()
        if error is not None:
            _index_state["error"] = error
        if bucket is not None:
            _index_state["bucket"] = bucket
        if prefix is not None:
            _index_state["prefix"] = prefix
        if local_dir is not None:
            _index_state["local_dir"] = local_dir
        if phase == "idle":
            _index_state["started_at"] = None
        elif _index_state["started_at"] is None:
            _index_state["started_at"] = time.time()
        if phase == "ready":
            _index_state["ready"] = True
            _index_state["error"] = None
        elif phase == "error":
            _index_state["ready"] = False
        else:
            _index_state["ready"] = False


def reset_state() -> None:
    """Reset state to idle (useful for retries)."""
    with _state_lock:
        _index_state["ready"] = False
        _index_state["phase"] = "idle"
        _index_state["error"] = None
        _index_state["started_at"] = None
        _index_state["updated_at"] = time.time()
        _index_state["files"] = {}
        _index_state["total_bytes"] = 0
        _index_state["bytes_downloaded"] = 0
        _index_state["files_done"] = 0
        _index_state["files_total"] = 0


def init_file_tracking(filenames: list[str], total_bytes: int = 0) -> None:
    """Initialize file tracking for a set of files."""
    with _state_lock:
        _index_state["files"] = {}
        _index_state["files_total"] = len(filenames)
        _index_state["files_done"] = 0
        _index_state["total_bytes"] = total_bytes
        _index_state["bytes_downloaded"] = 0
        for filename in filenames:
            _index_state["files"][filename] = {
                "size_bytes": 0,
                "downloaded": False,
                "elapsed_s": None,
                "status": "pending",
                "error": None,
                "attempt": 0,
            }


def update_file_start(filename: str, size_bytes: int = 0) -> None:
    """Mark a file download as started."""
    with _state_lock:
        if filename not in _index_state["files"]:
            _index_state["files"][filename] = {
                "size_bytes": 0,
                "downloaded": False,
                "elapsed_s": None,
                "status": "pending",
                "error": None,
                "attempt": 0,
            }
        file_info = _index_state["files"][filename]
        file_info["status"] = "downloading"
        file_info["size_bytes"] = size_bytes
        file_info["attempt"] = file_info.get("attempt", 0) + 1
        if size_bytes > 0:
            _index_state["total_bytes"] = max(_index_state["total_bytes"], _index_state["bytes_downloaded"] + size_bytes)


def update_file_success(filename: str, size_bytes: int, elapsed_s: float) -> None:
    """Mark a file download as successful."""
    with _state_lock:
        if filename not in _index_state["files"]:
            _index_state["files"][filename] = {
                "size_bytes": 0,
                "downloaded": False,
                "elapsed_s": None,
                "status": "pending",
                "error": None,
                "attempt": 0,
            }
        file_info = _index_state["files"][filename]
        file_info["downloaded"] = True
        file_info["status"] = "success"
        file_info["elapsed_s"] = elapsed_s
        file_info["size_bytes"] = size_bytes
        file_info["error"] = None
        _index_state["files_done"] += 1
        _index_state["bytes_downloaded"] += size_bytes


def update_file_error(filename: str, error: str, elapsed_s: Optional[float] = None) -> None:
    """Mark a file download as failed."""
    with _state_lock:
        if filename not in _index_state["files"]:
            _index_state["files"][filename] = {
                "size_bytes": 0,
                "downloaded": False,
                "elapsed_s": None,
                "status": "pending",
                "error": None,
                "attempt": 0,
            }
        file_info = _index_state["files"][filename]
        file_info["status"] = "error"
        file_info["error"] = error
        if elapsed_s is not None:
            file_info["elapsed_s"] = elapsed_s

