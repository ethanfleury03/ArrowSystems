"""
Lightweight resource monitoring for deployment sizing.
Logs RSS memory and timing at key checkpoints.

This module provides safe, cross-platform memory monitoring that:
- Works with or without psutil (optional dependency)
- Never raises exceptions (gracefully degrades)
- Compatible with Windows, Linux, and macOS
"""

import os
import time
import logging
from typing import Optional

# Global start time for elapsed time tracking
_START_TIME = time.time()

# Try to import psutil (optional dependency)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


def get_rss_mb() -> Optional[float]:
    """
    Get current RSS (Resident Set Size) memory usage in MB.
    
    Tries multiple methods in order:
    1. psutil (if available) - most accurate, cross-platform
    2. resource.getrusage (Unix/Linux/macOS only)
    3. Returns None if neither is available
    
    Returns:
        RSS memory in MB, or None if unavailable
    """
    # Method 1: Try psutil (best option, cross-platform)
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        except Exception:
            pass  # Fall through to next method
    
    # Method 2: Try resource module (Unix/Linux/macOS only)
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in KB on Linux, pages on macOS
        # Assume Linux (KB) - if on macOS, this will be slightly off but still useful
        rss_kb = usage.ru_maxrss
        # Convert KB to MB
        return rss_kb / 1024.0
    except (ImportError, AttributeError, OSError):
        # resource module not available (Windows) or getrusage failed
        pass
    
    # Method 3: No method available
    return None


def elapsed_s() -> float:
    """
    Get elapsed time in seconds since module was imported.
    
    Returns:
        Elapsed time in seconds (float)
    """
    return time.time() - _START_TIME


def get_memory_mb() -> Optional[float]:
    """
    Backward-compatible alias for get_rss_mb().
    
    Returns:
        RSS memory in MB, or None if unavailable
    """
    try:
        return get_rss_mb()
    except Exception:
        # Never raise - gracefully degrade
        return None


def get_elapsed_seconds() -> float:
    """
    Backward-compatible alias for elapsed_s().
    
    Returns:
        Elapsed time in seconds since module was imported
    """
    try:
        return elapsed_s()
    except Exception:
        # Never raise - return 0 as safe default
        return 0.0


def log_resource_checkpoint(name: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log memory and timing at a checkpoint.
    
    Format: [RESOURCE] {name} elapsed={elapsed:.2f}s rss_mb={rss:.1f}MB
    If RSS is unavailable, shows: [RESOURCE] {name} elapsed={elapsed:.2f}s rss_mb=unknown
    
    Args:
        name: Checkpoint name (e.g., "process_start", "model_init", "index_load")
        logger: Optional logger instance. If None, prints to stdout.
    
    Never raises exceptions - gracefully handles all errors.
    """
    try:
        memory_mb = get_rss_mb()
        elapsed = elapsed_s()
        
        # Format message string (standard Python logging, not structured)
        if memory_mb is not None:
            msg = f"[RESOURCE] {name} elapsed={elapsed:.2f}s rss_mb={memory_mb:.1f}MB"
        else:
            msg = f"[RESOURCE] {name} elapsed={elapsed:.2f}s rss_mb=unknown"
        
        if logger:
            logger.info(msg)
        else:
            print(msg, flush=True)
    except Exception:
        # Never crash - if logging fails, silently continue
        # This ensures instrumentation never breaks the app
        pass

