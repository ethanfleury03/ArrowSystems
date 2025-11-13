from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
import json
import re

import jwt
from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sqlalchemy import select, func, desc, and_, or_, case
from sqlalchemy.orm import Session

from ..security import decode_access_token
from ..utils.database_manager import DatabaseManager
from ..utils.db import SessionLocal, QueryHistory, User


class AdminUserResponse(BaseModel):
    id: str
    email: Optional[str] = None
    name: Optional[str] = None
    role: str
    company_name: Optional[str] = None
    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None
    machine_models: Optional[List[str]] = None  # List of machine model strings
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class AdminUserCreateRequest(BaseModel):
    email: str
    password: str
    role: str = "TECHNICIAN"
    name: Optional[str] = None
    company_name: Optional[str] = None
    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None
    machine_models: Optional[List[str]] = None  # List of machine model strings


class AdminUserUpdateRequest(BaseModel):
    email: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None
    name: Optional[str] = None
    company_name: Optional[str] = None
    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None
    machine_models: Optional[List[str]] = None  # List of machine model strings


def create_admin_router(db_manager_getter: Callable[[], Optional[DatabaseManager]]) -> APIRouter:
    router = APIRouter(prefix="/admin", tags=["admin"])
    security = HTTPBearer()

    async def get_db_manager() -> DatabaseManager:
        manager = db_manager_getter()
        if manager is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database not initialized")
        return manager

    async def get_current_admin(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        manager: DatabaseManager = Depends(get_db_manager),
    ) -> Dict[str, str]:
        if not credentials or not credentials.credentials:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authorization token")

        token = credentials.credentials
        try:
            payload = decode_access_token(token)
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired") from None
        except jwt.PyJWTError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from None

        email = payload.get("email")
        role = payload.get("role")
        if not email or not role:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token payload")

        user = await manager.get_user_by_email(email)
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User no longer exists")
        if user.get("role") != "ADMIN":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
        return user

    @router.get("/users", response_model=List[AdminUserResponse])
    async def list_users(
        _: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
    ):
        users = await manager.list_users()
        
        # Include allowed_machine_models in each user response for frontend
        try:
            from ..config.machine_models import get_allowed_machine_models
            allowed_models = get_allowed_machine_models()
        except ImportError:
            allowed_models = []
        
        # Add allowed_machine_models to each user response
        users_with_allowed = []
        for user in users:
            user_dict = dict(user) if isinstance(user, dict) else user
            user_dict["allowed_machine_models"] = allowed_models
            users_with_allowed.append(user_dict)
        
        return users_with_allowed

    @router.post("/create_user", response_model=AdminUserResponse, status_code=status.HTTP_201_CREATED)
    async def create_user(
        payload: AdminUserCreateRequest = Body(...),
        _: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
    ):
        """
        Create a new user (admin-only).
        
        Validation rules:
        - If role == "CUSTOMER" → machine_models is REQUIRED and must be a non-empty subset of ALLOWED_MACHINE_MODELS
        - If role in ["ADMIN", "TECHNICIAN"] → machine_models can be omitted or ignored
        """
        from ..config.machine_models import (
            normalize_machine_models,
            is_valid_machine_model_list,
            get_allowed_machine_models,
            get_machine_models_for_selection
        )
        
        role_upper = (payload.role or "TECHNICIAN").upper()
        
        # Normalize machine_models
        machine_models = normalize_machine_models(payload.machine_models)
        
        # Validation: Customers must have at least one machine assigned
        if role_upper == "CUSTOMER":
            if not machine_models or len(machine_models) == 0:
                allowed_models = get_machine_models_for_selection()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Customers must have at least one machine assigned. Available machines: {', '.join(allowed_models) if allowed_models else 'None'}"
                )
            
            # Validate all machines are in allowed list
            from ..config.machine_models import is_valid_machine_model
            invalid_models = [m for m in machine_models if not is_valid_machine_model(m)]
            if invalid_models:
                allowed_models = get_allowed_machine_models()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid machine_models: {', '.join(invalid_models)}. Must be a subset of: {', '.join(allowed_models) if allowed_models else 'None'}"
                )
        else:
            # For admin/technician, machine_models are optional (will be ignored in retrieval anyway)
            # But still validate if provided
            if machine_models and len(machine_models) > 0:
                from ..config.machine_models import is_valid_machine_model
                invalid_models = [m for m in machine_models if not is_valid_machine_model(m)]
                if invalid_models:
                    allowed_models = get_allowed_machine_models()
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid machine_models: {', '.join(invalid_models)}. Must be a subset of: {', '.join(allowed_models) if allowed_models else 'None'}"
                    )
        
        existing = await manager.get_user_by_email(payload.email)
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

        created = await manager.create_user(
            email=payload.email,
            password=payload.password,
            role=payload.role,
            name=payload.name,
            company_name=payload.company_name,
            contact_name=payload.contact_name,
            contact_phone=payload.contact_phone,
            machine_models=machine_models if role_upper == "CUSTOMER" else None,  # Only set for customers
        )
        return created

    @router.put("/edit_user/{user_id}", response_model=AdminUserResponse)
    async def edit_user(
        user_id: int,
        payload: AdminUserUpdateRequest = Body(...),
        _: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
    ):
        """
        Update user (admin-only).
        
        Validation rules:
        - If role is changed TO "CUSTOMER" → machine_models must be non-empty and valid
        - If role is changed FROM "CUSTOMER" to admin/technician → machine_models can be cleared
        - If role remains "CUSTOMER" and machine_models is updated → must be non-empty and valid
        """
        from ..config.machine_models import (
            normalize_machine_models,
            is_valid_machine_model_list,
            get_allowed_machine_models,
            get_machine_models_for_selection
        )
        
        # Get current user to check role changes
        current_user = await manager.get_user_by_id(user_id)
        if not current_user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        current_role = current_user.get("role", "TECHNICIAN").upper()
        new_role = (payload.role or current_role).upper()
        role_changed = new_role != current_role
        
        # Normalize machine_models
        machine_models = normalize_machine_models(payload.machine_models) if payload.machine_models is not None else None
        
        # Validation based on role changes
        if role_changed:
            # Role is being changed
            if new_role == "CUSTOMER":
                # Changed TO customer - require machine_models
                if not machine_models or len(machine_models) == 0:
                    # Try to keep existing machine_models if available
                    existing_machine_models = current_user.get("machine_models", [])
                    if not existing_machine_models or len(existing_machine_models) == 0:
                        allowed_models = get_machine_models_for_selection()
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Cannot change role to CUSTOMER without machine_models. Customers must have at least one machine assigned. Available machines: {', '.join(allowed_models) if allowed_models else 'None'}"
                        )
                    machine_models = existing_machine_models
            # If changed FROM customer to admin/technician, machine_models can be cleared
            # (retrieval will ignore them anyway via get_effective_machines_for_user)
        else:
            # Role not changed - validate based on current role
            if new_role == "CUSTOMER":
                if machine_models is not None:
                    # machine_models is being updated for a customer
                    if len(machine_models) == 0:
                        allowed_models = get_machine_models_for_selection()
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Cannot clear machine_models for CUSTOMER role. Customers must have at least one machine assigned. Available machines: {', '.join(allowed_models) if allowed_models else 'None'}"
                        )
        
        # Validate machine_models if provided
        if machine_models is not None and len(machine_models) > 0:
            if not is_valid_machine_model_list(machine_models):
                from ..config.machine_models import is_valid_machine_model
                invalid_models = [m for m in machine_models if not is_valid_machine_model(m)]
                allowed_models = get_allowed_machine_models()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid machine_models: {', '.join(invalid_models)}. Must be a subset of: {', '.join(allowed_models) if allowed_models else 'None'}"
                )
        
        try:
            # If role is admin/technician and machine_models is None, clear it
            # If role is customer and machine_models is None, keep existing (don't update)
            update_machine_models = machine_models
            if new_role != "CUSTOMER" and machine_models is None:
                # Admin/technician - can clear machine_models
                update_machine_models = []
            elif new_role == "CUSTOMER" and machine_models is None:
                # Customer - keep existing machine_models (don't update)
                update_machine_models = None
            
            updated = await manager.update_user(
                user_id,
                email=payload.email,
                name=payload.name,
                password=payload.password,
                role=payload.role,
                company_name=payload.company_name,
                contact_name=payload.contact_name,
                contact_phone=payload.contact_phone,
                machine_models=update_machine_models,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        return updated

    @router.delete("/delete_user/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_user(
        user_id: int,
        _: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
    ):
        deleted = await manager.delete_user(user_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        return None

    @router.post("/logs/test")
    async def test_logging(
        _: Dict[str, str] = Depends(get_current_admin),
    ):
        """
        Test endpoint to generate a test log entry.
        This helps verify that logging is working correctly.
        """
        import logging
        test_logger = logging.getLogger("test_logger")
        
        test_logger.info("TEST LOG: Admin logs endpoint test - INFO level")
        test_logger.warning("TEST LOG: Admin logs endpoint test - WARNING level")
        test_logger.error("TEST LOG: Admin logs endpoint test - ERROR level")
        
        return {
            "status": "success",
            "message": "Test log entries have been written. Check the logs endpoint to see them.",
            "timestamp": datetime.now().isoformat()
        }
    
    @router.get("/logs")
    async def get_logs(
        _: Dict[str, str] = Depends(get_current_admin),
        limit: int = 1000,
        level: Optional[str] = None,
        search: Optional[str] = None,
        tail: bool = True,
        max_lines_per_file: int = 10000,  # Max lines to read per file (for large files)
    ):
        """
        Get system logs from multiple log files (backend and frontend).
        
        Args:
            limit: Maximum number of log lines to return (default: 1000)
            level: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            search: Search term to filter logs
            tail: If True, return the last N lines. If False, return from the beginning.
        """
        import os
        import re
        from datetime import datetime
        
        def parse_log_line(line: str, source: str) -> Dict[str, Any]:
            """Parse a single log line into structured format."""
            line = line.strip()
            if not line:
                return None
            
            log_entry = {
                "raw": line,
                "timestamp": None,
                "level": None,
                "logger": None,
                "message": line,
                "source": source,
            }
            
            # Try to extract timestamp (various formats)
            timestamp_patterns = [
                r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d{3})?)',  # Python logging format
                r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z?)',  # ISO format
                r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]',  # Bracket format
            ]
            
            for pattern in timestamp_patterns:
                timestamp_match = re.match(pattern, line)
                if timestamp_match:
                    log_entry["timestamp"] = timestamp_match.group(1)
                    break
            
            # Extract log level (various formats)
            level_patterns = [
                r'\s-\s(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s-',  # Python format
                r'\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b',  # Generic
                r'\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]',  # Bracket format
            ]
            
            for pattern in level_patterns:
                level_match = re.search(pattern, line, re.IGNORECASE)
                if level_match:
                    log_entry["level"] = level_match.group(1).upper()
                    break
            
            # Extract logger name (between timestamp and level)
            if log_entry["level"]:
                parts = re.split(rf'\s*-\s*{re.escape(log_entry["level"])}\s*-\s*', line, 1, re.IGNORECASE)
                if len(parts) > 1:
                    log_entry["message"] = parts[1]
                    # Extract logger name from first part
                    if log_entry["timestamp"]:
                        logger_part = parts[0].replace(log_entry["timestamp"], "").strip(" -[]")
                        if logger_part:
                            log_entry["logger"] = logger_part
            
            return log_entry
        
        def read_log_file(file_path: str, source: str, max_lines: int = 10000) -> List[Dict[str, Any]]:
            """
            Read and parse a log file efficiently.
            For large files, only reads the tail (last N lines) to avoid memory issues.
            """
            entries = []
            try:
                if not os.path.exists(file_path):
                    return entries
                
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    # File exists but is empty - add info entry
                    entries.append({
                        "raw": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - log_reader - INFO - Log file {file_path} exists but is empty (0 bytes)",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "level": "INFO",
                        "logger": "log_reader",
                        "message": f"Log file {file_path} exists but is empty (0 bytes). Waiting for logs to be written.",
                        "source": "system",
                    })
                    return entries
                
                # For large files (>5MB), use tail reading to avoid loading entire file into memory
                # This is important for 24/7 operation where log files can grow very large
                large_file_threshold = 5 * 1024 * 1024  # 5 MB
                
                if file_size > large_file_threshold:
                    # Read from the end of the file (tail)
                    # Estimate: average log line is ~200 bytes, so max_lines * 200 bytes
                    bytes_to_read = min(max_lines * 200, file_size)
                    
                    with open(file_path, 'rb') as f:
                        # Seek to position near the end
                        f.seek(max(0, file_size - bytes_to_read))
                        # Read and decode
                        chunk = f.read()
                        try:
                            text = chunk.decode('utf-8', errors='ignore')
                        except:
                            # If UTF-8 fails, try to find the start of a line
                            # Skip partial line at the beginning
                            text = chunk.decode('utf-8', errors='ignore')
                            if '\n' in text:
                                text = text.split('\n', 1)[1]  # Skip first partial line
                        
                        lines = text.splitlines()
                        # Only take the last max_lines
                        lines = lines[-max_lines:] if len(lines) > max_lines else lines
                else:
                    # For smaller files, read normally but still limit lines
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        all_lines = f.readlines()
                        # Only take the last max_lines for consistency
                        lines = all_lines[-max_lines:] if len(all_lines) > max_lines else all_lines
                
                if not lines:
                    return entries
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    entry = parse_log_line(line, source)
                    if entry:
                        entries.append(entry)
            except Exception as e:
                # Log error but don't fail completely
                entries.append({
                    "raw": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - log_reader - ERROR - Error reading {file_path}: {str(e)}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "level": "ERROR",
                    "logger": "log_reader",
                    "message": f"Error reading {file_path}: {str(e)}",
                    "source": "system",
                })
            return entries
        
        # Find all possible log files
        log_files = []
        
        # Backend log files
        backend_log_paths = [
            "api.log",
            "../api.log",
            "backend/api.log",
            "/app/api.log",
            "/workspace/api.log",
            os.path.join(os.getcwd(), "api.log"),
        ]
        
        # Also check for other log files
        other_log_paths = [
            "rag_handler.log",
            "logs/api.log",
            "logs/backend.log",
            "logs/frontend.log",
            "../logs/api.log",
            "/app/logs/api.log",
        ]
        
        # Check for frontend logs
        frontend_log_paths = [
            "frontend.log",
            "logs/frontend.log",
            "../frontend.log",
            "frontend/.next/trace",
            "frontend/logs/frontend.log",
        ]
        
        # Also check for rotated log files (api.log.1, api.log.2, etc. from RotatingFileHandler)
        rotated_log_paths = []
        for base_path in backend_log_paths[:3]:  # Check first few common paths
            base_dir = os.path.dirname(base_path) if os.path.dirname(base_path) else '.'
            base_name = os.path.basename(base_path)
            if os.path.exists(base_dir) or base_dir == '.':
                for i in range(1, 6):  # Check for .1 through .5 (backup files)
                    rotated_path = os.path.join(base_dir, f"{base_name}.{i}") if base_dir != '.' else f"{base_name}.{i}"
                    if os.path.exists(rotated_path):
                        rotated_log_paths.append(rotated_path)
        
        all_paths = backend_log_paths + other_log_paths + frontend_log_paths + rotated_log_paths
        
        # Track seen files by absolute path to avoid duplicates
        seen_files = {}
        
        for path in all_paths:
            if os.path.exists(path) and os.path.isfile(path):
                # Get absolute path to check for duplicates
                abs_path = os.path.abspath(path)
                
                # Skip if we've already seen this file
                if abs_path in seen_files:
                    continue
                
                # Determine source
                if "frontend" in path.lower():
                    source = "frontend"
                elif "backend" in path.lower() or "api.log" in path or "rag_handler" in path:
                    source = "backend"
                else:
                    source = "system"
                
                seen_files[abs_path] = (path, source)
                log_files.append((path, source))
        
        # Read all log files
        all_entries = []
        found_files = []
        seen_entries = set()  # Track seen entries to avoid duplicates
        
        for file_path, source in log_files:
            entries = read_log_file(file_path, source, max_lines=max_lines_per_file)
            
            # Deduplicate entries by creating a unique key
            unique_entries = []
            for entry in entries:
                # Create a unique key from timestamp, level, logger, and message
                entry_key = (
                    entry.get("timestamp", ""),
                    entry.get("level", ""),
                    entry.get("logger", ""),
                    entry.get("message", "")[:200]  # First 200 chars of message
                )
                
                if entry_key not in seen_entries:
                    seen_entries.add(entry_key)
                    unique_entries.append(entry)
            
            all_entries.extend(unique_entries)
            if unique_entries:
                abs_path = os.path.abspath(file_path)
                found_files.append({
                    "path": abs_path,  # Use absolute path for display
                    "source": source,
                    "entries": len(unique_entries),
                    "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                })
        
        # If no logs found, return helpful message with debug info
        if not all_entries:
            # Add a test log entry to verify the endpoint is working
            test_entry = {
                "raw": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - log_reader - INFO - Log endpoint accessed. No log entries found in checked files.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "level": "INFO",
                "logger": "log_reader",
                "message": f"Log endpoint accessed. No log entries found. Checked {len(log_files)} file(s).",
                "source": "system",
            }
            
            # Check current working directory
            cwd = os.getcwd()
            test_entry["message"] += f" Current working directory: {cwd}"
            
            return {
                "logs": [test_entry],  # Return test entry so user knows endpoint is working
                "total": 0,
                "files_checked": [p for p, _ in log_files] if log_files else all_paths[:10],
                "files_found": found_files,
                "current_directory": cwd,
                "message": f"No log entries found in {len(log_files)} checked file(s). Logs will appear here once the application starts generating them."
            }
        
        # Sort by timestamp if available, otherwise by source and raw line
        # Sort in reverse (newest first) for log viewing
        def get_sort_key(entry):
            if entry.get("timestamp"):
                try:
                    # Try to parse timestamp for sorting (reverse order - newest first)
                    ts_str = entry["timestamp"].replace(",", ".")
                    # For reverse sort, we'll use the timestamp as-is and sort in reverse
                    # Timestamp format: "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD HH:MM:SS,mmm"
                    # String comparison works for ISO format timestamps
                    return (0, ts_str, entry.get("source", ""), entry["raw"])
                except:
                    pass
            return (1, entry.get("source", ""), entry["raw"])
        
        # Sort by timestamp descending (newest first), then reverse the list
        all_entries.sort(key=get_sort_key, reverse=True)
        
        # Filter by level if specified
        if level:
            level_upper = level.upper()
            all_entries = [e for e in all_entries if e.get("level", "").upper() == level_upper]
        
        # Filter by search term if specified
        if search:
            search_lower = search.lower()
            all_entries = [
                e for e in all_entries
                if search_lower in e["raw"].lower() or search_lower in e.get("message", "").lower()
            ]
        
        # Get last N lines if tail is True
        if tail:
            all_entries = all_entries[-limit:] if len(all_entries) > limit else all_entries
        else:
            all_entries = all_entries[:limit] if len(all_entries) > limit else all_entries
        
        return {
            "logs": all_entries,
            "total": len(all_entries),
            "files_found": found_files,
            "total_files": len(found_files),
        }

    # ============================================================================
    # Analytics Endpoints
    # ============================================================================
    
    @router.get("/analytics/queries_over_time")
    async def queries_over_time(
        _: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        user_id: Optional[int] = Query(None),
        machine_name: Optional[str] = Query(None),
    ):
        """Get query counts over time (daily aggregation)."""
        def _fetch():
            with SessionLocal() as session:
                # Build base query with filters
                conditions = []
                
                if start_date:
                    try:
                        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                        conditions.append(QueryHistory.created_at >= start_dt)
                    except Exception:
                        pass
                
                if end_date:
                    try:
                        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                        conditions.append(QueryHistory.created_at <= end_dt)
                    except Exception:
                        pass
                
                if user_id:
                    conditions.append(QueryHistory.user_id == user_id)
                
                if machine_name:
                    conditions.append(QueryHistory.machine_name == machine_name)
                
                # Group by date (daily)
                date_trunc = func.date(QueryHistory.created_at)
                query = select(
                    date_trunc.label('date'),
                    func.count(QueryHistory.id).label('query_count')
                ).select_from(QueryHistory)
                
                if conditions:
                    query = query.where(and_(*conditions))
                
                query = query.group_by(date_trunc).order_by(date_trunc)
                
                results = session.execute(query).all()
                return [
                    {"date": str(row.date), "query_count": row.query_count}
                    for row in results
                ]
        
        from ..utils.db import run_sync
        buckets = await run_sync(_fetch)
        return {"buckets": buckets}
    
    @router.get("/analytics/queries_per_user")
    async def queries_per_user(
        _: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        machine_name: Optional[str] = Query(None),
    ):
        """Get query counts per user."""
        def _fetch():
            with SessionLocal() as session:
                query = select(
                    QueryHistory.user_id,
                    User.email,
                    func.count(QueryHistory.id).label('query_count')
                ).join(User, QueryHistory.user_id == User.id)
                
                # Apply filters
                if start_date:
                    try:
                        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                        query = query.where(QueryHistory.created_at >= start_dt)
                    except Exception:
                        pass
                
                if end_date:
                    try:
                        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                        query = query.where(QueryHistory.created_at <= end_dt)
                    except Exception:
                        pass
                
                if machine_name:
                    query = query.where(QueryHistory.machine_name == machine_name)
                
                query = query.group_by(QueryHistory.user_id, User.email).order_by(desc('query_count'))
                
                results = session.execute(query).all()
                return [
                    {"user_id": row.user_id, "email": row.email, "query_count": row.query_count}
                    for row in results
                ]
        
        from ..utils.db import run_sync
        items = await run_sync(_fetch)
        return {"items": items}
    
    @router.get("/analytics/queries_by_machine")
    async def queries_by_machine(
        _: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        user_id: Optional[int] = Query(None),
    ):
        """Get query counts by machine type."""
        def _fetch():
            with SessionLocal() as session:
                query = select(
                    case(
                        (QueryHistory.machine_name.is_(None), "Unknown"),
                        (QueryHistory.machine_name == "", "Unknown"),
                        else_=QueryHistory.machine_name
                    ).label('machine_name'),
                    func.count(QueryHistory.id).label('query_count')
                ).select_from(QueryHistory)
                
                # Apply filters
                if start_date:
                    try:
                        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                        query = query.where(QueryHistory.created_at >= start_dt)
                    except Exception:
                        pass
                
                if end_date:
                    try:
                        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                        query = query.where(QueryHistory.created_at <= end_dt)
                    except Exception:
                        pass
                
                if user_id:
                    query = query.where(QueryHistory.user_id == user_id)
                
                query = query.group_by('machine_name').order_by(desc('query_count'))
                
                results = session.execute(query).all()
                return [
                    {"machine_name": row.machine_name, "query_count": row.query_count}
                    for row in results
                ]
        
        from ..utils.db import run_sync
        items = await run_sync(_fetch)
        return {"items": items}
    
    @router.get("/analytics/token_usage")
    async def token_usage(
        _: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        user_id: Optional[int] = Query(None),
        machine_name: Optional[str] = Query(None),
    ):
        """Get token usage over time."""
        def _fetch():
            with SessionLocal() as session:
                # Build conditions
                conditions = []
                
                if start_date:
                    try:
                        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                        conditions.append(QueryHistory.created_at >= start_dt)
                    except Exception:
                        pass
                
                if end_date:
                    try:
                        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                        conditions.append(QueryHistory.created_at <= end_dt)
                    except Exception:
                        pass
                
                if user_id:
                    conditions.append(QueryHistory.user_id == user_id)
                
                if machine_name:
                    conditions.append(QueryHistory.machine_name == machine_name)
                
                date_trunc = func.date(QueryHistory.created_at)
                query = select(
                    date_trunc.label('date'),
                    func.sum(QueryHistory.token_input).label('token_input'),
                    func.sum(QueryHistory.token_output).label('token_output'),
                    func.sum(QueryHistory.token_total).label('token_total'),
                    func.sum(QueryHistory.cost_usd).label('cost_usd')
                ).select_from(QueryHistory)
                
                if conditions:
                    query = query.where(and_(*conditions))
                
                query = query.group_by(date_trunc).order_by(date_trunc)
                
                results = session.execute(query).all()
                buckets = [
                    {
                        "date": str(row.date),
                        "token_input": int(row.token_input or 0),
                        "token_output": int(row.token_output or 0),
                        "token_total": int(row.token_total or 0),
                        "cost_usd": float(row.cost_usd or 0.0)
                    }
                    for row in results
                ]
                
                # Calculate totals
                totals = {
                    "token_input": sum(b["token_input"] for b in buckets),
                    "token_output": sum(b["token_output"] for b in buckets),
                    "token_total": sum(b["token_total"] for b in buckets),
                    "cost_usd": sum(b["cost_usd"] for b in buckets)
                }
                
                return buckets, totals
        
        from ..utils.db import run_sync
        buckets, totals = await run_sync(_fetch)
        return {"buckets": buckets, "totals": totals}
    
    @router.get("/analytics/token_usage_per_user")
    async def token_usage_per_user(
        _: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        machine_name: Optional[str] = Query(None),
    ):
        """Get token usage per user."""
        def _fetch():
            with SessionLocal() as session:
                query = select(
                    QueryHistory.user_id,
                    User.email,
                    func.sum(QueryHistory.token_total).label('token_total'),
                    func.sum(QueryHistory.cost_usd).label('cost_usd')
                ).join(User, QueryHistory.user_id == User.id)
                
                # Apply filters
                if start_date:
                    try:
                        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                        query = query.where(QueryHistory.created_at >= start_dt)
                    except Exception:
                        pass
                
                if end_date:
                    try:
                        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                        query = query.where(QueryHistory.created_at <= end_dt)
                    except Exception:
                        pass
                
                if machine_name:
                    query = query.where(QueryHistory.machine_name == machine_name)
                
                query = query.group_by(QueryHistory.user_id, User.email).order_by(desc('token_total'))
                
                results = session.execute(query).all()
                return [
                    {
                        "user_id": row.user_id,
                        "email": row.email,
                        "token_total": int(row.token_total or 0),
                        "cost_usd": float(row.cost_usd or 0.0)
                    }
                    for row in results
                ]
        
        from ..utils.db import run_sync
        items = await run_sync(_fetch)
        return {"items": items}
    
    @router.get("/analytics/document_usage")
    async def document_usage(
        _: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        user_id: Optional[int] = Query(None),
        machine_name: Optional[str] = Query(None),
    ):
        """Get document usage statistics."""
        def _fetch():
            with SessionLocal() as session:
                query = select(QueryHistory.sources_json).select_from(QueryHistory).where(QueryHistory.sources_json.isnot(None))
                
                # Apply filters
                if start_date:
                    try:
                        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                        query = query.where(QueryHistory.created_at >= start_dt)
                    except Exception:
                        pass
                
                if end_date:
                    try:
                        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                        query = query.where(QueryHistory.created_at <= end_dt)
                    except Exception:
                        pass
                
                if user_id:
                    query = query.where(QueryHistory.user_id == user_id)
                
                if machine_name:
                    query = query.where(QueryHistory.machine_name == machine_name)
                
                results = session.execute(query).all()
                
                # Aggregate document usage
                doc_counts = {}
                for row in results:
                    try:
                        sources = json.loads(row.sources_json) if isinstance(row.sources_json, str) else row.sources_json
                        if isinstance(sources, list):
                            for source in sources:
                                if isinstance(source, dict):
                                    doc_id = source.get('name') or source.get('id') or str(source)
                                else:
                                    doc_id = str(source)
                                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
                    except Exception:
                        continue
                
                items = [
                    {"document_id": doc_id, "display_name": doc_id, "usage_count": count}
                    for doc_id, count in sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)
                ]
                return items
        
        from ..utils.db import run_sync
        items = await run_sync(_fetch)
        return {"items": items}
    
    @router.get("/analytics/top_keywords")
    async def top_keywords(
        _: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        user_id: Optional[int] = Query(None),
        machine_name: Optional[str] = Query(None),
        limit: int = Query(20, ge=1, le=100),
    ):
        """Get top keywords from queries."""
        def _fetch():
            with SessionLocal() as session:
                query = select(QueryHistory.query_text).select_from(QueryHistory)
                
                # Apply filters
                if start_date:
                    try:
                        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                        query = query.where(QueryHistory.created_at >= start_dt)
                    except Exception:
                        pass
                
                if end_date:
                    try:
                        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                        query = query.where(QueryHistory.created_at <= end_dt)
                    except Exception:
                        pass
                
                if user_id:
                    query = query.where(QueryHistory.user_id == user_id)
                
                if machine_name:
                    query = query.where(QueryHistory.machine_name == machine_name)
                
                # Limit to recent queries for performance
                query = query.order_by(desc(QueryHistory.created_at)).limit(10000)
                
                results = session.execute(query).all()
                
                # Extract keywords
                stop_words = {'what', 'how', 'why', 'where', 'when', 'who', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'and', 'or', 'but', 'if', 'then', 'else', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should', 'may', 'might', 'must'}
                keyword_counts = {}
                
                for row in results:
                    query_text = row.query_text.lower()
                    # Tokenize: split on whitespace and punctuation
                    words = re.findall(r'\b\w+\b', query_text)
                    for word in words:
                        if len(word) > 2 and word not in stop_words:
                            keyword_counts[word] = keyword_counts.get(word, 0) + 1
                
                items = [
                    {"keyword": keyword, "count": count}
                    for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
                ]
                return items
        
        from ..utils.db import run_sync
        items = await run_sync(_fetch)
        return {"items": items}

    return router

