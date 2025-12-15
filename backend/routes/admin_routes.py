from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
import json
import re
import sys

import jwt
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel
from sqlalchemy import select, func, desc, and_, or_, case, text, inspect
from sqlalchemy.orm import Session, load_only

from ..security import decode_access_token
from ..utils.database_manager import DatabaseManager
from ..utils.db import SessionLocal, QueryHistory, User, AuditLog, MachineModel, DocumentIngestionMetadata, run_sync
from ..utils.audit_log import audit_log
from ..logging_config import get_logger
from ..schemas.query_insights import (
    QueryInsightsCustomer,
    CustomerQueriesResponse,
    CustomerQuerySummary,
    ConversationDetails,
    ConversationMessage,
    RecentQueryLogItem,
)

logger = get_logger(__name__)


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


class MachineListResponse(BaseModel):
    machines: List[str]

class MachineModelResponse(BaseModel):
    id: int
    name: str
    document_count: int
    created_at: str

class MachineModelsListResponse(BaseModel):
    machines: List[MachineModelResponse]
    total_documents: int
    matched_documents: int
    unmatched_documents: int
    unmatched_machine_models: List[str]


class MachineCreateRequest(BaseModel):
    name: str


def create_admin_router(db_manager_getter: Callable[[], Optional[DatabaseManager]]) -> APIRouter:
    router = APIRouter(prefix="/admin", tags=["admin"])

    async def get_db_manager() -> DatabaseManager:
        manager = db_manager_getter()
        if manager is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database not initialized")
        return manager

    async def get_current_admin(
        request: Request,
        manager: DatabaseManager = Depends(get_db_manager),
    ) -> Dict[str, str]:
        """
        Get the current admin user from the JWT.

        IMPORTANT:
        - The Cloud Run backend is protected by IAM and uses the Authorization
          header for the Google ID token.
        - Our own user JWT is passed separately in the `X-User-Token` header
          by the Next.js API routes.
        - Do NOT try to read the user JWT from the Authorization header here,
          or you'll be decoding the Google IAM token instead of our HS256 JWT.
        """
        # Prefer custom header for user JWT (set by frontend API routes)
        token = request.headers.get("X-User-Token")
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing user token",
            )

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
        admin: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
        http_request: Request = None,
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
        
        # Audit log user creation
        await audit_log(
            "admin_created_user",
            level="info",
            user_id=admin.get("email"),
            role=admin.get("role"),
            metadata={
                "created_user_email": payload.email,
                "created_user_role": payload.role,
                "created_user_id": str(created.get("id")),
                "machine_models": machine_models if role_upper == "CUSTOMER" else None,
            },
            request=http_request,
        )
        
        return created

    @router.post("/users", response_model=AdminUserResponse, status_code=status.HTTP_201_CREATED)
    async def create_user_rest(
        payload: AdminUserCreateRequest = Body(...),
        admin: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
        http_request: Request = None,
    ):
        """
        Create a new user (REST-style endpoint at /admin/users).
        This delegates to the existing create_user function.
        """
        return await create_user(payload, admin, manager, http_request)

    @router.put("/edit_user/{user_id}", response_model=AdminUserResponse)
    async def edit_user(
        user_id: int,
        payload: AdminUserUpdateRequest = Body(...),
        admin: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
        http_request: Request = None,
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
            
            # Get user before update for audit log
            user_before = await manager.get_user_by_id(user_id)
            user_before_machines = user_before.get("machine_models", []) if user_before else []
            
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
            
            # Audit log user update
            machines_changed = update_machine_models is not None and update_machine_models != user_before_machines
            role_changed = payload.role and payload.role.upper() != (user_before.get("role", "") if user_before else "").upper()
            
            await audit_log(
                "admin_updated_user",
                level="info",
                user_id=admin.get("email"),
                role=admin.get("role"),
                metadata={
                    "updated_user_id": str(user_id),
                    "updated_user_email": updated.get("email"),
                    "role_changed": role_changed,
                    "machines_changed": machines_changed,
                    "old_machines": user_before_machines,
                    "new_machines": update_machine_models if machines_changed else None,
                },
                request=http_request,
            )
            
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        return updated

    @router.put("/users/{user_id}", response_model=AdminUserResponse)
    async def edit_user_rest(
        user_id: int,
        payload: AdminUserUpdateRequest = Body(...),
        admin: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
        http_request: Request = None,
    ):
        """
        Update user (REST-style endpoint at /admin/users/{user_id}).
        This delegates to the existing edit_user function.
        """
        return await edit_user(user_id, payload, admin, manager, http_request)

    @router.delete("/delete_user/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_user(
        user_id: int,
        admin: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
        http_request: Request = None,
    ):
        # Get user before deletion for audit log
        user_to_delete = await manager.get_user_by_id(user_id)
        if not user_to_delete:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        deleted = await manager.delete_user(user_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        # Audit log user deletion
        await audit_log(
            "admin_deleted_user",
            level="info",
            user_id=admin.get("email"),
            role=admin.get("role"),
            metadata={
                "deleted_user_id": str(user_id),
                "deleted_user_email": user_to_delete.get("email"),
                "deleted_user_role": user_to_delete.get("role"),
            },
            request=http_request,
        )
        
        return None

    # REST-style alias for deleting a user. This keeps existing clients that call
    # /admin/delete_user/{user_id} working while allowing newer clients to use
    # the more conventional /admin/users/{user_id} path.
    @router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_user_rest(
        user_id: int,
        admin: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
        http_request: Request = None,
    ):
        return await delete_user(user_id, admin, manager, http_request)

    # NOTE: Removed duplicate file-based logs endpoint that was conflicting with the audit logs endpoint.
    # The file-based endpoint was returning the wrong response format expected by the frontend.
    # If file-based logging is needed in the future, it should be at a different path like /admin/system-logs.

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
    
    @router.get("/logs")
    async def get_audit_logs(
        admin: Dict[str, str] = Depends(get_current_admin),
        page: int = Query(1, ge=1),
        limit: int = Query(50, ge=1, le=200),
        level: Optional[str] = Query(None),
        event: Optional[str] = Query(None),
        user_id: Optional[str] = Query(None),
        start: Optional[str] = Query(None),
        end: Optional[str] = Query(None),
    ):
        """
        Get paginated audit logs (admin-only).
        
        Filters:
        - level: Filter by log level (info, warning, error)
        - event: Filter by event name
        - user_id: Filter by user ID
        - start: Start date (ISO format)
        - end: End date (ISO format)
        """
        def _fetch():
            with SessionLocal() as session:
                # Debug: Log database connection
                from ..utils.db import DATABASE_URL
                logger.info("audit_logs_query", database="postgres", message="Starting audit logs query")
                
                # Debug: Check if table exists and has data
                inspector = inspect(session.bind)
                tables = inspector.get_table_names()
                logger.info("audit_logs_query", available_tables=tables, message="Checking for audit_logs table")
                
                if "audit_logs" not in tables:
                    logger.warning("audit_logs_query", message="audit_logs table does NOT exist!")
                    return {
                        "logs": [],
                        "page": page,
                        "limit": limit,
                        "total": 0,
                        "total_pages": 1,
                    }
                
                # Direct count query to verify data exists
                try:
                    direct_count = session.execute(text("SELECT COUNT(*) FROM audit_logs")).scalar()
                    logger.info("audit_logs_query", direct_count=direct_count, message="Direct SQL COUNT query result")
                    
                    # Also try to fetch a few rows directly
                    direct_rows = session.execute(text("SELECT id, event, timestamp FROM audit_logs ORDER BY timestamp DESC LIMIT 5")).fetchall()
                    logger.info("audit_logs_query", direct_rows_count=len(direct_rows), message="Direct SQL SELECT result")
                    for row in direct_rows:
                        logger.info("audit_logs_query", row_id=row[0], event_name=row[1], timestamp=str(row[2]), message="Found audit log row")
                except Exception as e:
                    logger.error("audit_logs_query", error=str(e), exc_info=True, message="Direct SQL query failed")
                
                # Build query - use AuditLog model
                query = select(AuditLog)
                # Test if AuditLog is accessible
                try:
                    test_query = select(func.count()).select_from(AuditLog)
                    test_count = session.execute(test_query).scalar()
                    logger.info("audit_logs_query", test_count=test_count, message="Test query with AuditLog model works")
                except Exception as e:
                    logger.error("audit_logs_query", error=str(e), exc_info=True, message="AuditLog model query failed")
                
                # Apply filters
                filters = []
                
                if level:
                    filters.append(AuditLog.level == level.lower())
                
                if event:
                    filters.append(AuditLog.event == event)
                
                if user_id:
                    filters.append(AuditLog.user_id == user_id)
                
                if start:
                    try:
                        start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                        filters.append(AuditLog.timestamp >= start_dt)
                    except Exception:
                        pass
                
                if end:
                    try:
                        end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                        filters.append(AuditLog.timestamp <= end_dt)
                    except Exception:
                        pass
                
                if filters:
                    query = query.where(and_(*filters))
                
                # Get total count
                count_query = select(func.count()).select_from(AuditLog)
                if filters:
                    count_query = count_query.where(and_(*filters))
                total = session.execute(count_query).scalar() or 0
                
                # Apply pagination and ordering
                offset = (page - 1) * limit
                query = query.order_by(desc(AuditLog.timestamp)).offset(offset).limit(limit)
                
                # Execute query
                results = session.execute(query).scalars().all()
                
                # Debug: Log query results
                logger.info("audit_logs_query", 
                          results_count=len(results), 
                          total_count=total,
                          message=f"SQLAlchemy query returned {len(results)} audit logs (total count: {total})")
                if len(results) > 0:
                    logger.info("audit_logs_query", 
                              first_event=results[0].event if hasattr(results[0], 'event') else str(results[0]),
                              first_timestamp=str(results[0].timestamp) if hasattr(results[0], 'timestamp') else None,
                              message="First log from SQLAlchemy query")
                else:
                    logger.warning("audit_logs_query", message="SQLAlchemy query returned 0 results, but direct SQL may have found rows")
                
                # Serialize results
                logs = []
                for log in results:
                    # Handle metadata (could be JSON string or dict)
                    # Note: event_metadata is the Python attribute name, but it maps to 'metadata' column in DB
                    metadata = log.event_metadata
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except Exception:
                            metadata = {}
                    elif metadata is None:
                        metadata = {}
                    
                    logs.append({
                        "id": log.id,
                        "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                        "level": log.level,
                        "event": log.event,
                        "user_id": log.user_id,
                        "role": log.role,
                        "ip_address": log.ip_address,
                        "metadata": metadata,
                        "request_id": log.request_id,
                    })
                
                # Calculate total pages
                total_pages = (total + limit - 1) // limit if total > 0 else 1
                
                return {
                    "logs": logs,
                    "page": page,
                    "limit": limit,
                    "total": total,
                    "total_pages": total_pages,
                }
        
        return await run_sync(_fetch)
    
    @router.post("/logs/test")
    async def test_audit_log(
        admin: Dict[str, str] = Depends(get_current_admin),
        http_request: Request = None,
    ):
        """Test audit logging endpoint (admin-only)."""
        await audit_log(
            "test_event",
            level="info",
            user_id=admin.get("email"),
            role=admin.get("role"),
            metadata={"test": True, "admin": admin.get("email")},
            request=http_request,
        )
        return {"status": "success", "message": "Test audit log created"}

    @router.get("/machines")
    async def list_machines(
        _: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
    ):
        """Get list of all machine models with document counts."""
        def _fetch():
            with SessionLocal() as session:
                # Get all machine model names (for unmatched detection)
                all_machines = session.query(MachineModel).all()
                machine_names_upper = {m.name.upper() for m in all_machines}
                
                # Query machines with document counts
                # Use case-insensitive comparison for machine_model matching
                machines_query = (
                    session.query(
                        MachineModel.id,
                        MachineModel.name,
                        MachineModel.created_at,
                        func.count(DocumentIngestionMetadata.id).label('document_count')
                    )
                    .outerjoin(
                        DocumentIngestionMetadata,
                        func.upper(MachineModel.name) == func.upper(DocumentIngestionMetadata.machine_model)
                    )
                    .group_by(MachineModel.id, MachineModel.name, MachineModel.created_at)
                    .order_by(MachineModel.name)
                )
                
                results = machines_query.all()
                machines_list = [
                    {
                        "id": row.id,
                        "name": row.name,
                        "document_count": row.document_count or 0,
                        "created_at": row.created_at.isoformat() if row.created_at else "",
                    }
                    for row in results
                ]
                
                # Count total documents
                total_docs = session.query(func.count(DocumentIngestionMetadata.id)).scalar() or 0
                
                # Find documents that don't match any machine model (case-insensitive)
                # Get all unique machine_model values from documents (excluding NULL and empty strings)
                from sqlalchemy import or_
                all_doc_machine_models = session.query(
                    func.upper(DocumentIngestionMetadata.machine_model).label('machine_model_upper'),
                    DocumentIngestionMetadata.machine_model.label('machine_model_original')
                ).filter(
                    DocumentIngestionMetadata.machine_model.isnot(None),
                    DocumentIngestionMetadata.machine_model != ""
                ).distinct().all()
                
                unmatched_machine_models = []
                for row in all_doc_machine_models:
                    doc_model_upper = row.machine_model_upper
                    doc_model_original = row.machine_model_original
                    # Skip if the upper value is None (shouldn't happen with filter, but safety check)
                    if doc_model_upper is None:
                        continue
                    # Check if this document's machine_model matches any machine model (case-insensitive)
                    if doc_model_upper not in machine_names_upper:
                        unmatched_machine_models.append(doc_model_original)
                
                # Count unmatched documents (only those with non-empty machine_model that don't match)
                unmatched_count = 0
                if unmatched_machine_models:
                    unmatched_count = (
                        session.query(func.count(DocumentIngestionMetadata.id))
                        .filter(
                            DocumentIngestionMetadata.machine_model.isnot(None),
                            DocumentIngestionMetadata.machine_model != "",
                            func.upper(DocumentIngestionMetadata.machine_model).in_(
                                [m.upper() for m in unmatched_machine_models if m]
                            )
                        )
                        .scalar() or 0
                    )
                
                # Calculate sum of matched documents
                matched_count = sum(m["document_count"] for m in machines_list)
                
                # Log warning if totals don't match
                if total_docs != matched_count + unmatched_count:
                    logger.warning(
                        f"Document count mismatch: total={total_docs}, matched={matched_count}, unmatched={unmatched_count}, difference={total_docs - matched_count - unmatched_count}"
                    )
                    # Log unmatched machine models for debugging
                    if unmatched_machine_models:
                        logger.warning(f"Unmatched machine models found: {unmatched_machine_models}")
                
                return {
                    "machines": machines_list,
                    "total_documents": total_docs,
                    "matched_documents": matched_count,
                    "unmatched_documents": unmatched_count,
                    "unmatched_machine_models": unmatched_machine_models,
                }
        
        return await run_sync(_fetch)

    @router.post("/machines")
    async def create_machine(
        request: MachineCreateRequest,
        admin: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
        http_request: Request = None,
    ):
        """Add a new machine model."""
        # Validate and normalize name
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Machine name cannot be empty")
        
        # Normalize: uppercase, remove extra spaces
        name = " ".join(name.upper().split())
        
        def _create():
            with SessionLocal() as session:
                # Check for duplicates (case-insensitive)
                existing = session.query(MachineModel).filter(
                    func.upper(MachineModel.name) == name.upper()
                ).first()
                
                if existing:
                    raise HTTPException(status_code=400, detail=f"Machine model '{name}' already exists")
                
                # Create new machine model
                machine = MachineModel(name=name)
                session.add(machine)
                session.commit()
                session.refresh(machine)
                return {"name": machine.name, "id": machine.id}
        
        try:
            result = await run_sync(_create)
            
            # Audit log
            await audit_log(
                "machine_model_created",
                level="info",
                user_id=admin.get("email"),
                role=admin.get("role"),
                metadata={"machine_name": result["name"]},
                request=http_request,
            )
            
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating machine model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to create machine model: {str(e)}")

    @router.delete("/machines/{machine_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_machine(
        machine_id: int,
        admin: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
        http_request: Request = None,
    ):
        """Delete a machine model (only if it has no associated documents)."""
        def _delete():
            with SessionLocal() as session:
                # Check if machine exists
                machine = session.query(MachineModel).filter(MachineModel.id == machine_id).first()
                if not machine:
                    raise HTTPException(status_code=404, detail="Machine model not found")
                
                # Check if machine has any associated documents (case-insensitive)
                doc_count = session.query(func.count(DocumentIngestionMetadata.id)).filter(
                    func.upper(DocumentIngestionMetadata.machine_model) == func.upper(machine.name)
                ).scalar() or 0
                
                if doc_count > 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot delete machine model '{machine.name}' because it has {doc_count} associated document(s). Remove all documents first."
                    )
                
                # Delete the machine
                session.delete(machine)
                session.commit()
                return None
        
        try:
            await run_sync(_delete)
            
            # Audit log
            await audit_log(
                "machine_model_deleted",
                level="info",
                user_id=admin.get("email"),
                role=admin.get("role"),
                metadata={"machine_id": machine_id},
                request=http_request,
            )
            
            return None
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting machine model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to delete machine model: {str(e)}")

    # ============================================================================
    # Query Insights Endpoints
    # ============================================================================
    
    @router.get("/query-insights/customers", response_model=List[QueryInsightsCustomer])
    async def get_query_insights_customers(
        _: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
    ):
        """
        Return a list of customers (role='CUSTOMER'), each with:
        - id
        - name
        - total_queries: count of query records for that customer
        - last_query_at: max(created_at) over their queries
        """
        def _fetch():
            try:
                with SessionLocal() as session:
                    # Get all customers with their query stats using LEFT JOIN
                    query = (
                        select(
                            User.id,
                            User.contact_name,
                            User.name,
                            User.email,
                            func.count(QueryHistory.id).label('total_queries'),
                            func.max(QueryHistory.created_at).label('last_query_at')
                        )
                        .outerjoin(QueryHistory, User.id == QueryHistory.user_id)
                        .where(User.role == "CUSTOMER")
                        .group_by(User.id, User.contact_name, User.name, User.email)
                    )
                    
                    results = session.execute(query).all()
                    
                    result = []
                    for row in results:
                        # Use contact_name or name for display
                        customer_name = row.contact_name or row.name or row.email or "Unknown"
                        
                        result.append({
                            "id": str(row.id),
                            "name": customer_name,
                            "total_queries": row.total_queries or 0,
                            "last_query_at": row.last_query_at,
                        })
                    
                    logger.info(f"Query insights: found {len(result)} customers")
                    return result
            except Exception as e:
                logger.error(f"Error fetching query insights customers: {e}", exc_info=True)
                raise
        
        try:
            customers = await run_sync(_fetch)
            return customers
        except Exception as e:
            logger.error(f"Error in get_query_insights_customers: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch customers: {str(e)}"
            )
    
    @router.get(
        "/query-insights/customers/{customer_id}/queries",
        response_model=CustomerQueriesResponse,
    )
    async def get_customer_queries(
        customer_id: str,
        search: Optional[str] = Query(None),
        _: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
    ):
        """
        For a given customer, return:
        - customer_id
        - customer_name
        - total_queries: number of queries for this customer
        - last_query_at: max(created_at) over queries
        - queries: list of CustomerQuerySummary sorted by created_at DESC
        """
        def _fetch():
            with SessionLocal() as session:
                logger.info(f"[QueryInsights] Fetching queries for customer_id={customer_id}")
                
                # Get customer by ID
                try:
                    customer_id_int = int(customer_id)
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid customer_id"
                    )
                
                customer = session.query(User).filter(
                    User.id == customer_id_int,
                    User.role == "CUSTOMER"
                ).first()
                
                if not customer:
                    logger.warning(f"[QueryInsights] Customer not found: id={customer_id}, role=CUSTOMER")
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Customer not found"
                    )
                
                customer_name = customer.contact_name or customer.name or customer.email or "Unknown"
                logger.info(f"[QueryInsights] Found customer: id={customer.id}, email={customer.email}, name={customer_name}")

                # Determine all users that belong to this customer org.
                # We treat users with the same company_name as belonging to the
                # same customer, and include both CUSTOMER and TECHNICIAN roles.
                company_name = (customer.company_name or "").strip()
                users_for_customer = session.query(User).filter(
                    User.company_name == company_name,
                    User.role.in_(["CUSTOMER", "TECHNICIAN"]),
                ).all()

                if not users_for_customer:
                    logger.info("[QueryInsights] No associated users found for customer_id=%s company_name=%s", customer_id, company_name)
                    return {
                        "customer_id": str(customer.id),
                        "customer_name": customer_name,
                        "total_queries": 0,
                        "last_query_at": None,
                        "queries": [],
                    }

                user_ids = [u.id for u in users_for_customer]
                logger.info(
                    "[QueryInsights] Associated users for customer_id=%s: %s",
                    customer_id,
                    [{"id": u.id, "email": u.email, "role": u.role} for u in users_for_customer],
                )

                # Get all queries for this customer org (customer + technicians)
                base_query = (
                    session.query(QueryHistory, User)
                    .join(User, QueryHistory.user_id == User.id)
                    .filter(QueryHistory.user_id.in_(user_ids))
                )
                
                # Apply search filter if provided
                if search:
                    search_term = f"%{search}%"
                    base_query = base_query.filter(
                        QueryHistory.query_text.ilike(search_term)
                    )

                rows = base_query.order_by(desc(QueryHistory.created_at)).all()
                logger.info(
                    "[QueryInsights] Found %d query_history rows for customer org (customer_id=%s)",
                    len(rows),
                    customer_id,
                )

                if not rows:
                    return {
                        "customer_id": str(customer.id),
                        "customer_name": customer_name,
                        "total_queries": 0,
                        "last_query_at": None,
                        "queries": [],
                    }

                # Convert queries to summaries - one per query (simpler than grouping by conversation)
                query_summaries = []
                for qh, user in rows:
                    # Prefer explicit conversation_id from column; fall back to sessionId metadata or query_id
                    conversation_id = qh.conversation_id or f"query_{qh.id}"
                    if not qh.conversation_id and qh.metadata_json:
                        if isinstance(qh.metadata_json, dict):
                            session_id = qh.metadata_json.get("sessionId")
                            if session_id:
                                conversation_id = str(session_id)
                        elif isinstance(qh.metadata_json, str):
                            try:
                                metadata = json.loads(qh.metadata_json)
                                if isinstance(metadata, dict):
                                    session_id = metadata.get("sessionId")
                                    if session_id:
                                        conversation_id = str(session_id)
                            except (json.JSONDecodeError, TypeError):
                                pass

                    # Each query row represents one user+assistant pair
                    message_count = 2 if qh.answer_text else 1

                    query_summaries.append({
                        "id": str(qh.id),
                        "conversation_id": conversation_id,
                        "created_at": qh.created_at,
                        "query_text": qh.query_text or "",
                        "message_count": message_count,
                        "user_id": user.id,
                        "user_email": user.email,
                        "user_role": user.role or "",
                    })

                # Sort by created_at DESC (already sorted by DB query, but ensure)
                query_summaries.sort(key=lambda x: x["created_at"], reverse=True)
                
                # Calculate totals
                total_queries = len(query_summaries)
                last_query_at = query_summaries[0]["created_at"] if query_summaries else None
                
                logger.info(f"[QueryInsights] Returning {total_queries} queries for customer_id={customer_id}")
                
                return {
                    "customer_id": str(customer.id),
                    "customer_name": customer_name,
                    "total_queries": total_queries,
                    "last_query_at": last_query_at,
                    "queries": query_summaries,
                }
        
        try:
            result = await run_sync(_fetch)
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error fetching customer queries: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch customer queries: {str(e)}"
            )
    
    @router.get(
        "/query-insights/conversations/{conversation_id}",
        response_model=ConversationDetails,
    )
    async def get_conversation_details(
        conversation_id: str,
        _: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
    ):
        """
        Return full conversation details:
        - conversation_id
        - customer_id
        - customer_name
        - created_at: timestamp of first message
        - messages: list of messages with role, content, created_at
        """
        def _fetch():
            with SessionLocal() as session:
                # Get all queries for this conversation (by sessionId)
                # Use load_only to exclude updated_at column which may not exist in DB
                # PostgreSQL JSONB query: metadata_json->>'sessionId'
                queries = session.query(QueryHistory).options(
                    load_only(
                        QueryHistory.id,
                        QueryHistory.user_id,
                        QueryHistory.query_text,
                        QueryHistory.answer_text,
                        QueryHistory.response_time_ms,
                        QueryHistory.metadata_json,
                        QueryHistory.created_at,
                        QueryHistory.machine_name,
                        QueryHistory.token_input,
                        QueryHistory.token_output,
                        QueryHistory.token_total,
                        QueryHistory.cost_usd,
                        QueryHistory.sources_json
                    )
                ).filter(
                    QueryHistory.metadata_json.op('->>')('sessionId') == conversation_id
                ).order_by(QueryHistory.created_at).all()
                
                # If no results with sessionId, try treating conversation_id as a query ID
                if not queries:
                    try:
                        query_id = int(conversation_id.replace("query_", ""))
                        queries = session.query(QueryHistory).options(
                            load_only(
                                QueryHistory.id,
                                QueryHistory.user_id,
                                QueryHistory.query_text,
                                QueryHistory.answer_text,
                                QueryHistory.response_time_ms,
                                QueryHistory.metadata_json,
                                QueryHistory.created_at,
                                QueryHistory.machine_name,
                                QueryHistory.token_input,
                                QueryHistory.token_output,
                                QueryHistory.token_total,
                                QueryHistory.cost_usd,
                                QueryHistory.sources_json
                            )
                        ).filter(
                            QueryHistory.id == query_id
                        ).order_by(QueryHistory.created_at).all()
                    except (ValueError, AttributeError):
                        pass
                
                if not queries:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Conversation not found"
                    )
                
                # Get customer info from first query
                first_query = queries[0]
                customer = session.query(User).filter(User.id == first_query.user_id).first()
                
                if not customer:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Customer not found"
                    )
                
                customer_name = customer.contact_name or customer.name or customer.email or "Unknown"
                
                # Build messages: each query has user message (query_text) and assistant message (answer_text)
                messages = []
                for query in queries:
                    # User message
                    if query.query_text:
                        messages.append({
                            "id": f"user_{query.id}",
                            "role": "user",
                            "content": query.query_text,
                            "created_at": query.created_at,
                        })
                    
                    # Assistant message
                    if query.answer_text:
                        messages.append({
                            "id": f"assistant_{query.id}",
                            "role": "assistant",
                            "content": query.answer_text,
                            "created_at": query.created_at,
                        })
                
                if not messages:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="No messages found in conversation"
                    )
                
                return {
                    "conversation_id": conversation_id,
                    "customer_id": str(customer.id),
                    "customer_name": customer_name,
                    "created_at": messages[0]["created_at"],
                    "messages": messages,
                }
        
        try:
            result = await run_sync(_fetch)
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error fetching conversation details: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch conversation: {str(e)}"
            )
    
    @router.get("/query-insights/recent-queries", response_model=List[RecentQueryLogItem])
    async def get_recent_queries(
        limit: int = Query(50, ge=1, le=200),
        _: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
    ):
        """
        Get recent queries across all customers, ordered by created_at DESC.
        Returns queries from both customers and technicians, with customer info resolved.
        """
        def _fetch():
            with SessionLocal() as session:
                # Join QueryHistory with User
                rows = (
                    session.query(QueryHistory, User)
                    .join(User, QueryHistory.user_id == User.id)
                    .order_by(desc(QueryHistory.created_at))
                    .limit(limit)
                    .all()
                )
                
                items: list[RecentQueryLogItem] = []
                for qh, user in rows:
                    # Determine customer_id and customer_name
                    # If user is a CUSTOMER, use their own id/name
                    # If user is a TECHNICIAN, find the CUSTOMER with same company_name
                    if user.role == "CUSTOMER":
                        customer_id = user.id
                        customer_name = user.contact_name or user.name or user.email or "Unknown"
                    else:
                        # Find customer with same company_name
                        customer = session.query(User).filter(
                            User.company_name == user.company_name,
                            User.role == "CUSTOMER"
                        ).first()
                        if customer:
                            customer_id = customer.id
                            customer_name = customer.contact_name or customer.name or customer.email or "Unknown"
                        else:
                            # Fallback: use technician's info if no customer found
                            customer_id = user.id
                            customer_name = user.contact_name or user.name or user.email or "Unknown"
                    
                    # Get conversation_id
                    conversation_id = qh.conversation_id or str(qh.id)
                    
                    items.append(
                        RecentQueryLogItem(
                            id=qh.id,
                            created_at=qh.created_at,
                            customer_id=customer_id,
                            customer_name=customer_name,
                            user_id=user.id,
                            user_email=user.email,
                            user_role=user.role or "",
                            query_text=qh.query_text or "",
                            machine_name=qh.machine_name,
                            conversation_id=conversation_id,
                        )
                    )
                
                return items
        
        try:
            result = await run_sync(_fetch)
            return result
        except Exception as e:
            logger.error(f"Error fetching recent queries: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch recent queries: {str(e)}"
            )

    return router

