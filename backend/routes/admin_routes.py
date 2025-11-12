from __future__ import annotations

from typing import Callable, Dict, List, Optional

import jwt
from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from ..security import decode_access_token
from ..utils.database_manager import DatabaseManager


class AdminUserResponse(BaseModel):
    id: str
    email: Optional[str] = None
    name: Optional[str] = None
    role: str
    company_name: Optional[str] = None
    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None
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


class AdminUserUpdateRequest(BaseModel):
    email: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None
    name: Optional[str] = None
    company_name: Optional[str] = None
    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None


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
        return users

    @router.post("/create_user", response_model=AdminUserResponse, status_code=status.HTTP_201_CREATED)
    async def create_user(
        payload: AdminUserCreateRequest = Body(...),
        _: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
    ):
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
        )
        return created

    @router.put("/edit_user/{user_id}", response_model=AdminUserResponse)
    async def edit_user(
        user_id: int,
        payload: AdminUserUpdateRequest = Body(...),
        _: Dict[str, str] = Depends(get_current_admin),
        manager: DatabaseManager = Depends(get_db_manager),
    ):
        try:
            updated = await manager.update_user(
                user_id,
                email=payload.email,
                name=payload.name,
                password=payload.password,
                role=payload.role,
                company_name=payload.company_name,
                contact_name=payload.contact_name,
                contact_phone=payload.contact_phone,
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

    return router

