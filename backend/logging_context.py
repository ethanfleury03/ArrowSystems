"""
Logging Context - Context variables for structured logging

Provides context variables for request_id, user_id, and user_role
that are automatically included in all log entries.
"""

from contextvars import ContextVar
from typing import Optional

# Context variables for logging
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
user_role_var: ContextVar[Optional[str]] = ContextVar('user_role', default=None)


def set_request_id(request_id: str) -> None:
    """Set the request ID in the current context."""
    request_id_var.set(request_id)


def get_request_id() -> Optional[str]:
    """Get the request ID from the current context."""
    return request_id_var.get()


def set_user_id(user_id: str) -> None:
    """Set the user ID in the current context."""
    user_id_var.set(str(user_id))


def get_user_id() -> Optional[str]:
    """Get the user ID from the current context."""
    return user_id_var.get()


def set_user_role(role: str) -> None:
    """Set the user role in the current context."""
    user_role_var.set(role)


def get_user_role() -> Optional[str]:
    """Get the user role from the current context."""
    return user_role_var.get()


def clear_context() -> None:
    """Clear all context variables."""
    request_id_var.set(None)
    user_id_var.set(None)
    user_role_var.set(None)


def get_logging_context() -> dict:
    """Get all logging context as a dictionary."""
    context = {}
    
    request_id = get_request_id()
    if request_id:
        context['request_id'] = request_id
    
    user_id = get_user_id()
    if user_id:
        context['user_id'] = user_id
    
    user_role = get_user_role()
    if user_role:
        context['role'] = user_role
    
    return context






























