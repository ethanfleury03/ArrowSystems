"""
Global Machine Models Configuration

This module defines the allowed machine models for the system.
All machine_model values must be one of the models in ALLOWED_MACHINE_MODELS.

To update the list:
1. Add new models to ALLOWED_MACHINE_MODELS below
2. Restart the backend server
"""

# Special value indicating document applies to any machine
ANY_MACHINE = "Any"

# Special value indicating document applies to all users (always included)
GENERAL_MACHINE = "GENERAL"

# TODO: Populate this list with the final machine models
ALLOWED_MACHINE_MODELS: list[str] = [
    ANY_MACHINE,  # Special option for documents that apply to any machine
    GENERAL_MACHINE,  # Special option for documents that apply to all users (always included)
    "2800 Series Mini Laser Pro",
    "Duraflex",
    "Anycut",
    "anyCutII",
    "anyCutIII",
    "Anytron AnyJet",
    "ANYTRON Any-002",
    "Digital Die Cutter VR350",
    "DuraLink",
    "DuraBolt",
    "DuraCore",
    "EZCut 330",
    "EZCut 350R"
]


def is_valid_machine_model(model: str | None) -> bool:
    """
    Check if a machine model is in the allowed list.
    
    Args:
        model: Machine model string to validate (can be None)
        
    Returns:
        True if model is in ALLOWED_MACHINE_MODELS, False otherwise
    """
    if model is None:
        return False
    return model in ALLOWED_MACHINE_MODELS


def is_valid_machine_model_list(models: list[str] | None) -> bool:
    """
    Check if a list of machine models are all valid.
    
    Args:
        models: List of machine model strings to validate (can be None or empty)
        
    Returns:
        True if all models are in ALLOWED_MACHINE_MODELS, False otherwise
    """
    if models is None or len(models) == 0:
        return False
    # If "Any" is in the list, it should be the only item
    if ANY_MACHINE in models and len(models) > 1:
        return False
    # Check all models are valid
    return all(is_valid_machine_model(model) for model in models)


def get_allowed_machine_models() -> list[str]:
    """
    Get the list of allowed machine models.
    
    Returns:
        List of allowed machine model strings
    """
    return ALLOWED_MACHINE_MODELS.copy()


def get_machine_models_for_selection() -> list[str]:
    """
    Get machine models that can be selected by customers in the UI.
    Excludes special values like "GENERAL" and "Any".
    
    Returns:
        List of selectable machine model strings (excludes GENERAL and Any)
    """
    return [m for m in ALLOWED_MACHINE_MODELS if m not in [GENERAL_MACHINE, ANY_MACHINE]]


def normalize_machine_models(raw) -> list[str]:
    """
    Accepts machine_models stored as JSON/text/list and always returns a list[str].
    Filters out any values not in ALLOWED_MACHINE_MODELS.
    
    Args:
        raw: Machine models in various formats (list, str, None, JSON string)
        
    Returns:
        Normalized list of valid machine model strings
    """
    import json
    
    if raw is None:
        return []
    
    # Handle JSON string (SQLite)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw else []
        except (json.JSONDecodeError, TypeError):
            return []
    
    # Handle list
    if isinstance(raw, list):
        # Filter to only valid models
        normalized = [m for m in raw if isinstance(m, str) and is_valid_machine_model(m)]
        return normalized
    
    return []


def get_effective_machines_for_user(role: str, user_machine_models: list[str]) -> list[str]:
    """
    Returns the effective list of machine models that this user should see.
    
    Rules:
        - If user has machine_models assigned: use those machines (for all roles including ADMIN)
        - If user has NO machine_models assigned AND is ADMIN/TECHNICIAN: all machines (ALLOWED_MACHINE_MODELS)
        - If user has NO machine_models assigned AND is CUSTOMER: empty list (admin must assign machines)
        - "GENERAL" is ALWAYS included for ALL users (including customers with no assigned machines)
        - Customers get: GENERAL + (admin-assigned machines)
        - If role is unknown: fall back to customer-like behavior
    
    Args:
        role: User role (ADMIN, TECHNICIAN, CUSTOMER, or lowercase variants)
        user_machine_models: List of machine models assigned to the user
        
    Returns:
        List of effective machine models (always includes GENERAL for all users)
    
    NOTE: GENERAL is automatically included for all users - admin doesn't need to select it.
    """
    role_upper = role.upper() if role else ""
    
    # Normalize user_machine_models
    user_machine_models = normalize_machine_models(user_machine_models)
    
    # If user has machine_models assigned, use those (for all roles including ADMIN)
    if user_machine_models and len(user_machine_models) > 0:
        effective_machines = user_machine_models.copy()
    else:
        # User has no machine_models assigned
        if role_upper in ["ADMIN", "TECHNICIAN"]:
            # Admins and technicians without assigned machines get full access
            effective_machines = ALLOWED_MACHINE_MODELS.copy()
        else:
            # Customers without assigned machines get no machine access (only GENERAL)
            effective_machines = []
    
    # Always ensure GENERAL is included for ALL users (even customers with no other machines)
    # GENERAL doesn't need to be selected by admin - it's automatically included
    if GENERAL_MACHINE in ALLOWED_MACHINE_MODELS and GENERAL_MACHINE not in effective_machines:
        effective_machines.append(GENERAL_MACHINE)
    
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for m in effective_machines:
        if m not in seen:
            seen.add(m)
            result.append(m)
    
    return result

