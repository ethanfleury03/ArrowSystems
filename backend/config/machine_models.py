"""
Global Machine Models Configuration

This module defines the allowed machine models for the system.
All machine_model values must be one of the models in ALLOWED_MACHINE_MODELS.

To update the list:
1. Add new models to ALLOWED_MACHINE_MODELS below
2. Restart the backend server
"""

# TODO: Populate this list with the final machine models
ALLOWED_MACHINE_MODELS: list[str] = [
    # I will populate this later
    # Example format:
    # "330R",
    # "DuraFlex",
    # "DuraCore",
    # "anyCUT",
    # "EZCut",
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


def get_allowed_machine_models() -> list[str]:
    """
    Get the list of allowed machine models.
    
    Returns:
        List of allowed machine model strings
    """
    return ALLOWED_MACHINE_MODELS.copy()

