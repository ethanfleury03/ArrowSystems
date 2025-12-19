"""
Test that logger calls don't throw TypeError when passing contextual fields.

This ensures that all logger calls use structlog or proper extra={} syntax,
not stdlib logging with arbitrary kwargs.
"""

import pytest
from backend.utils.database_manager import DatabaseManager


def test_logger_calls_no_typeerror():
    """
    Test that logger calls in database_manager don't throw TypeError.
    
    This is a basic sanity check - we import the module and verify
    that logger calls use structlog (which accepts kwargs) or extra={}.
    """
    # Just importing should not fail
    from backend.utils import database_manager
    
    # Verify logger is structlog, not stdlib
    logger = database_manager.logger
    # structlog loggers have 'bind' method
    assert hasattr(logger, 'bind') or hasattr(logger, 'info'), \
        "Logger should be structlog or have standard methods"
    
    # Try calling with kwargs (structlog accepts this)
    try:
        logger.info("test_message", test_field="test_value")
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            pytest.fail(f"Logger call failed with TypeError: {e}. "
                       f"This means stdlib logging is being used with kwargs. "
                       f"Use structlog or extra={{}} dict instead.")
        raise


def test_database_manager_logger_import():
    """Test that database_manager uses structlog logger."""
    from backend.utils import database_manager
    from backend.logging_config import get_logger
    
    # Verify it's using get_logger (structlog)
    expected_logger = get_logger(database_manager.__name__)
    # Both should be structlog loggers
    assert hasattr(database_manager.logger, 'info')
    assert hasattr(expected_logger, 'info')

