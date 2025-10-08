"""
Session Management Utilities
Handles session state, caching, and application state management
"""

import streamlit as st
from typing import Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def init_session_state():
    """Initialize all session state variables."""
    
    # Authentication
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    if 'user_info' not in st.session_state:
        st.session_state['user_info'] = None
    
    # RAG System
    if 'rag_system' not in st.session_state:
        st.session_state['rag_system'] = None
    
    if 'models_initialized' not in st.session_state:
        st.session_state['models_initialized'] = False
    
    # Query Management
    if 'query_history' not in st.session_state:
        st.session_state['query_history'] = []
    
    if 'last_processed_query' not in st.session_state:
        st.session_state['last_processed_query'] = ''
    
    if 'current_response' not in st.session_state:
        st.session_state['current_response'] = None
    
    # UI State
    if 'query_input' not in st.session_state:
        st.session_state['query_input'] = ''
    
    if 'show_advanced' not in st.session_state:
        st.session_state['show_advanced'] = False
    
    # Statistics
    if 'total_queries' not in st.session_state:
        st.session_state['total_queries'] = 0
    
    if 'session_start' not in st.session_state:
        st.session_state['session_start'] = datetime.now()


def get_session_value(key: str, default: Any = None) -> Any:
    """
    Safely get a value from session state.
    
    Args:
        key: Session state key
        default: Default value if key doesn't exist
    
    Returns:
        Value from session state or default
    """
    return st.session_state.get(key, default)


def set_session_value(key: str, value: Any):
    """
    Set a value in session state.
    
    Args:
        key: Session state key
        value: Value to set
    """
    st.session_state[key] = value


def increment_query_count():
    """Increment the total query counter."""
    st.session_state['total_queries'] = st.session_state.get('total_queries', 0) + 1


def get_session_stats() -> dict:
    """
    Get session statistics.
    
    Returns:
        Dictionary of session statistics
    """
    session_start = st.session_state.get('session_start', datetime.now())
    session_duration = datetime.now() - session_start
    
    return {
        'total_queries': st.session_state.get('total_queries', 0),
        'session_duration': str(session_duration).split('.')[0],  # Remove microseconds
        'queries_in_history': len(st.session_state.get('query_history', [])),
        'user': st.session_state.get('user_info', {}).get('name', 'Unknown'),
        'models_loaded': st.session_state.get('models_initialized', False)
    }


def clear_session():
    """Clear all session state (except authentication)."""
    keys_to_keep = ['authenticated', 'user_info', 'login_time', 'last_activity']
    
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    
    # Reinitialize
    init_session_state()
    logger.info("Session state cleared")

