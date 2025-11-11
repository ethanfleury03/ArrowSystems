"""Components module for DuraFlex Technical Assistant."""

# Lazy imports to prevent cascade failures
# Import components directly in app.py instead of through __init__

__all__ = [
    'AuthManager',
    'render_login_page',
    'render_user_profile_sidebar',
    'render_query_controls',
    'render_query_input',
    'add_to_query_history',
    'render_results'
]

