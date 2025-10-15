"""
Authentication Module for DuraFlex Technical Assistant
Handles user login, session management, and role-based access control
"""

import streamlit as st
import yaml
import hashlib
import hmac
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple


class AuthManager:
    """Manages authentication and user sessions."""
    
    def __init__(self, users_config_path: str = "config/users.yaml"):
        self.users_config_path = Path(users_config_path)
        self.users = self._load_users()
    
    def _load_users(self) -> Dict:
        """Load user credentials from YAML file."""
        if self.users_config_path.exists():
            with open(self.users_config_path) as f:
                config = yaml.safe_load(f)
                return config.get('credentials', {}).get('usernames', {})
        return {}
    
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt using SHA-256."""
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def verify_credentials(self, username: str, password: str) -> Tuple[bool, Optional[Dict]]:
        """
        Verify user credentials.
        
        Returns:
            Tuple of (is_valid, user_info)
        """
        if username not in self.users:
            return False, None
        
        user_data = self.users[username]
        stored_hash = user_data.get('password', '')
        salt = user_data.get('salt', '')
        
        password_hash = self._hash_password(password, salt)
        
        if hmac.compare_digest(password_hash, stored_hash):
            return True, {
                'username': username,
                'name': user_data.get('name', username),
                'email': user_data.get('email', ''),
                'role': user_data.get('role', 'technician')
            }
        
        return False, None
    
    def initialize_session(self, user_info: Dict):
        """Initialize user session state."""
        st.session_state['authenticated'] = True
        st.session_state['user_info'] = user_info
        st.session_state['login_time'] = datetime.now()
        st.session_state['last_activity'] = datetime.now()
    
    def logout(self):
        """Clear session state and logout user."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return st.session_state.get('authenticated', False)
    
    def get_current_user(self) -> Optional[Dict]:
        """Get current user information."""
        return st.session_state.get('user_info')
    
    def check_session_timeout(self, timeout_hours: int = 24) -> bool:
        """
        Check if session has timed out.
        
        Returns:
            True if session is valid, False if timed out
        """
        if not self.is_authenticated():
            return False
        
        login_time = st.session_state.get('login_time')
        if login_time:
            elapsed = datetime.now() - login_time
            if elapsed > timedelta(hours=timeout_hours):
                self.logout()
                return False
        
        # Update last activity
        st.session_state['last_activity'] = datetime.now()
        return True
    
    def has_role(self, required_role: str) -> bool:
        """Check if current user has required role."""
        user = self.get_current_user()
        if not user:
            return False
        return user.get('role') == required_role or user.get('role') == 'admin'


def render_login_page(auth_manager: AuthManager):
    """Render the login page UI."""
    
    # Custom CSS for login page
    st.markdown("""
    <style>
    .login-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .login-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .login-header p {
        margin: 1rem 0 0 0;
        opacity: 0.95;
        font-size: 1.2rem;
    }
    
    .login-form {
        background: white;
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }
    
    .demo-credentials {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-top: 1.5rem;
    }
    
    .demo-credentials h4 {
        margin-top: 0;
        color: #667eea;
    }
    
    .demo-credentials code {
        background: #e9ecef;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="login-header">
        <h1>🔧 DuraFlex Technical Assistant</h1>
        <p>Intelligent Knowledge System for Technical Support</p>
        <p style="font-size: 0.95rem; opacity: 0.8; margin-top: 0.5rem;">Powered by Advanced AI & Hybrid Search</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-form">', unsafe_allow_html=True)
        st.markdown("### 🔐 Technician Login")
        st.markdown("---")
        
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                help="Contact your administrator if you need access"
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password"
            )
            
            
            col_a, col_b = st.columns(2)
            with col_a:
                submit = st.form_submit_button("🚀 Login", use_container_width=True, type="primary")
            with col_b:
                remember = st.checkbox("Remember me", value=True)
            
            if submit:
                if not username or not password:
                    st.error("⚠️ Please enter both username and password.")
                else:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Login attempt for user: {username}")
                    
                    is_valid, user_info = auth_manager.verify_credentials(username, password)
                    logger.info(f"Credentials valid: {is_valid}")
                    
                    if is_valid:
                        auth_manager.initialize_session(user_info)
                        logger.info(f"Session initialized for: {user_info['name']}")
                        st.success(f"✅ Welcome back, {user_info['name']}!")
                        st.balloons()
                        # Small delay to show success message
                        import time
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        logger.warning(f"Invalid login attempt for: {username}")
                        st.error("❌ Invalid username or password. Please try again.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Demo credentials info
        st.markdown("""
        <div class="demo-credentials">
            <h4>📝 Demo Credentials</h4>
            <p><strong>Admin:</strong> <code>admin</code> / <code>admin123</code></p>
            <p><strong>Technician:</strong> <code>tech1</code> / <code>tech123</code></p>
            <p style="font-size: 0.9rem; color: #6c757d; margin-top: 1rem;">
                💡 Contact your system administrator for production credentials
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_user_profile_sidebar(auth_manager: AuthManager):
    """Render user profile in sidebar."""
    user = auth_manager.get_current_user()
    
    if user:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 User Profile")
        
        col1, col2 = st.sidebar.columns([1, 3])
        with col1:
            # User avatar (you can customize with real images)
            st.markdown(f"""
            <div style="
                width: 50px;
                height: 50px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 1.5rem;
                font-weight: bold;
            ">
                {user['name'][0].upper()}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**{user['name']}**")
            st.caption(f"@{user['username']}")
            role_badge = "🔑 Admin" if user['role'] == 'admin' else "🔧 Technician"
            st.caption(role_badge)
        
        # Logout button
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            auth_manager.logout()
            st.success("✅ Logged out successfully! RAG system will reset on next login.")
            st.rerun()
        
        st.sidebar.markdown("---")

