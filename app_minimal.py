"""
Minimal version of app.py to isolate the crash
"""
import streamlit as st
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="DuraFlex Technical Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

print("✅ Step 1: Page config set")

# Import components
from components.auth import AuthManager, render_login_page
from utils.session_manager import init_session_state

print("✅ Step 2: Imports successful")

def main():
    """Main entry point."""
    print("✅ Step 3: main() called")
    
    # Initialize session state
    init_session_state()
    print("✅ Step 4: Session state initialized")
    
    # Initialize auth manager
    if 'auth_manager' not in st.session_state:
        st.session_state['auth_manager'] = AuthManager()
    print("✅ Step 5: Auth manager created")
    
    auth_manager = st.session_state['auth_manager']
    
    # Check authentication
    if not auth_manager.is_authenticated():
        print("✅ Step 6: Rendering login page")
        render_login_page(auth_manager)
    else:
        print("✅ Step 7: User authenticated, showing main app")
        st.title("🔧 DuraFlex Technical Assistant")
        st.success("✅ Login successful! Main app would go here.")
        
        if st.sidebar.button("Logout"):
            auth_manager.logout()
            st.rerun()

if __name__ == "__main__":
    try:
        print("✅ Step 0: Starting app")
        main()
        print("✅ Step 8: App completed without errors")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error(f"❌ Application Error: {e}")
        import traceback
        st.code(traceback.format_exc())

