"""
Minimal test app to isolate the crash point
"""
import streamlit as st

print("1. Imports successful")

# Test 1: Can we run set_page_config?
try:
    st.set_page_config(
        page_title="Test",
        page_icon="🔧",
        layout="wide"
    )
    print("2. set_page_config successful")
except Exception as e:
    print(f"ERROR in set_page_config: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Can we import session manager?
try:
    from utils.session_manager import init_session_state
    print("3. Import session_manager successful")
except Exception as e:
    print(f"ERROR importing session_manager: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Can we initialize session?
try:
    init_session_state()
    print("4. init_session_state successful")
except Exception as e:
    print(f"ERROR in init_session_state: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Can we import AuthManager?
try:
    from components.auth import AuthManager
    print("5. Import AuthManager successful")
except Exception as e:
    print(f"ERROR importing AuthManager: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Can we create AuthManager?
try:
    auth_manager = AuthManager()
    print("6. AuthManager() successful")
except Exception as e:
    print(f"ERROR creating AuthManager: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Can we render something?
try:
    st.title("✅ Test App Works!")
    st.success(f"All {6} tests passed!")
    print("7. Rendering successful")
except Exception as e:
    print(f"ERROR in rendering: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Test app completed without crashes!")

