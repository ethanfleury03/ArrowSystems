"""
DuraFlex Technical Assistant - Main Application
Enterprise RAG Interface with Authentication and Visual Content Display

Version: 1.0.0
Author: Arrow Systems Inc
"""

import streamlit as st
import sys
import warnings
from pathlib import Path
import logging
from datetime import datetime
import threading

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import components
from components.auth import AuthManager, render_login_page, render_user_profile_sidebar
from components.query_interface import (
    render_query_controls,
    render_query_input,
    render_query_history,
    add_to_query_history
)
from components.results_display import render_results
from utils.session_manager import init_session_state, increment_query_count, get_session_stats
from utils.export_utils import render_export_options

# Import RAG system - will be imported conditionally based on mock mode
# from query import EliteRAGQuery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_handler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="DuraFlex Technical Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_custom_css():
    """Load custom CSS for polished UI."""
    st.markdown("""
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background-color: #f5f7fa;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        animation: fadeIn 0.5s ease-in;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        margin: 0.8rem 0 0 0;
        opacity: 0.95;
        font-size: 1.15rem;
    }
    
    /* Stats bar */
    .stats-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    
    /* Query box styling */
    .stTextArea textarea {
        font-size: 1.05rem !important;
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 1rem !important;
        transition: border-color 0.3s;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s;
        border: none;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:not([kind="primary"]) {
        background: white;
        border: 2px solid #e0e0e0;
        color: #333;
    }
    
    .stButton > button:not([kind="primary"]):hover {
        border-color: #667eea;
        color: #667eea;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: white;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f5f7fa;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Metrics */
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 8px;
        font-weight: 600;
        padding: 1rem;
        border: 1px solid #e0e0e0;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea;
    }
    
    /* Success/Info/Warning boxes */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 10px;
        padding: 1rem 1.5rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animated-fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #667eea !important;
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        font-size: 0.95rem;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: #28a745;
        color: white;
        border: none;
    }
    
    .stDownloadButton > button:hover {
        background: #218838;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def initialize_database():
    """Initialize DynamoDB connection (cached for performance)."""
    from utils.dynamodb_manager import DynamoDBManager
    try:
        db = DynamoDBManager()
        logger.info("✅ Database connection initialized")
        return db
    except Exception as e:
        logger.warning(f"⚠️ Database initialization failed: {e}")
        logger.warning("Application will continue without database persistence")
        return None


@st.cache_resource(show_spinner=False)
def initialize_rag_system():
    """Initialize RAG system (cached for performance)."""
    import os
    
    # Check if mock mode is enabled
    use_mock = os.getenv('USE_MOCK_RAG', 'false').lower() == 'true'
    
    if use_mock:
        logger.info("🎭 MOCK MODE ENABLED - Using mock RAG system for UI development")
        from mock_rag import MockEliteRAGQuery
        rag = MockEliteRAGQuery()
        rag.initialize()
        logger.info("✅ Mock RAG system initialized successfully")
        return rag
    
    # Real RAG system
    logger.info("Initializing RAG system...")
    try:
        # Import only when needed (avoids loading torch in mock mode)
        from query import EliteRAGQuery
        
        # Determine storage path (check latest_model first for two-pod workflow)
        if os.path.exists("latest_model"):
            storage_path = "latest_model"
        elif os.path.exists("/workspace/latest_model"):
            storage_path = "/workspace/latest_model"
        # Backward compatibility: check old storage locations
        elif os.path.exists("/workspace/storage"):
            storage_path = "/workspace/storage"
        elif os.path.exists("./storage"):
            storage_path = "./storage"
        else:
            raise FileNotFoundError(
                "Index not found. Please run 'python ingest.py' first, or pull from git if using pre-built index."
            )
        
        logger.info(f"Using storage path: {storage_path}")
        rag = EliteRAGQuery(cache_dir="/root/.cache/huggingface/hub")
        rag.initialize(storage_dir=storage_path)
        logger.info("RAG system initialized successfully")
        return rag
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}", exc_info=True)
        raise


def render_header():
    """Render application header."""
    import os
    
    st.markdown("""
    <div class="main-header animated-fade-in">
        <h1>🔧 DuraFlex Technical Assistant</h1>
        <p>Intelligent Knowledge System • Powered by Advanced AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show mock mode indicator
    if os.getenv('USE_MOCK_RAG', 'false').lower() == 'true':
        st.warning("🎭 **MOCK MODE ACTIVE** - Using simulated responses for UI development. Set `USE_MOCK_RAG=false` for real RAG system.", icon="⚠️")


def render_stats_bar():
    """Render statistics bar."""
    stats = get_session_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👤 User", stats['user'])
    
    with col2:
        st.metric("📊 Queries This Session", stats['total_queries'])
    
    with col3:
        st.metric("⏱️ Session Duration", stats['session_duration'])
    
    with col4:
        status = "🟢 Ready" if stats['models_loaded'] else "🟡 Loading..."
        st.metric("🤖 AI Status", status)


def main_application():
    """Main application interface (after authentication)."""
    
    # Load custom CSS (with error handling)
    try:
        load_custom_css()
    except Exception as e:
        logger.warning(f"Could not load custom CSS: {e}")
    
    # Render header
    render_header()
    
    # Render stats bar (models not yet loaded)
    render_stats_bar()
    
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        render_user_profile_sidebar(st.session_state['auth_manager'])
        
        # Navigation
        st.markdown("---")
        st.markdown("### 📂 Navigation")
        if st.button("💾 View Saved Answers", use_container_width=True):
            st.session_state['show_saved_answers'] = True
            st.rerun()
        
        if st.button("🔍 Search Interface", use_container_width=True):
            st.session_state['show_saved_answers'] = False
            st.rerun()
        
        st.markdown("---")
        
        query_params = render_query_controls()
        render_query_history()
        
        # System info
        with st.sidebar.expander("ℹ️ System Information"):
            st.markdown("""
            **Model:** BGE-Large-EN-v1.5  
            **Search:** Hybrid (Dense + BM25)  
            **Reranking:** BGE-Reranker-Large  
            **Content Types:** Text, Tables, Images
            """)
    
    # Main content area - show saved answers or search interface
    if st.session_state.get('show_saved_answers', False):
        from components.feedback_ui import render_saved_answers_page
        render_saved_answers_page()
        return  # Exit early, don't show search interface
    
    query = render_query_input()
    
    if query:
        # Ensure models are loaded before querying
        if not st.session_state.get('models_initialized', False):
            st.warning("⚠️ AI models are still loading. Please wait a moment and try again...")
            with st.spinner("Loading models now..."):
                try:
                    st.session_state['rag_system'] = initialize_rag_system()
                    st.session_state['models_initialized'] = True
                    st.success("✅ Models ready! Rerun your query.")
                except Exception as e:
                    st.error(f"Failed to load models: {e}")
            st.stop()
        
        rag_system = st.session_state['rag_system']
        # Persist last-used query params for caching keys
        st.session_state['last_top_k'] = query_params['top_k']
        st.session_state['last_alpha'] = query_params['alpha']
        
        # Add to history
        add_to_query_history(query)
        increment_query_count()
        
        # Execute query
        with st.spinner("🔍 Searching knowledge base and generating answer..."):
            try:
                import time
                start_time = time.time()
                
                response = rag_system.query(
                    query=query,
                    top_k=query_params['top_k'],
                    alpha=query_params['alpha'],
                    metadata_filters=query_params.get('metadata_filters'),
                    dynamic_windowing=query_params.get('dynamic_windowing', True)
                )
                
                response_time_ms = int((time.time() - start_time) * 1000)
                
                # Store in session
                st.session_state['current_response'] = response
                st.session_state['last_processed_query'] = query
                st.session_state['feedback_query'] = query  # For feedback system (different key to avoid conflict)
                
                # Save to database
                if 'db' in st.session_state and st.session_state['db'] is not None:
                    try:
                        query_id = st.session_state['db'].save_query(
                            user=st.session_state.get('username', 'unknown'),
                            query_text=query,
                            answer_text=response.answer,
                            intent_type=response.intent.intent_type,
                            intent_confidence=response.intent.confidence,
                            sources=[s['name'] for s in response.sources],
                            confidence=response.confidence,
                            response_time_ms=response_time_ms,
                            session_id=st.session_state.get('session_id', 'unknown')
                        )
                        st.session_state['current_query_id'] = query_id
                        logger.info(f"Query saved to database: {query_id}")
                    except Exception as e:
                        logger.warning(f"Failed to save query to database: {e}")
                
                logger.info(f"Query processed successfully: {query[:50]}...")
                
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                st.error(f"❌ Error processing query: {e}")
                st.stop()
        
        # Display results
        st.markdown("## 📋 Results")
        render_results(response)
        
        st.markdown("---")
        
        # Export options
        render_export_options(response, query)
        
    elif st.session_state.get('current_response'):
        # Show previous results
        st.markdown("## 📋 Previous Results")
        last_query = st.session_state.get('last_processed_query', 'Previous query')
        st.info(f"Showing results for: *{last_query}*")
        render_results(st.session_state['current_response'])
        
        st.markdown("---")
        render_export_options(st.session_state['current_response'], last_query)
    
    else:
        # Show welcome message
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <h2 style="color: #667eea;">👋 Welcome to DuraFlex Technical Assistant!</h2>
            <p style="font-size: 1.1rem; color: #666;">
                Ask any question about DuraFlex printers, troubleshooting, maintenance, or specifications.
            </p>
            <p style="color: #999; margin-top: 1rem;">
                💡 Try the example questions below to get started
            </p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main entry point."""
    
    # Initialize session state
    init_session_state()
    
    # Kick off background preload of RAG system (once per session)
    if not st.session_state.get('preload_started', False):
        st.session_state['preload_started'] = True
        try:
            # initialize_rag_system is @st.cache_resource → warms the cache
            threading.Thread(target=initialize_rag_system, daemon=True).start()
            logger.info("🔄 Background preload started for RAG system")
        except Exception as e:
            logger.warning(f"Preload thread failed to start: {e}")
    
    # Initialize auth manager
    if 'auth_manager' not in st.session_state:
        st.session_state['auth_manager'] = AuthManager()
    
    auth_manager = st.session_state['auth_manager']
    
    # Check authentication
    if not auth_manager.is_authenticated():
        # Show login page
        render_login_page(auth_manager)
    else:
        # Check session timeout
        if not auth_manager.check_session_timeout(timeout_hours=24):
            st.warning("⚠️ Your session has expired. Please login again.")
            st.stop()
        
        # Ensure models are ready after login (instant if background preload finished)
        if not st.session_state.get('models_initialized', False):
            try:
                st.session_state['rag_system'] = initialize_rag_system()
                st.session_state['models_initialized'] = True
                logger.info("✅ RAG system ready after login (preloaded if available)")
            except Exception as e:
                logger.error(f"Failed to initialize RAG system after login: {e}", exc_info=True)
                st.error("⚠️ Model loading failed. Please refresh.")
                st.stop()
        
        # Initialize database (once per session)
        if 'db' not in st.session_state:
            st.session_state['db'] = initialize_database()
        
        # Show main application
        main_application()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error(f"❌ Application Error: {e}")
        st.code(str(e))
        
        # Show traceback for debugging
        import traceback
        with st.expander("🔍 Error Details (for debugging)"):
            st.code(traceback.format_exc())
        
        st.info("Please refresh the page or contact support if the problem persists.")

