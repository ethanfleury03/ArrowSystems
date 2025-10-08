"""
Query Interface Component
Handles the query input, parameter controls, and example queries
"""

import streamlit as st
from typing import Dict, List, Any
import yaml


def load_app_config() -> Dict:
    """Load application configuration."""
    try:
        with open("config/app_config.yaml") as f:
            return yaml.safe_load(f)
    except:
        return {}


def render_query_controls() -> Dict[str, Any]:
    """
    Render query parameter controls in sidebar.
    
    Returns:
        Dictionary of query parameters
    """
    config = load_app_config()
    rag_config = config.get('rag', {})
    
    st.sidebar.markdown("### ⚙️ Query Settings")
    
    # Top-K slider
    top_k = st.sidebar.slider(
        "📊 Chunks to Retrieve",
        min_value=rag_config.get('min_top_k', 1),
        max_value=rag_config.get('max_top_k', 50),
        value=rag_config.get('default_top_k', 10),
        help="Number of document chunks to retrieve. More chunks = more context but slower."
    )
    
    # Alpha slider (hybrid search weight)
    alpha = st.sidebar.slider(
        "🔍 Search Mode",
        min_value=0.0,
        max_value=1.0,
        value=rag_config.get('default_alpha', 0.5),
        step=0.1,
        help="0.0 = Keyword search (BM25) only\n0.5 = Balanced (recommended)\n1.0 = Semantic search only"
    )
    
    # Visual indicator for alpha
    if alpha < 0.3:
        search_mode = "🔤 Keyword-focused"
        mode_color = "#ffc107"
    elif alpha > 0.7:
        search_mode = "🧠 Semantic-focused"
        mode_color = "#17a2b8"
    else:
        search_mode = "⚖️ Balanced"
        mode_color = "#28a745"
    
    st.sidebar.markdown(f"""
    <div style="background: {mode_color}20; padding: 0.5rem; border-radius: 5px; border-left: 3px solid {mode_color};">
        <strong style="color: {mode_color};">{search_mode}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Advanced options expander
    with st.sidebar.expander("🔧 Advanced Options"):
        # Content type filters
        st.markdown("**Content Types:**")
        include_text = st.checkbox("📄 Text", value=True)
        include_tables = st.checkbox("📊 Tables", value=True)
        include_images = st.checkbox("🖼️ Images", value=True)
        include_captions = st.checkbox("🏷️ Captions", value=True)
        
        # Dynamic windowing
        dynamic_windowing = st.checkbox(
            "📐 Dynamic Context Windowing",
            value=True,
            help="Automatically adjust context window based on relevance scores"
        )
        
        # Metadata filters
        st.markdown("**Document Filters:**")
        filter_by_doc = st.checkbox("Filter by document", value=False)
        selected_doc = None
        if filter_by_doc:
            # This will be populated with actual documents
            selected_doc = st.selectbox(
                "Select document:",
                ["All Documents", "Operations Guide", "Troubleshooting Guide", "Installation Guide"]
            )
    
    # Build content types list
    content_types = []
    if include_text:
        content_types.append("text")
    if include_tables:
        content_types.append("table")
    if include_images:
        content_types.append("image")
    if include_captions:
        content_types.append("figure_caption")
    
    return {
        'top_k': top_k,
        'alpha': alpha,
        'content_types': content_types if content_types else None,
        'dynamic_windowing': dynamic_windowing,
        'metadata_filters': None  # Can be extended
    }


def render_example_queries():
    """Render example queries section."""
    config = load_app_config()
    examples = config.get('ui', {}).get('example_queries', [])
    
    if examples:
        with st.expander("💡 Example Questions", expanded=False):
            st.markdown("Click on any example to use it:")
            
            cols = st.columns(2)
            for idx, example in enumerate(examples):
                col = cols[idx % 2]
                with col:
                    if st.button(
                        f"🔹 {example[:50]}..." if len(example) > 50 else f"🔹 {example}",
                        key=f"example_{idx}",
                        use_container_width=True
                    ):
                        st.session_state['query_input'] = example
                        st.rerun()


def render_query_history():
    """Render query history in sidebar."""
    if 'query_history' not in st.session_state:
        st.session_state['query_history'] = []
    
    history = st.session_state['query_history']
    
    if history:
        st.sidebar.markdown("### 📜 Recent Queries")
        
        # Show last 5 queries
        for idx, item in enumerate(reversed(history[-5:])):
            query_text = item['query']
            timestamp = item['timestamp']
            
            # Truncate long queries
            display_text = query_text if len(query_text) <= 40 else query_text[:40] + "..."
            
            if st.sidebar.button(
                f"🕐 {display_text}",
                key=f"history_{idx}",
                help=f"Asked at {timestamp}"
            ):
                st.session_state['query_input'] = query_text
                st.rerun()
        
        # Clear history button
        if st.sidebar.button("🗑️ Clear History", use_container_width=True):
            st.session_state['query_history'] = []
            st.rerun()
        
        st.sidebar.markdown("---")


def render_query_input() -> str:
    """
    Render the main query input interface.
    
    Returns:
        The query string
    """
    st.markdown("### 💬 Ask a Question")
    
    # Initialize session state for query if not exists
    if 'query_input' not in st.session_state:
        st.session_state['query_input'] = ""
    
    # Query input
    query = st.text_area(
        "Type your question here:",
        value=st.session_state.get('query_input', ''),
        height=100,
        placeholder="e.g., How do I troubleshoot print quality issues on the DuraFlex printer?",
        help="Ask any question about DuraFlex printers, troubleshooting, maintenance, or technical specifications",
        key="current_query"
    )
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        search_button = st.button(
            "🔍 Search Knowledge Base",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        clear_button = st.button("🔄 Clear", use_container_width=True)
    
    with col3:
        # Voice input placeholder (can be implemented with speech recognition)
        voice_button = st.button("🎤 Voice", use_container_width=True, disabled=True)
    
    if clear_button:
        st.session_state['query_input'] = ""
        st.rerun()
    
    # Render example queries
    render_example_queries()
    
    return query if search_button and query.strip() else None


def add_to_query_history(query: str):
    """Add query to history."""
    from datetime import datetime
    
    if 'query_history' not in st.session_state:
        st.session_state['query_history'] = []
    
    # Avoid duplicates of the last query
    if not st.session_state['query_history'] or \
       st.session_state['query_history'][-1]['query'] != query:
        st.session_state['query_history'].append({
            'query': query,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Keep only last 50 queries
        if len(st.session_state['query_history']) > 50:
            st.session_state['query_history'] = st.session_state['query_history'][-50:]

