"""
Results Display Component
Handles rendering of query results, sources, tables, images, and structured responses
"""

import streamlit as st
from pathlib import Path
import json
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
from typing import Dict, Any, List
from orchestrator import StructuredResponse


def get_extracted_content_dir():
    """Get the extracted content directory, checking multiple locations."""
    # Check both relative and absolute paths (for RunPod compatibility)
    possible_paths = [
        Path("extracted_content"),
        Path("./extracted_content"),
        Path("/workspace/extracted_content"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def render_answer_chunks(response):
    """Split answer by citations and render each in its own card with feedback."""
    import re
    
    # Split answer by "According to" pattern
    answer_text = response.answer
    
    # Pattern to split: "According to ... [X]:"
    pattern = r'(According to [^[]+\[\d+\]:)'
    parts = re.split(pattern, answer_text)
    
    # Initialize chunk ratings in session state
    if 'chunk_ratings' not in st.session_state:
        st.session_state['chunk_ratings'] = {}
    
    # Process chunks
    chunks = []
    current_chunk = {"header": None, "content": ""}
    
    for i, part in enumerate(parts):
        if re.match(pattern, part):
            # This is a header
            if current_chunk["content"]:
                chunks.append(current_chunk)
            current_chunk = {"header": part.strip(), "content": ""}
        else:
            # This is content
            current_chunk["content"] += part.strip()
    
    # Add last chunk
    if current_chunk["content"]:
        chunks.append(current_chunk)
    
    # If no chunks found (no citations), show as single block
    if not chunks or (len(chunks) == 1 and not chunks[0]["header"]):
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05); border-left: 4px solid #667eea;">
            {answer_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Overall feedback
        st.markdown("")
        st.markdown("**Was this answer helpful?**")
        from components.feedback_ui import render_feedback_buttons
        render_feedback_buttons(response, st.session_state.get('feedback_query', ''), 
                               st.session_state.get('username', 'Unknown'))
        return
    
    # Render each chunk in its own card
    for chunk_idx, chunk in enumerate(chunks):
        if chunk["header"] and chunk["content"]:
            with st.container():
                # Extract citation number from header
                citation_match = re.search(r'\[(\d+)\]', chunk["header"])
                citation_num = citation_match.group(1) if citation_match else str(chunk_idx + 1)
                
                # Card with gradient border
                border_colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']
                border_color = border_colors[chunk_idx % len(border_colors)]
                
                st.markdown(f"""
                <div style="background: white; padding: 1.2rem; border-radius: 10px; 
                            margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); 
                            border-left: 4px solid {border_color};">
                    <strong style="color: {border_color}; font-size: 0.95rem;">{chunk["header"]}</strong>
                    <div style="margin-top: 0.8rem; line-height: 1.6;">
                        {chunk["content"].replace(chr(10), '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Feedback for this chunk
                chunk_key = f"chunk_{chunk_idx}_{citation_num}"
                existing_rating = st.session_state['chunk_ratings'].get(chunk_key)
                
                col1, col2, col3 = st.columns([1, 1, 8])
                
                with col1:
                    if existing_rating == 'helpful':
                        st.button("👍", disabled=True, key=f"chunk_up_{chunk_idx}", 
                                 help="Marked helpful")
                    else:
                        if st.button("👍", key=f"chunk_up_{chunk_idx}", 
                                    help="This citation was helpful"):
                            st.session_state['chunk_ratings'][chunk_key] = 'helpful'
                            st.rerun()
                
                with col2:
                    if existing_rating == 'unhelpful':
                        st.button("👎", disabled=True, key=f"chunk_down_{chunk_idx}", 
                                 help="Marked unhelpful")
                    else:
                        if st.button("👎", key=f"chunk_down_{chunk_idx}", 
                                    help="This citation was unhelpful"):
                            st.session_state['chunk_ratings'][chunk_key] = 'unhelpful'
                            st.rerun()
                
                with col3:
                    if existing_rating == 'helpful':
                        st.caption("✅ Helpful citation")
                    elif existing_rating == 'unhelpful':
                        st.caption("⚠️ Unhelpful citation")
                
                st.markdown("")  # Spacing


def render_source_feedback_buttons(source, source_id, idx):
    """Render thumbs up/down buttons for individual sources."""
    # Generate unique key for this source
    if hasattr(source, 'file_name'):
        source_key = f"{source.file_name}_{source.page_number}"
    else:
        source_key = f"{source.get('name', 'unknown')}_{source.get('page_number', 0)}"
    
    # Check if already rated (using session state for now)
    if 'source_ratings' not in st.session_state:
        st.session_state['source_ratings'] = {}
    
    rating_key = f"rating_{source_key}"
    existing_rating = st.session_state['source_ratings'].get(rating_key)
    
    col1, col2, col3 = st.columns([1, 1, 6])
    
    with col1:
        if existing_rating == 'helpful':
            st.button("👍", disabled=True, key=f"src_up_{idx}_{hash(source_key)}", 
                     help="Marked as helpful")
        else:
            if st.button("👍", key=f"src_up_{idx}_{hash(source_key)}", 
                        help="Mark source as helpful"):
                st.session_state['source_ratings'][rating_key] = 'helpful'
                st.success("✅ Source marked helpful!")
                st.rerun()
    
    with col2:
        if existing_rating == 'unhelpful':
            st.button("👎", disabled=True, key=f"src_down_{idx}_{hash(source_key)}", 
                     help="Marked as unhelpful")
        else:
            if st.button("👎", key=f"src_down_{idx}_{hash(source_key)}", 
                        help="Mark source as unhelpful"):
                st.session_state['source_ratings'][rating_key] = 'unhelpful'
                st.warning("📝 Source marked unhelpful")
                st.rerun()
    
    with col3:
        if existing_rating == 'helpful':
            st.caption("✅ Helpful source")
        elif existing_rating == 'unhelpful':
            st.caption("⚠️ Unhelpful source")


def render_confidence_meter(confidence: float):
    """Render a visual confidence meter."""
    # Determine color based on confidence
    if confidence >= 0.8:
        color = "#28a745"
        label = "High Confidence"
        icon = "🟢"
    elif confidence >= 0.5:
        color = "#ffc107"
        label = "Medium Confidence"
        icon = "🟡"
    else:
        color = "#dc3545"
        label = "Low Confidence"
        icon = "🔴"
    
    confidence_pct = int(confidence * 100)
    
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-weight: 600;">{icon} {label}</span>
            <span style="font-weight: 700; color: {color};">{confidence_pct}%</span>
        </div>
        <div style="background: #e9ecef; border-radius: 10px; height: 12px; overflow: hidden;">
            <div style="background: {color}; width: {confidence_pct}%; height: 100%; border-radius: 10px; transition: width 0.3s;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_intent_badge(intent_type: str, confidence: float):
    """Render intent classification badge."""
    # Intent icons and colors
    intent_config = {
        'definition': {'icon': '📖', 'color': '#17a2b8', 'label': 'Definition'},
        'troubleshooting': {'icon': '🔧', 'color': '#dc3545', 'label': 'Troubleshooting'},
        'comparison': {'icon': '⚖️', 'color': '#6f42c1', 'label': 'Comparison'},
        'reasoning': {'icon': '🧠', 'color': '#28a745', 'label': 'Procedural'},
        'lookup': {'icon': '🔍', 'color': '#ffc107', 'label': 'Lookup'}
    }
    
    config = intent_config.get(intent_type, {'icon': '❓', 'color': '#6c757d', 'label': 'General'})
    
    st.markdown(f"""
    <div style="display: inline-block; padding: 0.4rem 1rem; background: {config['color']}20; 
                border: 2px solid {config['color']}; border-radius: 20px; margin: 0.5rem 0;">
        <span style="font-weight: 600; color: {config['color']};">
            {config['icon']} Query Type: {config['label']} ({int(confidence*100)}% confidence)
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_answer_tab(response: StructuredResponse):
    """Render the answer tab with formatted response and referenced content side-by-side."""
    st.markdown("### 💡 Answer")
    
    # Show intent and confidence
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_intent_badge(response.intent.intent_type, response.intent.confidence)
    
    with col2:
        render_confidence_meter(response.confidence)
    
    st.markdown("---")
    
    # SIDE-BY-SIDE LAYOUT: Answer on left, Referenced content on right
    answer_col, content_col = st.columns([1.5, 1])
    
    with answer_col:
        # Split answer into chunks by citation
        render_answer_chunks(response)
        
        # Keywords
        if response.intent.keywords:
            st.markdown("#### 🔑 Key Terms:")
            keyword_badges = " ".join([
                f"<span style='background: #e9ecef; padding: 0.3rem 0.8rem; border-radius: 15px; "
                f"margin: 0.2rem; display: inline-block; font-size: 0.9rem;'>{kw}</span>"
                for kw in response.intent.keywords[:10]
            ])
            st.markdown(keyword_badges, unsafe_allow_html=True)
    
    with content_col:
        # Display referenced content (tables, images, charts)
        render_referenced_content(response)


def render_referenced_content(response: StructuredResponse):
    """Render tables, images, and charts referenced in the answer (right column)."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0; font-size: 1.1rem;">📎 Referenced Content</h3>
        <p style="color: rgba(255,255,255,0.9); margin: 0.3rem 0 0 0; font-size: 0.85rem;">
            Visual content from cited sources
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    has_content = False
    
    # Track already displayed content to avoid duplicates
    displayed_pages = set()  # Track (filename, page_num) pairs
    displayed_tables = set()  # Track table file paths
    displayed_images = set()  # Track image file paths
    
    # Collect visual content from top sources
    for idx, source in enumerate(response.sources[:5]):  # Top 5 sources
        # Handle both dict and dataclass sources
        if hasattr(source, 'metadata'):
            # Dataclass (MockSource)
            content_type = source.metadata.get('content_type', 'text')
            source_name = source.file_name
            page_num = source.page_number
        else:
            # Dict (real source)
            content_type = source.get('content_type', 'text')
            source_name = source.get('name', 'Unknown')
            page_num = source.get('page_number', 1)
        
        # Display tables
        if content_type == 'table':
            has_content = True
            with st.container():
                st.markdown(f"**📊 Table [{idx + 1}]**")
                st.caption(f"From: {source_name} (p.{page_num})")
                render_extracted_table(source, compact=True)
                st.markdown("---")
        
        # Display images
        elif content_type == 'image':
            has_content = True
            with st.container():
                st.markdown(f"**🖼️ Image [{idx + 1}]**")
                st.caption(f"From: {source_name} (p.{page_num})")
                render_extracted_image(source, compact=True)
                st.markdown("---")
        
        # Display text sources - but also check for tables/images on that page!
        elif content_type == 'text':
            # Check if this page has any visual content
            extracted_dir = get_extracted_content_dir()
            page_has_visuals = False
            
            if extracted_dir:
                # Check for tables from this page
                page_tables = list(extracted_dir.glob(f"*_page{page_num}_table*.json"))
                # Check for images from this page  
                page_images = list(extracted_dir.glob(f"*_page{page_num}_img*.png"))
                
                # Display tables from this page (skip if already displayed)
                if page_tables and idx < 5:  # Check up to 5 sources
                    table_file = str(page_tables[0])
                    if table_file not in displayed_tables:
                        displayed_tables.add(table_file)
                        has_content = True
                        page_has_visuals = True
                        with st.container():
                            st.markdown(f"**📊 Table from Page {page_num}**")
                            st.caption(f"From: {source_name}")
                            try:
                                with open(table_file) as f:
                                    table_info = json.load(f)
                                    df = pd.DataFrame(table_info['table_data'])
                                    
                                    # Display table preview
                                    if not df.empty:
                                        st.dataframe(df.head(3), use_container_width=True, height=150)
                                        if len(df) > 3:
                                            st.caption(f"+ {len(df) - 3} more rows")
                                    else:
                                        st.caption("Table is empty")
                                    
                                    # Download
                                    csv = df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 CSV",
                                        data=csv,
                                        file_name=f"table_p{page_num}.csv",
                                        mime="text/csv",
                                        key=f"dl_tbl_{hash(table_file)}"
                                    )
                            except Exception as e:
                                st.caption(f"Table preview error")
                            st.markdown("---")
                
                # Display images from this page (skip if already displayed)
                if page_images and idx < 5:  # Check up to 5 sources
                    img_file = str(page_images[0])
                    if img_file not in displayed_images:
                        displayed_images.add(img_file)
                        has_content = True
                        page_has_visuals = True
                        with st.container():
                            st.markdown(f"**🖼️ Image from Page {page_num}**")
                            st.caption(f"From: {source_name}")
                            try:
                                img = Image.open(img_file)
                                st.image(img, use_container_width=True, caption=f"Page {page_num}")
                                
                                # Download
                                buf = BytesIO()
                                img.save(buf, format='PNG')
                                st.download_button(
                                    label="📥 PNG",
                                    data=buf.getvalue(),
                                    file_name=f"image_p{page_num}.png",
                                    mime="image/png",
                                    key=f"dl_img_{hash(img_file)}"
                                )
                                
                                if len(page_images) > 1:
                                    st.caption(f"+ {len(page_images) - 1} more images on this page")
                            except Exception as e:
                                st.caption(f"Image preview error")
                            st.markdown("---")
            
            # If no visuals found but want to show text preview
            if not page_has_visuals and idx < 2:
                has_content = True
                with st.container():
                    st.markdown(f"**📄 Source [{idx + 1}]**")
                    st.caption(f"{source_name}")
                    st.caption(f"Page {page_num}")
                    
                    # Show snippet if available
                    if hasattr(source, 'content'):
                        snippet = source.content[:200] + "..." if len(source.content) > 200 else source.content
                        st.text_area(f"Preview", snippet, height=100, disabled=True, key=f"preview_{idx}")
                    
                    st.markdown("---")
    
    if not has_content:
        st.info("No visual content found in sources.\n\nThe answer is based on text documents.")


def render_sources_tab(response: StructuredResponse):
    """Render the sources tab with document references."""
    st.markdown("### 📚 Sources & References")
    
    if not response.sources:
        st.info("No sources found for this query.")
        return
    
    st.markdown(f"Found **{len(response.sources)}** relevant source(s)")
    st.markdown("---")
    
    for idx, source in enumerate(response.sources):
        # Determine icon based on content type
        # Handle both dict and dataclass sources
        if hasattr(source, 'metadata'):
            content_type = source.metadata.get('content_type', 'text')
        else:
            content_type = source.get('content_type', 'text')
        if content_type == 'table':
            icon = "📊"
            type_label = "Table"
            type_color = "#28a745"
        elif content_type == 'image':
            icon = "🖼️"
            type_label = "Image"
            type_color = "#17a2b8"
        elif content_type == 'figure_caption':
            icon = "🏷️"
            type_label = "Caption"
            type_color = "#ffc107"
        else:
            icon = "📄"
            type_label = "Text"
            type_color = "#6c757d"
        
        # Source card
        # Handle both dict and dataclass sources
        if hasattr(source, 'file_name'):
            # Dataclass (MockSource)
            source_id = f"[{idx + 1}]"
            source_name = source.file_name
            source_pages = str(source.page_number)
        else:
            # Dict (real source)
            source_id = source['id']
            source_name = source['name']
            source_pages = source['pages']
            
        with st.expander(f"{source_id} {icon} {source_name}", expanded=(idx == 0)):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Document:** {source_name}")
                if source_pages != 'N/A':
                    st.markdown(f"**Pages:** {source_pages}")
            
            with col2:
                st.markdown(f"""
                <div style="background: {type_color}20; padding: 0.5rem; border-radius: 5px; 
                            text-align: center; border: 1px solid {type_color};">
                    <strong style="color: {type_color};">{type_label}</strong>
                </div>
                """, unsafe_allow_html=True)
            
            # Show extracted content if available
            if content_type == 'table':
                render_extracted_table(source)
            elif content_type == 'image':
                render_extracted_image(source)
            
            # Feedback buttons for individual source
            st.markdown("---")
            st.markdown("**Was this source helpful?**")
            render_source_feedback_buttons(source, source_id, idx)


def render_extracted_table(source: Dict, compact=False):
    """Render extracted table data. Compact mode for side panel."""
    if not compact:
        st.markdown("#### 📊 Table Data:")
    
    # Try to load the table JSON
    try:
        # Handle both dict and dataclass sources  
        if hasattr(source, 'metadata'):
            table_json = source.metadata.get('table_json', '')
        else:
            source_path = source.get('source_path', '')
            table_json = source.get('metadata', {}).get('table_json', '')
        
        if table_json:
            # Parse and display
            table_data = json.loads(table_json)
            df = pd.DataFrame(table_data)
            
            # Compact mode: show preview
            if compact:
                st.dataframe(df.head(3), use_container_width=True, height=150)
                if len(df) > 3:
                    st.caption(f"+ {len(df) - 3} more rows")
            else:
                st.dataframe(df, use_container_width=True)
        else:
            # Try to find the extracted table file
            extracted_dir = get_extracted_content_dir()
            if extracted_dir:
                # Handle both dict and dataclass sources
                if hasattr(source, 'file_name'):
                    source_name = source.file_name
                else:
                    source_name = source.get('name', 'Unknown')
                
                page_num = source.get('page_number', 1)
                
                # Try multiple matching patterns (exact match, stem match, fuzzy match)
                source_stem = Path(source_name).stem
                
                # Pattern 1: Exact stem match
                table_files = list(extracted_dir.glob(f"{source_stem}_page{page_num}_table*.json"))
                
                # Pattern 2: If no match, try wildcard on page
                if not table_files:
                    table_files = list(extracted_dir.glob(f"*_page{page_num}_table*.json"))
                
                # Pattern 3: Get first few words of filename for partial match
                if not table_files and len(source_stem.split()) > 2:
                    first_words = "_".join(source_stem.split()[:3])
                    table_files = list(extracted_dir.glob(f"{first_words}*_page{page_num}_table*.json"))
                
                if table_files:
                    with open(table_files[0]) as f:
                        table_info = json.load(f)
                        df = pd.DataFrame(table_info['table_data'])
                        st.dataframe(df, use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name=f"{source_name}_table.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("Table data not available in extracted content.")
            else:
                st.info("Extracted content directory not found.")
    except Exception as e:
        st.warning(f"Could not display table: {e}")


def render_extracted_image(source: Dict, compact=False):
    """Render extracted image. Compact mode for side panel."""
    if not compact:
        st.markdown("#### 🖼️ Image:")
    
    try:
        # Handle both dict and dataclass sources
        if hasattr(source, 'file_name'):
            source_name = Path(source.file_name).stem
            page_num = source.page_number
        else:
            source_name = Path(source['name']).stem
            page_num = source.get('metadata', {}).get('page_number', 1)
        
        # Try to find the extracted image file
        extracted_dir = get_extracted_content_dir()
        if extracted_dir:
            # Try multiple matching patterns
            source_stem = Path(source_name).stem
            
            # Pattern 1: Exact stem match
            img_files = list(extracted_dir.glob(f"{source_stem}_page{page_num}_img*.png"))
            
            # Pattern 2: If no match, try wildcard on page (show ANY image from that page)
            if not img_files:
                img_files = list(extracted_dir.glob(f"*_page{page_num}_img*.png"))
            
            # Pattern 3: Partial filename match
            if not img_files and len(source_stem.split()) > 2:
                first_words = "_".join(source_stem.split()[:3])
                img_files = list(extracted_dir.glob(f"{first_words}*_page{page_num}_img*.png"))
            
            if img_files:
                img = Image.open(img_files[0])
                
                # Compact mode: smaller preview
                if compact:
                    st.image(img, use_container_width=True, caption=f"Page {page_num}")
                else:
                    st.image(img, use_container_width=True, caption=source.get('caption', 'Extracted image'))
                
                # Download button
                buf = BytesIO()
                img.save(buf, format='PNG')
                st.download_button(
                    label="📥 PNG" if compact else "📥 Download Image",
                    data=buf.getvalue(),
                    file_name=f"image_p{page_num}.png",
                    mime="image/png",
                    key=f"download_img_{hash(source_name)}_{page_num}"
                )
            else:
                st.caption("🖼️ Image preview not available")
        else:
            st.caption("💡 Run ingestion to extract images")
    except Exception as e:
        st.caption(f"Image preview unavailable")


def render_reasoning_tab(response: StructuredResponse):
    """Render the reasoning and retrieval details tab."""
    st.markdown("### 🧠 Reasoning & Retrieval Details")
    
    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; 
                border-left: 4px solid #17a2b8;">
        {response.reasoning}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Query analysis
    st.markdown("#### 🔍 Query Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Intent Type", response.intent.intent_type.title())
    
    with col2:
        st.metric("Intent Confidence", f"{response.intent.confidence:.0%}")
    
    with col3:
        st.metric("Keywords Found", len(response.intent.keywords))
    
    # Retrieval strategy
    if response.intent.requires_subqueries:
        st.info("🔄 This query used multiple search variations for better results")


def render_context_tab(response: StructuredResponse):
    """Render the retrieved context chunks tab."""
    st.markdown("### 📋 Retrieved Context Chunks")
    
    st.info(f"Showing context from {len(response.sources)} source document(s)")
    
    # This would show the actual retrieved chunks
    # For now, we'll show source information
    for idx, source in enumerate(response.sources):
        # Handle both dict and dataclass sources
        if hasattr(source, 'file_name'):
            # Dataclass (MockSource)
            source_name = source.file_name
            source_pages = str(source.page_number)
            content_type = source.metadata.get('content_type', 'text')
            source_content = source.content
        else:
            # Dict (real source)
            source_name = source['name']
            source_pages = source['pages']
            content_type = source.get('content_type', 'text')
            source_content = "Full chunk content would be displayed here..."
            
        with st.expander(f"Chunk {idx+1}: {source_name}", expanded=False):
            st.markdown(f"**Source:** {source_name}")
            st.markdown(f"**Pages:** {source_pages}")
            st.markdown(f"**Type:** {content_type}")
            
            # Show content if available
            if hasattr(source, 'content'):
                st.text_area("Content", source_content, height=150, disabled=True)
            else:
                st.caption(source_content)


def render_results(response: StructuredResponse):
    """
    Main function to render all results in tabs.
    
    Args:
        response: StructuredResponse from the RAG system
    """
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💡 Answer",
        "📚 Sources",
        "🧠 Reasoning",
        "📋 Context"
    ])
    
    with tab1:
        render_answer_tab(response)
    
    with tab2:
        render_sources_tab(response)
    
    with tab3:
        render_reasoning_tab(response)
    
    with tab4:
        render_context_tab(response)

