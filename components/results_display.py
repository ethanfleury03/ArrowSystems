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
        # Display answer with nice formatting
        formatted_answer = response.answer.replace('\n', '<br>')
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05); border-left: 4px solid #667eea;">
            {formatted_answer}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Feedback buttons
        st.markdown("**Was this answer helpful?**")
        from components.feedback_ui import render_feedback_buttons
        render_feedback_buttons(response, st.session_state.get('feedback_query', ''), 
                               st.session_state.get('username', 'Unknown'))
        
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
        
        # Display text sources with page preview link
        elif content_type == 'text':
            # Only show first 2 text sources to avoid clutter
            if idx < 2:
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
            extracted_dir = Path("extracted_content")
            if extracted_dir.exists():
                # Look for matching table file
                source_name = Path(source['name']).stem
                page_num = source.get('page_number', 1)
                table_files = list(extracted_dir.glob(f"{source_name}_page{page_num}_table*.json"))
                
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
        extracted_dir = Path("extracted_content")
        if extracted_dir.exists():
            img_files = list(extracted_dir.glob(f"{source_name}_page{page_num}_img*.png"))
            
            if img_files:
                img = Image.open(img_files[0])
                
                # Compact mode: smaller preview
                if compact:
                    st.image(img, use_column_width=True, caption=f"Page {page_num}")
                else:
                    st.image(img, use_column_width=True, caption=source.get('caption', 'Extracted image'))
                
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

