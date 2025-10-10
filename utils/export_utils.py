"""
Export Utilities
Handles exporting query results to various formats (PDF, Excel, Word)
"""

import streamlit as st
from io import BytesIO
from datetime import datetime
from typing import Dict, Any
import pandas as pd
from orchestrator import StructuredResponse


def export_to_excel(response: StructuredResponse, query: str) -> BytesIO:
    """
    Export query results to Excel format.
    
    Args:
        response: StructuredResponse object
        query: Original query string
    
    Returns:
        BytesIO object containing Excel file
    """
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = {
            'Query': [query],
            'Answer': [response.answer],
            'Confidence': [f"{response.confidence:.0%}"],
            'Intent Type': [response.intent.intent_type],
            'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sources sheet
        if response.sources:
            sources_data = []
            for idx, source in enumerate(response.sources):
                # Handle both dict and dataclass sources
                if hasattr(source, 'file_name'):
                    # Dataclass (MockSource)
                    sources_data.append({
                        'ID': f"[{idx + 1}]",
                        'Document': source.file_name,
                        'Pages': str(source.page_number),
                        'Content Type': source.metadata.get('content_type', 'text')
                    })
                else:
                    # Dict (real source)
                    sources_data.append({
                        'ID': source['id'],
                        'Document': source['name'],
                        'Pages': source['pages'],
                        'Content Type': source.get('content_type', 'text')
                    })
            sources_df = pd.DataFrame(sources_data)
            sources_df.to_excel(writer, sheet_name='Sources', index=False)
        
        # Keywords sheet
        if response.intent.keywords:
            keywords_df = pd.DataFrame({'Keywords': response.intent.keywords})
            keywords_df.to_excel(writer, sheet_name='Keywords', index=False)
    
    output.seek(0)
    return output


def export_to_text(response: StructuredResponse, query: str) -> str:
    """
    Export query results to plain text format.
    
    Args:
        response: StructuredResponse object
        query: Original query string
    
    Returns:
        Formatted text string
    """
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("DURAFLEX TECHNICAL ASSISTANT - QUERY REPORT")
    output_lines.append("=" * 80)
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("")
    
    output_lines.append("-" * 80)
    output_lines.append("QUERY:")
    output_lines.append("-" * 80)
    output_lines.append(query)
    output_lines.append("")
    
    output_lines.append("-" * 80)
    output_lines.append("ANSWER:")
    output_lines.append("-" * 80)
    output_lines.append(response.answer)
    output_lines.append("")
    
    output_lines.append("-" * 80)
    output_lines.append("METADATA:")
    output_lines.append("-" * 80)
    output_lines.append(f"Confidence: {response.confidence:.0%}")
    output_lines.append(f"Intent Type: {response.intent.intent_type}")
    output_lines.append(f"Keywords: {', '.join(response.intent.keywords)}")
    output_lines.append("")
    
    output_lines.append("-" * 80)
    output_lines.append("SOURCES:")
    output_lines.append("-" * 80)
    for idx, source in enumerate(response.sources):
        # Handle both dict and dataclass sources
        if hasattr(source, 'file_name'):
            # Dataclass (MockSource)
            source_line = f"[{idx + 1}] {source.file_name} (Pages: {source.page_number})"
        else:
            # Dict (real source)
            source_line = f"{source['id']} {source['name']} (Pages: {source['pages']})"
        output_lines.append(source_line)
    output_lines.append("")
    
    output_lines.append("-" * 80)
    output_lines.append("REASONING:")
    output_lines.append("-" * 80)
    output_lines.append(response.reasoning)
    output_lines.append("")
    
    output_lines.append("=" * 80)
    output_lines.append("End of Report")
    output_lines.append("=" * 80)
    
    return "\n".join(output_lines)


def render_export_options(response: StructuredResponse, query: str):
    """
    Render export buttons for various formats.
    
    Args:
        response: StructuredResponse object
        query: Original query string
    """
    st.markdown("### 📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export to Excel
        excel_data = export_to_excel(response, query)
        st.download_button(
            label="📊 Download Excel",
            data=excel_data,
            file_name=f"duraflex_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col2:
        # Export to Text
        text_data = export_to_text(response, query)
        st.download_button(
            label="📄 Download Text",
            data=text_data,
            file_name=f"duraflex_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        # Copy to clipboard (using text format)
        st.button(
            "📋 Copy to Clipboard",
            use_container_width=True,
            help="Click to copy results as text",
            disabled=True  # Will be enabled with custom JS
        )
    
    st.markdown("---")

