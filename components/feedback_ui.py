"""
Feedback UI Components
Thumbs up/down buttons and saved answers display
"""
import streamlit as st
from typing import Optional
from utils.feedback_manager import FeedbackManager
from orchestrator import RAGOrchestrator  # for cache access via session rag_system

def render_feedback_buttons(response, query: str, user: str = "Unknown"):
    """
    Render thumbs up/down feedback buttons for a response.
    
    Args:
        response: The RAG response object
        query: The original query
        user: Current username
    """
    feedback_manager = FeedbackManager()
    
    # Extract source names
    source_names = []
    for source in response.sources:
        if hasattr(source, 'file_name'):
            source_names.append(source.file_name)
        else:
            source_names.append(source.get('name', 'Unknown'))
    
    # Check if already rated
    existing_rating = feedback_manager.check_if_rated(query, response.answer)
    
    # Create columns for buttons
    col1, col2, col3 = st.columns([1, 1, 8])
    
    with col1:
        # Thumbs up button
        if existing_rating == True:
            st.button("👍", disabled=True, key=f"thumbs_up_{hash(query)}", 
                     help="Already marked as helpful")
        else:
            if st.button("👍", key=f"thumbs_up_{hash(query)}", 
                        help="Mark as helpful"):
                # Save positive feedback
                success = feedback_manager.save_feedback(
                    query=query,
                    answer=response.answer,
                    is_helpful=True,
                    confidence=response.confidence,
                    intent_type=response.intent.intent_type,
                    sources=source_names,
                    user=user
                )
                if success:
                    # Also cache the validated response for instant future answers
                    try:
                        if 'rag_system' in st.session_state and hasattr(st.session_state['rag_system'], 'orchestrator'):
                            rag = st.session_state['rag_system'].orchestrator
                            # Retrieve latest query params if present
                            top_k = st.session_state.get('last_top_k', 10)
                            alpha = st.session_state.get('last_alpha', 0.5)
                            # Exact cache
                            rag.cache.set(query, response, top_k=top_k, alpha=alpha)
                            # Semantic cache
                            if getattr(rag, 'semantic_cache', None) is not None:
                                rag.semantic_cache.set(query, response)
                    except Exception:
                        # Non-fatal; caching is best-effort
                        pass
                    st.success("✅ Saved to helpful answers!", icon="✅")
                    st.rerun()
    
    with col2:
        # Thumbs down button
        if existing_rating == False:
            st.button("👎", disabled=True, key=f"thumbs_down_{hash(query)}", 
                     help="Already marked as unhelpful")
        else:
            if st.button("👎", key=f"thumbs_down_{hash(query)}", 
                        help="Mark as unhelpful"):
                # Save negative feedback
                success = feedback_manager.save_feedback(
                    query=query,
                    answer=response.answer,
                    is_helpful=False,
                    confidence=response.confidence,
                    intent_type=response.intent.intent_type,
                    sources=source_names,
                    user=user
                )
                if success:
                    # Ensure any cached version is removed
                    try:
                        if 'rag_system' in st.session_state and hasattr(st.session_state['rag_system'], 'orchestrator'):
                            rag = st.session_state['rag_system'].orchestrator
                            top_k = st.session_state.get('last_top_k', 10)
                            alpha = st.session_state.get('last_alpha', 0.5)
                            rag.cache.remove(query, top_k=top_k, alpha=alpha)
                            if getattr(rag, 'semantic_cache', None) is not None:
                                rag.semantic_cache.remove(query)
                    except Exception:
                        pass
                    st.warning("📝 Marked as unhelpful", icon="📝")
                    st.rerun()
    
    with col3:
        # Show status if rated
        if existing_rating == True:
            st.markdown("✅ **Marked as helpful**")
        elif existing_rating == False:
            st.markdown("⚠️ **Marked as unhelpful**")


def render_saved_answers_page():
    """Render the saved answers page"""
    st.title("💾 Saved Answers")
    
    feedback_manager = FeedbackManager()
    
    # Tabs for helpful/unhelpful
    tab1, tab2, tab3 = st.tabs(["👍 Helpful Answers", "👎 Unhelpful Answers", "📊 Statistics"])
    
    with tab1:
        render_helpful_answers(feedback_manager)
    
    with tab2:
        render_unhelpful_answers(feedback_manager)
    
    with tab3:
        render_feedback_stats(feedback_manager)


def render_helpful_answers(feedback_manager: FeedbackManager):
    """Render helpful answers list"""
    helpful = feedback_manager.get_helpful_answers()
    
    if not helpful:
        st.info("No helpful answers saved yet. Mark answers as helpful using the 👍 button!")
        return
    
    st.success(f"Found {len(helpful)} helpful answer(s)")
    
    # Export button
    if st.button("📥 Export to CSV"):
        csv_data = feedback_manager.export_feedback_csv()
        st.download_button(
            "Download CSV",
            csv_data,
            "helpful_answers.csv",
            "text/csv",
            key='download-csv-helpful'
        )
    
    st.markdown("---")
    
    # Display each helpful answer
    for i, entry in enumerate(reversed(helpful)):
        with st.expander(f"🔍 {entry['query']}", expanded=(i == 0)):
            # Metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{entry.get('confidence', 0):.0%}")
            with col2:
                st.metric("Intent", entry.get('intent_type', 'N/A'))
            with col3:
                st.caption(f"Saved: {entry['timestamp'][:10]}")
            
            st.markdown("---")
            
            # Answer
            st.markdown("**Answer:**")
            st.markdown(entry['answer'])
            
            # Sources
            if entry.get('sources'):
                st.markdown("**Sources:**")
                for source in entry['sources']:
                    st.caption(f"📄 {source}")
            
            # Delete button
            if st.button("🗑️ Delete", key=f"delete_helpful_{entry['id']}"):
                if feedback_manager.delete_feedback(entry['id']):
                    st.success("Deleted!")
                    st.rerun()


def render_unhelpful_answers(feedback_manager: FeedbackManager):
    """Render unhelpful answers list"""
    unhelpful = feedback_manager.get_unhelpful_answers()
    
    if not unhelpful:
        st.info("No unhelpful answers marked yet.")
        return
    
    st.warning(f"Found {len(unhelpful)} unhelpful answer(s)")
    st.caption("These answers can help identify areas for improvement.")
    
    st.markdown("---")
    
    # Display each unhelpful answer
    for i, entry in enumerate(reversed(unhelpful)):
        with st.expander(f"❓ {entry['query']}", expanded=(i == 0)):
            # Metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{entry.get('confidence', 0):.0%}")
            with col2:
                st.metric("Intent", entry.get('intent_type', 'N/A'))
            with col3:
                st.caption(f"Saved: {entry['timestamp'][:10]}")
            
            st.markdown("---")
            
            # Answer
            st.markdown("**Answer:**")
            st.markdown(entry['answer'])
            
            # Sources
            if entry.get('sources'):
                st.markdown("**Sources:**")
                for source in entry['sources']:
                    st.caption(f"📄 {source}")
            
            # Delete button
            if st.button("🗑️ Delete", key=f"delete_unhelpful_{entry['id']}"):
                if feedback_manager.delete_feedback(entry['id']):
                    st.success("Deleted!")
                    st.rerun()


def render_feedback_stats(feedback_manager: FeedbackManager):
    """Render feedback statistics"""
    st.subheader("📊 Feedback Statistics")
    
    stats = feedback_manager.get_feedback_stats()
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Feedback", stats['total'])
    
    with col2:
        st.metric("Helpful", stats['helpful'], 
                 delta=f"{stats['helpful_percentage']:.1f}%")
    
    with col3:
        st.metric("Unhelpful", stats['unhelpful'])
    
    # Chart
    if stats['total'] > 0:
        import pandas as pd
        
        chart_data = pd.DataFrame({
            'Feedback Type': ['Helpful', 'Unhelpful'],
            'Count': [stats['helpful'], stats['unhelpful']]
        })
        
        st.bar_chart(chart_data.set_index('Feedback Type'))
    
    st.markdown("---")
    
    # Export all feedback
    if st.button("📥 Export All Feedback to CSV"):
        csv_data = feedback_manager.export_feedback_csv()
        st.download_button(
            "Download All Feedback CSV",
            csv_data,
            "all_feedback.csv",
            "text/csv",
            key='download-csv-all'
        )
    
    # Additional insights
    st.markdown("### 💡 Insights")
    
    all_feedback = feedback_manager.load_all_feedback()
    
    if all_feedback:
        # Intent type breakdown
        intent_counts = {}
        for entry in all_feedback:
            intent = entry.get('intent_type', 'Unknown')
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        st.markdown("**Most Common Query Types:**")
        for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
            st.caption(f"• {intent}: {count} queries")
        
        # Average confidence
        confidences = [e.get('confidence', 0) for e in all_feedback if e.get('is_helpful')]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            st.markdown(f"**Average Confidence (Helpful Answers):** {avg_confidence:.1%}")

