"""
DynamoDB-specific rendering functions for feedback UI
"""
import streamlit as st
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict


def render_helpful_answers_db(db):
    """Render helpful answers from DynamoDB"""
    st.markdown("### 👍 Helpful Answers")
    
    # Get all feedback
    stats = db.get_feedback_stats()
    
    if stats['helpful'] == 0:
        st.info("No helpful answers saved yet. Rate answers with 👍 to save them!")
        return
    
    st.markdown(f"**Total helpful answers:** {stats['helpful']}")
    st.markdown("---")
    
    # We'll need to scan to get helpful answers - for now show stats
    st.info("💡 Viewing individual helpful answers coming soon! For now, see Statistics tab.")


def render_unhelpful_answers_db(db):
    """Render unhelpful answers from DynamoDB"""
    st.markdown("### 👎 Unhelpful Answers")
    
    stats = db.get_feedback_stats()
    
    if stats['unhelpful'] == 0:
        st.info("No unhelpful answers recorded yet.")
        return
    
    st.markdown(f"**Total unhelpful answers:** {stats['unhelpful']}")
    st.markdown("---")
    
    st.info("💡 Use this feedback to improve your documentation!")


def render_feedback_stats_db(db):
    """Render feedback statistics from DynamoDB"""
    st.markdown("### 📊 Feedback Statistics")
    
    # Get stats
    stats = db.get_feedback_stats()
    metrics = db.get_average_metrics(days=30)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Feedback", stats['total'])
    
    with col2:
        st.metric("Helpful", stats['helpful'], 
                 delta=f"{stats['helpful_percentage']:.1f}%")
    
    with col3:
        st.metric("Unhelpful", stats['unhelpful'])
    
    with col4:
        st.metric("Total Queries", metrics['total_queries'])
    
    st.markdown("---")
    
    # More detailed stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Average Metrics (Last 30 Days)")
        st.metric("Avg Confidence", f"{metrics['avg_confidence']:.2%}")
        st.metric("Avg Response Time", f"{metrics['avg_response_time_ms']:.0f}ms")
        st.metric("Avg Intent Confidence", f"{metrics['avg_intent_confidence']:.2%}")
    
    with col2:
        st.markdown("#### Helpful Rate")
        if stats['total'] > 0:
            helpful_rate = stats['helpful_percentage']
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = helpful_rate,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps' : [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"},
                        {'range': [75, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                },
                title = {'text': "Helpful %"}
            ))
            
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feedback data yet")


def render_analytics_dashboard(db):
    """Render advanced analytics dashboard for DynamoDB data"""
    st.markdown("### 📈 Advanced Analytics")
    
    # Time range selector
    time_range = st.selectbox(
        "Time Range",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days"],
        index=1
    )
    
    days = {'Last 7 Days': 7, 'Last 30 Days': 30, 'Last 90 Days': 90}[time_range]
    
    # Get data
    intent_dist = db.get_intent_distribution(days=days)
    metrics = db.get_average_metrics(days=days)
    stats = db.get_feedback_stats()
    
    # Overview metrics
    st.markdown("#### Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", metrics['total_queries'])
    
    with col2:
        st.metric("Avg Confidence", f"{metrics['avg_confidence']:.1%}")
    
    with col3:
        st.metric("Avg Response Time", f"{metrics['avg_response_time_ms']:.0f}ms")
    
    with col4:
        helpful_rate = stats['helpful_percentage'] if stats['total'] > 0 else 0
        st.metric("Helpful Rate", f"{helpful_rate:.1%}")
    
    st.markdown("---")
    
    # Intent distribution
    if intent_dist:
        st.markdown("#### Query Intent Distribution")
        
        # Create pie chart
        fig = px.pie(
            values=list(intent_dist.values()),
            names=list(intent_dist.keys()),
            title=f"Query Types ({time_range})",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show table
        st.markdown("**Breakdown:**")
        for intent, count in sorted(intent_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / sum(intent_dist.values())) * 100
            st.markdown(f"- **{intent.title()}**: {count} queries ({percentage:.1f}%)")
    else:
        st.info("No query data for the selected time range")
    
    st.markdown("---")
    
    # Query performance insights
    st.markdown("#### Performance Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Response Time**")
        avg_time = metrics['avg_response_time_ms']
        if avg_time < 1000:
            st.success(f"✅ Excellent: {avg_time:.0f}ms")
        elif avg_time < 2000:
            st.info(f"✓ Good: {avg_time:.0f}ms")
        else:
            st.warning(f"⚠️ Slow: {avg_time:.0f}ms")
    
    with col2:
        st.markdown("**Answer Quality**")
        avg_conf = metrics['avg_confidence']
        if avg_conf > 0.8:
            st.success(f"✅ High confidence: {avg_conf:.1%}")
        elif avg_conf > 0.6:
            st.info(f"✓ Good confidence: {avg_conf:.1%}")
        else:
            st.warning(f"⚠️ Low confidence: {avg_conf:.1%}")
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("#### 💡 Recommendations")
    
    recommendations = []
    
    # Check helpful rate
    if stats['total'] >= 5:
        if helpful_rate < 0.5:
            recommendations.append("⚠️ **Low helpful rate** - Consider reviewing unhelpful answers and improving documentation")
        elif helpful_rate > 0.8:
            recommendations.append("✅ **High helpful rate** - Great job! Users are finding answers useful")
    
    # Check response time
    if avg_time > 2000:
        recommendations.append("⚠️ **Slow response times** - Consider optimizing your index or using fewer chunks")
    
    # Check confidence
    if avg_conf < 0.7 and metrics['total_queries'] >= 10:
        recommendations.append("⚠️ **Low confidence scores** - Consider adding more relevant documentation")
    
    # Check intent distribution
    if intent_dist:
        most_common = max(intent_dist, key=intent_dist.get)
        if intent_dist[most_common] / sum(intent_dist.values()) > 0.5:
            recommendations.append(f"📊 **Most queries are '{most_common}'** - Consider expanding documentation in this area")
    
    if recommendations:
        for rec in recommendations:
            st.markdown(rec)
    else:
        if metrics['total_queries'] < 5:
            st.info("📊 Not enough data yet. Ask more questions to see insights!")
        else:
            st.success("✅ Everything looks good! No issues detected.")
    
    st.markdown("---")
    
    # Export option
    st.markdown("#### 📥 Export Data")
    if st.button("Export to S3 (Future Feature)"):
        st.info("Coming soon: Export your data to S3 for advanced analytics with Athena")

