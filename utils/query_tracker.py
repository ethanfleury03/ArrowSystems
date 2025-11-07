"""
Query Tracking and Analytics

Tracks all queries for analytics and failed query monitoring.
Stores query data in JSON file for persistence.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

QUERIES_FILE = "data/query_analytics.json"
FAILED_THRESHOLD = 0.5  # Confidence threshold for failed queries


def load_queries() -> List[Dict[str, Any]]:
    """Load all queries from JSON file."""
    if not os.path.exists(QUERIES_FILE):
        return []
    
    try:
        with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading queries: {e}")
        return []


def save_queries(queries: List[Dict[str, Any]]):
    """Save queries to JSON file."""
    os.makedirs(os.path.dirname(QUERIES_FILE), exist_ok=True)
    
    try:
        with open(QUERIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving queries: {e}")
        raise


def log_query(
    query_text: str,
    session_id: str,
    answer_text: str,
    documents_retrieved: List[str],
    relevance_score: Optional[float] = None,
    confidence: Optional[float] = None,
    response_time_ms: Optional[float] = None,
    matched_machine_name: Optional[str] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    is_resolved: bool = False,
    user_feedback: Optional[str] = None
) -> str:
    """
    Log a query for analytics.
    
    Returns:
        Query ID (timestamp-based)
    """
    query_id = datetime.now().isoformat()
    
    query_data = {
        "query_id": query_id,
        "query_text": query_text,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "answer_text": answer_text[:500] if answer_text else "",  # Truncate for storage
        "documents_retrieved": documents_retrieved,
        "document_count": len(documents_retrieved),
        "relevance_score": relevance_score,
        "confidence": confidence,
        "response_time_ms": response_time_ms,
        "matched_machine_name": matched_machine_name,
        "is_resolved": is_resolved,
        "user_feedback": user_feedback,
        "sources": sources or []
    }
    
    queries = load_queries()
    queries.append(query_data)
    
    # Keep only last 10,000 queries to prevent file from growing too large
    if len(queries) > 10000:
        queries = queries[-10000:]
    
    save_queries(queries)
    
    return query_id


def get_all_queries(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    machine_type: Optional[str] = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
    limit: int = 1000,
    offset: int = 0,
    sort_by: str = "timestamp",
    sort_order: str = "desc"
) -> Dict[str, Any]:
    """
    Get all queries with filtering and sorting.
    
    Args:
        start_date: ISO format date string
        end_date: ISO format date string
        machine_type: Filter by matched machine name
        min_confidence: Minimum confidence threshold
        max_confidence: Maximum confidence threshold
        limit: Maximum number of results
        offset: Pagination offset
        sort_by: Field to sort by (timestamp, confidence, relevance_score, document_count)
        sort_order: "asc" or "desc"
    """
    queries = load_queries()
    
    # Filter queries
    filtered = []
    for query in queries:
        # Date filter
        if start_date and query.get("timestamp", "") < start_date:
            continue
        if end_date and query.get("timestamp", "") > end_date:
            continue
        
        # Machine type filter
        if machine_type and query.get("matched_machine_name") != machine_type:
            continue
        
        # Confidence filter
        confidence = query.get("confidence")
        if confidence is not None:
            if min_confidence is not None and confidence < min_confidence:
                continue
            if max_confidence is not None and confidence > max_confidence:
                continue
        
        filtered.append(query)
    
    # Sort queries
    reverse = sort_order.lower() == "desc"
    
    if sort_by == "confidence":
        filtered.sort(key=lambda x: x.get("confidence", 0), reverse=reverse)
    elif sort_by == "relevance_score":
        filtered.sort(key=lambda x: x.get("relevance_score", 0), reverse=reverse)
    elif sort_by == "document_count":
        filtered.sort(key=lambda x: x.get("document_count", 0), reverse=reverse)
    elif sort_by == "frequency":
        # Count frequency of query text
        freq_map = defaultdict(int)
        for q in filtered:
            freq_map[q.get("query_text", "")] += 1
        filtered.sort(key=lambda x: freq_map.get(x.get("query_text", ""), 0), reverse=reverse)
    else:  # timestamp (default)
        filtered.sort(key=lambda x: x.get("timestamp", ""), reverse=reverse)
    
    # Pagination
    total = len(filtered)
    paginated = filtered[offset:offset + limit]
    
    return {
        "queries": paginated,
        "total": total,
        "limit": limit,
        "offset": offset
    }


def get_failed_queries(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    machine_type: Optional[str] = None,
    include_resolved: bool = False,
    limit: int = 1000,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Get failed queries (low confidence or no documents retrieved).
    
    Args:
        start_date: ISO format date string
        end_date: ISO format date string
        machine_type: Filter by matched machine name
        include_resolved: Include resolved queries
        limit: Maximum number of results
        offset: Pagination offset
    """
    queries = load_queries()
    
    failed = []
    for query in queries:
        # Check if failed
        confidence = query.get("confidence", 1.0)
        doc_count = query.get("document_count", 0)
        is_resolved = query.get("is_resolved", False)
        
        # Failed if: low confidence OR no documents retrieved
        is_failed = confidence < FAILED_THRESHOLD or doc_count == 0
        
        if not is_failed:
            continue
        
        # Skip resolved if not including them
        if is_resolved and not include_resolved:
            continue
        
        # Date filter
        if start_date and query.get("timestamp", "") < start_date:
            continue
        if end_date and query.get("timestamp", "") > end_date:
            continue
        
        # Machine type filter
        if machine_type and query.get("matched_machine_name") != machine_type:
            continue
        
        failed.append(query)
    
    # Sort by timestamp (most recent first)
    failed.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Pagination
    total = len(failed)
    paginated = failed[offset:offset + limit]
    
    return {
        "queries": paginated,
        "total": total,
        "limit": limit,
        "offset": offset
    }


def mark_query_resolved(query_id: str) -> bool:
    """Mark a query as resolved."""
    queries = load_queries()
    
    for query in queries:
        if query.get("query_id") == query_id:
            query["is_resolved"] = True
            query["resolved_at"] = datetime.now().isoformat()
            save_queries(queries)
            return True
    
    return False


def get_query_stats() -> Dict[str, Any]:
    """Get aggregate statistics about queries."""
    queries = load_queries()
    
    if not queries:
        return {
            "total_queries": 0,
            "failed_queries": 0,
            "resolved_queries": 0,
            "avg_confidence": 0.0,
            "avg_response_time_ms": 0.0,
            "top_machines": [],
            "top_failed_queries": []
        }
    
    total = len(queries)
    failed = 0
    resolved = 0
    confidences = []
    response_times = []
    machine_counts = defaultdict(int)
    query_freq = defaultdict(int)
    
    for query in queries:
        confidence = query.get("confidence")
        if confidence is not None:
            confidences.append(confidence)
            if confidence < FAILED_THRESHOLD:
                failed += 1
        
        if query.get("is_resolved", False):
            resolved += 1
        
        response_time = query.get("response_time_ms")
        if response_time is not None:
            response_times.append(response_time)
        
        machine = query.get("matched_machine_name")
        if machine:
            machine_counts[machine] += 1
        
        query_text = query.get("query_text", "")
        if query.get("confidence", 1.0) < FAILED_THRESHOLD:
            query_freq[query_text] += 1
    
    # Top failed queries
    top_failed = sorted(query_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Top machines
    top_machines = sorted(machine_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "total_queries": total,
        "failed_queries": failed,
        "resolved_queries": resolved,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else 0.0,
        "top_machines": [{"machine": k, "count": v} for k, v in top_machines],
        "top_failed_queries": [{"query": k, "count": v} for k, v in top_failed]
    }

