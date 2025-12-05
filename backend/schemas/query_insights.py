"""
Pydantic schemas for Query Insights admin endpoints.
These match the frontend TypeScript types in frontend/types/queryInsights.ts
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class QueryInsightsCustomer(BaseModel):
    id: str
    name: str
    total_queries: int
    last_query_at: Optional[datetime] = None


class CustomerQuerySummary(BaseModel):
    id: str
    conversation_id: str
    created_at: datetime
    query_text: str
    message_count: Optional[int] = None


class CustomerQueriesResponse(BaseModel):
    customer_id: str
    customer_name: str
    total_queries: int
    last_query_at: Optional[datetime] = None
    queries: List[CustomerQuerySummary]


class ConversationMessage(BaseModel):
    id: str
    role: str  # "user" or "assistant"
    content: str
    created_at: datetime


class ConversationDetails(BaseModel):
    conversation_id: str
    customer_id: str
    customer_name: str
    created_at: datetime
    messages: List[ConversationMessage]

