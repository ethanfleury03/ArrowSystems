"""
DynamoDB Manager for RAG Application
Handles all database operations for query history, feedback, and analytics
"""

import os
import boto3
from boto3.dynamodb.conditions import Key, Attr
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import json
from decimal import Decimal

logger = logging.getLogger(__name__)


class DecimalEncoder(json.JSONEncoder):
    """Helper to convert Decimal to int/float for JSON serialization"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        return super(DecimalEncoder, self).default(obj)


class DynamoDBManager:
    """
    Manages all DynamoDB operations for the RAG application.
    
    Tables:
    - QueryHistory: Stores all user queries and answers
    - Feedback: Stores user feedback (thumbs up/down)
    - ValidatedQnA: Stores validated Q&A pairs for caching
    """
    
    def __init__(self, local_mode: bool = None):
        """
        Initialize DynamoDB manager.
        
        Args:
            local_mode: If True, use local DynamoDB. If None, auto-detect.
        """
        # Auto-detect local vs AWS
        if local_mode is None:
            local_mode = not bool(os.getenv('AWS_EXECUTION_ENV'))
        
        self.local_mode = local_mode
        self.dynamodb = self._initialize_dynamodb()
        
        # Table names
        self.query_table_name = 'RAG_QueryHistory'
        self.feedback_table_name = 'RAG_Feedback'
        self.validated_qna_table_name = 'RAG_ValidatedQnA'
        
        # Get table references
        self.query_table = self.dynamodb.Table(self.query_table_name)
        self.feedback_table = self.dynamodb.Table(self.feedback_table_name)
        self.validated_qna_table = self.dynamodb.Table(self.validated_qna_table_name)
        
        logger.info(f"DynamoDB Manager initialized ({'local' if local_mode else 'AWS'} mode)")
    
    def _initialize_dynamodb(self):
        """Initialize DynamoDB resource."""
        if self.local_mode:
            # Local DynamoDB
            logger.info("Using local DynamoDB (endpoint: http://localhost:8000)")
            return boto3.resource(
                'dynamodb',
                endpoint_url='http://localhost:8000',
                region_name='us-east-1',
                aws_access_key_id='dummy',
                aws_secret_access_key='dummy'
            )
        else:
            # AWS DynamoDB
            region = os.getenv('AWS_REGION', 'us-east-1')
            logger.info(f"Using AWS DynamoDB (region: {region})")
            return boto3.resource('dynamodb', region_name=region)
    
    def create_tables(self):
        """
        Create all required DynamoDB tables.
        Should only be run once during initial setup.
        """
        try:
            # Table 1: QueryHistory
            # Build GSI configuration based on mode
            gsi_config = {
                'IndexName': 'DateIndex',
                'KeySchema': [
                    {'AttributeName': 'GSI1PK', 'KeyType': 'HASH'},
                    {'AttributeName': 'GSI1SK', 'KeyType': 'RANGE'}
                ],
                'Projection': {'ProjectionType': 'ALL'}
            }
            if self.local_mode:
                gsi_config['ProvisionedThroughput'] = {
                    'ReadCapacityUnits': 1,
                    'WriteCapacityUnits': 1
                }
            
            # Build table configuration
            table_config = {
                'TableName': self.query_table_name,
                'KeySchema': [
                    {'AttributeName': 'PK', 'KeyType': 'HASH'},
                    {'AttributeName': 'SK', 'KeyType': 'RANGE'}
                ],
                'AttributeDefinitions': [
                    {'AttributeName': 'PK', 'AttributeType': 'S'},
                    {'AttributeName': 'SK', 'AttributeType': 'S'},
                    {'AttributeName': 'GSI1PK', 'AttributeType': 'S'},
                    {'AttributeName': 'GSI1SK', 'AttributeType': 'S'}
                ],
                'GlobalSecondaryIndexes': [gsi_config],
                'BillingMode': 'PROVISIONED' if self.local_mode else 'PAY_PER_REQUEST'
            }
            if self.local_mode:
                table_config['ProvisionedThroughput'] = {
                    'ReadCapacityUnits': 1,
                    'WriteCapacityUnits': 1
                }
            
            self.dynamodb.create_table(**table_config)
            logger.info(f"Created table: {self.query_table_name}")
            
            # Table 2: Feedback
            table2_config = {
                'TableName': self.feedback_table_name,
                'KeySchema': [
                    {'AttributeName': 'PK', 'KeyType': 'HASH'},
                    {'AttributeName': 'SK', 'KeyType': 'RANGE'}
                ],
                'AttributeDefinitions': [
                    {'AttributeName': 'PK', 'AttributeType': 'S'},
                    {'AttributeName': 'SK', 'AttributeType': 'S'}
                ],
                'BillingMode': 'PROVISIONED' if self.local_mode else 'PAY_PER_REQUEST'
            }
            if self.local_mode:
                table2_config['ProvisionedThroughput'] = {
                    'ReadCapacityUnits': 1,
                    'WriteCapacityUnits': 1
                }
            
            self.dynamodb.create_table(**table2_config)
            logger.info(f"Created table: {self.feedback_table_name}")
            
            # Table 3: ValidatedQnA
            table3_config = {
                'TableName': self.validated_qna_table_name,
                'KeySchema': [
                    {'AttributeName': 'query_hash', 'KeyType': 'HASH'}
                ],
                'AttributeDefinitions': [
                    {'AttributeName': 'query_hash', 'AttributeType': 'S'}
                ],
                'BillingMode': 'PROVISIONED' if self.local_mode else 'PAY_PER_REQUEST'
            }
            if self.local_mode:
                table3_config['ProvisionedThroughput'] = {
                    'ReadCapacityUnits': 1,
                    'WriteCapacityUnits': 1
                }
            
            self.dynamodb.create_table(**table3_config)
            logger.info(f"Created table: {self.validated_qna_table_name}")
            
            logger.info("✅ All tables created successfully!")
            return True
            
        except Exception as e:
            if 'ResourceInUseException' in str(e):
                logger.info("Tables already exist")
                return True
            else:
                logger.error(f"Error creating tables: {e}")
                return False
    
    # ==================== Query History Operations ====================
    
    def save_query(
        self,
        user: str,
        query_text: str,
        answer_text: str,
        intent_type: str,
        intent_confidence: float,
        sources: List[str],
        confidence: float,
        response_time_ms: int,
        session_id: str = None
    ) -> str:
        """
        Save a query and its answer to the database.
        
        Returns:
            query_id: Unique identifier for this query
        """
        timestamp = datetime.utcnow().isoformat()
        query_id = f"{user}_{timestamp}"
        
        item = {
            'PK': f'USER#{user}',
            'SK': f'QUERY#{timestamp}',
            'GSI1PK': f'DATE#{timestamp[:10]}',  # For querying by date
            'GSI1SK': timestamp,
            'query_id': query_id,
            'query_text': query_text,
            'answer_text': answer_text,
            'intent_type': intent_type,
            'intent_confidence': Decimal(str(intent_confidence)),
            'sources': sources,
            'confidence': Decimal(str(confidence)),
            'response_time_ms': response_time_ms,
            'session_id': session_id or 'unknown',
            'timestamp': timestamp
        }
        
        try:
            self.query_table.put_item(Item=item)
            logger.debug(f"Saved query for user {user}")
            return query_id
        except Exception as e:
            logger.error(f"Error saving query: {e}")
            raise
    
    def get_user_query_history(
        self,
        user: str,
        limit: int = 20,
        start_date: datetime = None
    ) -> List[Dict[str, Any]]:
        """
        Get query history for a specific user.
        
        Args:
            user: Username
            limit: Maximum number of queries to return
            start_date: Optional filter for queries after this date
            
        Returns:
            List of query dictionaries
        """
        try:
            if start_date:
                response = self.query_table.query(
                    KeyConditionExpression=Key('PK').eq(f'USER#{user}') & 
                                         Key('SK').gte(f'QUERY#{start_date.isoformat()}'),
                    ScanIndexForward=False,  # Most recent first
                    Limit=limit
                )
            else:
                response = self.query_table.query(
                    KeyConditionExpression=Key('PK').eq(f'USER#{user}'),
                    ScanIndexForward=False,
                    Limit=limit
                )
            
            items = response.get('Items', [])
            
            # Convert Decimal to float for JSON serialization
            return json.loads(json.dumps(items, cls=DecimalEncoder))
            
        except Exception as e:
            logger.error(f"Error getting user history: {e}")
            return []
    
    def get_query_by_id(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific query by its ID."""
        try:
            # Parse query_id (format: username_timestamp)
            user, timestamp = query_id.rsplit('_', 1)
            
            response = self.query_table.get_item(
                Key={
                    'PK': f'USER#{user}',
                    'SK': f'QUERY#{timestamp}'
                }
            )
            
            item = response.get('Item')
            if item:
                return json.loads(json.dumps(item, cls=DecimalEncoder))
            return None
            
        except Exception as e:
            logger.error(f"Error getting query by ID: {e}")
            return None
    
    # ==================== Feedback Operations ====================
    
    def save_feedback(
        self,
        query_id: str,
        user: str,
        is_helpful: bool,
        feedback_text: str = None
    ) -> bool:
        """
        Save user feedback for a query-answer pair.
        
        Args:
            query_id: ID of the query being rated
            user: Username providing feedback
            is_helpful: True for thumbs up, False for thumbs down
            feedback_text: Optional text feedback
            
        Returns:
            bool: True if saved successfully
        """
        timestamp = datetime.utcnow().isoformat()
        
        item = {
            'PK': f'QUERY#{query_id}',
            'SK': f'FEEDBACK#{timestamp}',
            'user': user,
            'is_helpful': is_helpful,
            'feedback_text': feedback_text or '',
            'timestamp': timestamp
        }
        
        try:
            self.feedback_table.put_item(Item=item)
            logger.debug(f"Saved feedback for query {query_id}")
            
            # If helpful, consider adding to ValidatedQnA
            if is_helpful:
                self._update_validated_qna(query_id, is_helpful=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            return False
    
    def get_query_feedback(self, query_id: str) -> List[Dict[str, Any]]:
        """Get all feedback for a specific query."""
        try:
            response = self.feedback_table.query(
                KeyConditionExpression=Key('PK').eq(f'QUERY#{query_id}')
            )
            return response.get('Items', [])
        except Exception as e:
            logger.error(f"Error getting feedback: {e}")
            return []
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get overall feedback statistics."""
        try:
            # Scan all feedback (expensive, but OK for small datasets)
            response = self.feedback_table.scan()
            items = response.get('Items', [])
            
            helpful = sum(1 for item in items if item.get('is_helpful'))
            unhelpful = len(items) - helpful
            
            return {
                'total': len(items),
                'helpful': helpful,
                'unhelpful': unhelpful,
                'helpful_percentage': (helpful / len(items) * 100) if items else 0
            }
        except Exception as e:
            logger.error(f"Error getting feedback stats: {e}")
            return {'total': 0, 'helpful': 0, 'unhelpful': 0, 'helpful_percentage': 0}
    
    # ==================== Validated Q&A Operations ====================
    
    def _update_validated_qna(self, query_id: str, is_helpful: bool):
        """
        Internal method to update validated Q&A based on feedback.
        
        Strategy: First validated answer wins (never overwrite).
        - First thumbs up: Saves the answer as canonical
        - Subsequent thumbs up: Only increment helpful_count
        - This prevents Claude's answer variations from overwriting good answers
        """
        try:
            # Get the original query
            query = self.get_query_by_id(query_id)
            if not query:
                return
            
            # Create hash of query for deduplication
            import hashlib
            query_hash = hashlib.md5(query['query_text'].lower().encode()).hexdigest()
            
            # Update or create validated QnA entry
            # Use if_not_exists() to preserve the first validated answer
            self.validated_qna_table.update_item(
                Key={'query_hash': query_hash},
                UpdateExpression='SET query_text = if_not_exists(query_text, :query), '
                               'answer_text = if_not_exists(answer_text, :answer), '
                               'sources = if_not_exists(sources, :sources), '
                               'helpful_count = if_not_exists(helpful_count, :zero) + :inc, '
                               'last_used = :timestamp, is_active = :active, '
                               'first_validated = if_not_exists(first_validated, :timestamp)',
                ExpressionAttributeValues={
                    ':query': query['query_text'],
                    ':answer': query['answer_text'],
                    ':sources': query['sources'],
                    ':zero': 0,
                    ':inc': 1 if is_helpful else 0,
                    ':timestamp': datetime.utcnow().isoformat(),
                    ':active': True
                }
            )
            logger.info(f"✅ Updated ValidatedQnA for query (helpful_count +1, answer preserved)")
        except Exception as e:
            logger.error(f"Error updating validated QnA: {e}")
    
    def get_validated_answer(self, query_text: str) -> Optional[Dict[str, Any]]:
        """
        Get a validated answer for a similar query (if exists).
        Uses exact hash match for now.
        """
        try:
            import hashlib
            query_hash = hashlib.md5(query_text.lower().encode()).hexdigest()
            
            response = self.validated_qna_table.get_item(
                Key={'query_hash': query_hash}
            )
            
            item = response.get('Item')
            if item and item.get('is_active') and item.get('helpful_count', 0) > 0:
                return json.loads(json.dumps(item, cls=DecimalEncoder))
            return None
            
        except Exception as e:
            logger.error(f"Error getting validated answer: {e}")
            return None
    
    # ==================== Analytics Operations ====================
    
    def get_queries_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get all queries within a date range."""
        try:
            # Query using the DateIndex GSI
            response = self.query_table.query(
                IndexName='DateIndex',
                KeyConditionExpression=Key('GSI1PK').eq(f'DATE#{start_date.strftime("%Y-%m-%d")}')
            )
            
            items = response.get('Items', [])
            return json.loads(json.dumps(items, cls=DecimalEncoder))
            
        except Exception as e:
            logger.error(f"Error getting queries by date: {e}")
            return []
    
    def get_intent_distribution(self, days: int = 30) -> Dict[str, int]:
        """Get distribution of intent types over the last N days."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Scan with filter (OK for small datasets)
            response = self.query_table.scan(
                FilterExpression=Attr('timestamp').gte(start_date.isoformat())
            )
            
            items = response.get('Items', [])
            
            # Count by intent type
            distribution = {}
            for item in items:
                intent = item.get('intent_type', 'unknown')
                distribution[intent] = distribution.get(intent, 0) + 1
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error getting intent distribution: {e}")
            return {}
    
    def get_average_metrics(self, days: int = 30) -> Dict[str, float]:
        """Get average confidence and response time metrics."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            response = self.query_table.scan(
                FilterExpression=Attr('timestamp').gte(start_date.isoformat())
            )
            
            items = response.get('Items', [])
            
            if not items:
                return {
                    'avg_confidence': 0.0,
                    'avg_response_time_ms': 0.0,
                    'avg_intent_confidence': 0.0,
                    'total_queries': 0
                }
            
            total_confidence = sum(float(item.get('confidence', 0)) for item in items)
            total_response_time = sum(item.get('response_time_ms', 0) for item in items)
            total_intent_confidence = sum(float(item.get('intent_confidence', 0)) for item in items)
            
            return {
                'avg_confidence': total_confidence / len(items),
                'avg_response_time_ms': total_response_time / len(items),
                'avg_intent_confidence': total_intent_confidence / len(items),
                'total_queries': len(items)
            }
            
        except Exception as e:
            logger.error(f"Error getting average metrics: {e}")
            return {
                'avg_confidence': 0.0,
                'avg_response_time_ms': 0.0,
                'avg_intent_confidence': 0.0,
                'total_queries': 0
            }

