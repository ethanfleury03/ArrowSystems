"""
Feedback Manager for RAG Responses
Handles saving, loading, and managing user feedback on answers
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

class FeedbackManager:
    """Manages user feedback on RAG responses"""
    
    def __init__(self, feedback_file: str = "saved_answers.json"):
        self.feedback_file = feedback_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create feedback file if it doesn't exist"""
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w') as f:
                json.dump([], f)
    
    def save_feedback(
        self, 
        query: str, 
        answer: str, 
        is_helpful: bool,
        confidence: float = 0.0,
        intent_type: str = "",
        sources: List[str] = None,
        user: str = "Unknown"
    ) -> bool:
        """
        Save user feedback for a query-answer pair.
        
        Args:
            query: The user's question
            answer: The RAG system's answer
            is_helpful: True for thumbs up, False for thumbs down
            confidence: Response confidence score
            intent_type: Query intent classification
            sources: List of source document names
            user: Username who provided feedback
            
        Returns:
            bool: True if saved successfully
        """
        try:
            feedback_entry = {
                'id': self._generate_id(),
                'timestamp': datetime.now().isoformat(),
                'user': user,
                'query': query,
                'answer': answer,
                'is_helpful': is_helpful,
                'confidence': confidence,
                'intent_type': intent_type,
                'sources': sources or [],
                'feedback_type': 'positive' if is_helpful else 'negative'
            }
            
            # Load existing feedback
            feedback_list = self.load_all_feedback()
            
            # Check if this query-answer pair already has feedback
            existing_index = self._find_existing_feedback(feedback_list, query, answer)
            
            if existing_index is not None:
                # Update existing feedback
                feedback_list[existing_index] = feedback_entry
            else:
                # Add new feedback
                feedback_list.append(feedback_entry)
            
            # Save to file
            with open(self.feedback_file, 'w') as f:
                json.dump(feedback_list, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False
    
    def _generate_id(self) -> str:
        """Generate unique ID for feedback entry"""
        return datetime.now().strftime("%Y%m%d%H%M%S%f")
    
    def _find_existing_feedback(
        self, 
        feedback_list: List[Dict], 
        query: str, 
        answer: str
    ) -> Optional[int]:
        """Find if feedback already exists for this query-answer pair"""
        for i, entry in enumerate(feedback_list):
            if entry['query'] == query and entry['answer'][:100] == answer[:100]:
                return i
        return None
    
    def load_all_feedback(self) -> List[Dict[str, Any]]:
        """Load all feedback entries"""
        try:
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading feedback: {e}")
            return []
    
    def get_helpful_answers(self) -> List[Dict[str, Any]]:
        """Get all answers marked as helpful"""
        all_feedback = self.load_all_feedback()
        return [f for f in all_feedback if f['is_helpful']]
    
    def get_unhelpful_answers(self) -> List[Dict[str, Any]]:
        """Get all answers marked as unhelpful"""
        all_feedback = self.load_all_feedback()
        return [f for f in all_feedback if not f['is_helpful']]
    
    def check_if_rated(self, query: str, answer: str) -> Optional[bool]:
        """
        Check if this query-answer pair has been rated.
        
        Returns:
            None if not rated, True if helpful, False if unhelpful
        """
        all_feedback = self.load_all_feedback()
        for entry in all_feedback:
            if entry['query'] == query and entry['answer'][:100] == answer[:100]:
                return entry['is_helpful']
        return None
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about feedback"""
        all_feedback = self.load_all_feedback()
        helpful = len([f for f in all_feedback if f['is_helpful']])
        unhelpful = len([f for f in all_feedback if not f['is_helpful']])
        
        return {
            'total': len(all_feedback),
            'helpful': helpful,
            'unhelpful': unhelpful,
            'helpful_percentage': (helpful / len(all_feedback) * 100) if all_feedback else 0
        }
    
    def delete_feedback(self, feedback_id: str) -> bool:
        """Delete a feedback entry by ID"""
        try:
            feedback_list = self.load_all_feedback()
            feedback_list = [f for f in feedback_list if f['id'] != feedback_id]
            
            with open(self.feedback_file, 'w') as f:
                json.dump(feedback_list, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error deleting feedback: {e}")
            return False
    
    def export_feedback_csv(self) -> str:
        """Export feedback to CSV format"""
        import csv
        from io import StringIO
        
        output = StringIO()
        all_feedback = self.load_all_feedback()
        
        if not all_feedback:
            return ""
        
        fieldnames = ['timestamp', 'user', 'query', 'answer', 'is_helpful', 
                     'confidence', 'intent_type', 'sources']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        writer.writeheader()
        for entry in all_feedback:
            row = {k: entry.get(k, '') for k in fieldnames}
            row['sources'] = ', '.join(entry.get('sources', []))
            writer.writerow(row)
        
        return output.getvalue()

