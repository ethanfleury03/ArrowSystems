#!/usr/bin/env python3
"""
Migrate existing JSON feedback data to DynamoDB
"""

import json
import os
from datetime import datetime
from utils.dynamodb_manager import DynamoDBManager

def migrate_feedback():
    """Migrate saved_answers.json to DynamoDB."""
    
    print("=" * 60)
    print("Migrating JSON Feedback to DynamoDB")
    print("=" * 60)
    print()
    
    # Check if JSON file exists
    json_file = "logs/saved_answers.json"
    if not os.path.exists(json_file):
        print(f"⚠️  No feedback file found at {json_file}")
        print("Nothing to migrate.")
        return
    
    # Load JSON data
    try:
        with open(json_file, 'r') as f:
            feedback_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
        return
    
    if not feedback_data:
        print("⚠️  Feedback file is empty.")
        print("Nothing to migrate.")
        return
    
    print(f"Found {len(feedback_data)} feedback entries to migrate")
    print()
    
    # Initialize DynamoDB
    try:
        db = DynamoDBManager(local_mode=True)
    except Exception as e:
        print(f"❌ Error connecting to DynamoDB: {e}")
        print()
        print("Make sure DynamoDB Local is running:")
        print("  docker-compose -f docker-compose.dynamodb.yml up -d")
        return
    
    # Migrate each entry
    migrated = 0
    errors = 0
    
    for entry in feedback_data:
        try:
            # Extract data
            user = entry.get('user', 'Unknown')
            query = entry.get('query', '')
            answer = entry.get('answer', '')
            is_helpful = entry.get('is_helpful', False)
            confidence = entry.get('confidence', 0.0)
            intent_type = entry.get('intent_type', 'unknown')
            sources = entry.get('sources', [])
            timestamp = entry.get('timestamp', datetime.utcnow().isoformat())
            
            # Save query to DynamoDB
            query_id = db.save_query(
                user=user,
                query_text=query,
                answer_text=answer,
                intent_type=intent_type,
                intent_confidence=0.8,  # Default since old data doesn't have this
                sources=sources,
                confidence=confidence,
                response_time_ms=0,  # Unknown for old data
                session_id='migrated'
            )
            
            # Save feedback
            db.save_feedback(
                query_id=query_id,
                user=user,
                is_helpful=is_helpful,
                feedback_text=None
            )
            
            migrated += 1
            print(f"✅ Migrated: {query[:50]}...")
            
        except Exception as e:
            errors += 1
            print(f"❌ Error migrating entry: {e}")
    
    print()
    print("=" * 60)
    print("Migration Complete")
    print("=" * 60)
    print(f"Migrated: {migrated}")
    print(f"Errors: {errors}")
    print()
    
    if migrated > 0:
        # Backup original file
        backup_file = f"{json_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            os.rename(json_file, backup_file)
            print(f"✅ Original file backed up to: {backup_file}")
        except:
            print(f"⚠️  Could not backup original file")
    
    print()
    print("Next steps:")
    print("  1. Verify data in DynamoDB:")
    print("     python -c \"from utils.dynamodb_manager import DynamoDBManager; db = DynamoDBManager(); print(db.get_feedback_stats())\"")
    print("  2. Test the application with DynamoDB")
    print()


if __name__ == "__main__":
    migrate_feedback()

