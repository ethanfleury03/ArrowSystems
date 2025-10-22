#!/usr/bin/env python3
"""
Setup script for DynamoDB tables
Run this once to initialize tables locally or on AWS
"""

import sys
import argparse
from utils.dynamodb_manager import DynamoDBManager

def main():
    parser = argparse.ArgumentParser(description='Setup DynamoDB tables for RAG application')
    parser.add_argument('--aws', action='store_true', 
                       help='Setup tables on AWS (default: local)')
    parser.add_argument('--list', action='store_true',
                       help='List existing tables')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DynamoDB Setup for RAG Application")
    print("=" * 60)
    print()
    
    # Initialize manager
    local_mode = not args.aws
    mode_str = "Local DynamoDB" if local_mode else "AWS DynamoDB"
    
    print(f"Mode: {mode_str}")
    print()
    
    try:
        db = DynamoDBManager(local_mode=local_mode)
        
        if args.list:
            # List tables
            print("Existing tables:")
            tables = list(db.dynamodb.tables.all())
            if tables:
                for table in tables:
                    print(f"  - {table.name}")
            else:
                print("  (No tables found)")
            print()
            return
        
        # Create tables
        print("Creating tables...")
        print()
        
        success = db.create_tables()
        
        if success:
            print()
            print("=" * 60)
            print("✅ Setup Complete!")
            print("=" * 60)
            print()
            print("Tables created:")
            print(f"  - {db.query_table_name}")
            print(f"  - {db.feedback_table_name}")
            print(f"  - {db.validated_qna_table_name}")
            print()
            
            if local_mode:
                print("Next steps:")
                print("  1. Verify DynamoDB is running: docker-compose -f docker-compose.dynamodb.yml ps")
                print("  2. Start your RAG application: ./start.sh")
            else:
                print("Next steps:")
                print("  1. Verify tables in AWS Console")
                print("  2. Set AWS credentials in your environment")
                print("  3. Deploy your application")
            print()
        else:
            print()
            print("❌ Setup failed. Check logs for details.")
            sys.exit(1)
            
    except Exception as e:
        print()
        print(f"❌ Error: {e}")
        print()
        
        if local_mode:
            print("Troubleshooting:")
            print("  1. Is DynamoDB Local running?")
            print("     Run: docker-compose -f docker-compose.dynamodb.yml up -d")
            print("  2. Check if port 8000 is available")
            print("  3. Try: curl http://localhost:8000/")
        else:
            print("Troubleshooting:")
            print("  1. Are AWS credentials configured?")
            print("  2. Do you have DynamoDB permissions?")
            print("  3. Check AWS_REGION environment variable")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()

