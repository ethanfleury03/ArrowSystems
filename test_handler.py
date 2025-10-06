#!/usr/bin/env python3
"""
Test script for the RunPod handler.
Run this to test the handler locally before deploying.
"""

import json
from handler import handler

def test_handler():
    """Test the handler with various inputs."""
    
    print("🧪 Testing RunPod Handler")
    print("=" * 40)
    
    # Test cases
    test_cases = [
        {
            "name": "Valid question",
            "input": {
                "input": {
                    "question": "What is the DuraFlex system?"
                }
            }
        },
        {
            "name": "Empty question",
            "input": {
                "input": {
                    "question": ""
                }
            }
        },
        {
            "name": "Missing question field",
            "input": {
                "input": {}
            }
        },
        {
            "name": "Missing input field",
            "input": {}
        },
        {
            "name": "String input (JSON)",
            "input": json.dumps({
                "input": {
                    "question": "How do I install DuraFlex?"
                }
            })
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            result = handler(test_case["input"])
            print("✅ Handler executed successfully")
            print(f"📤 Output: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"❌ Handler failed: {str(e)}")
        
        print()

if __name__ == "__main__":
    test_handler()


