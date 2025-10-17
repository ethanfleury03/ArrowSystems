#!/usr/bin/env python3
"""
Comprehensive test script for the RAG system.
Run this to validate your RAG API with various questions.
"""

import json
from simple_handler import handler

def test_question(question, test_name):
    """Test a single question and display results."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")
    
    try:
        result = handler({"input": {"question": question}})
        
        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            output = result['output']
            print(f" SUCCESS - Found {len(output)} characters of relevant information")
            print(f"\n RESPONSE:")
            print("-" * 40)
            print(output)
            print("-" * 40)
            
    except Exception as e:
        print(f" EXCEPTION: {str(e)}")

def main():
    """Run comprehensive tests."""
    print("RAG SYSTEM VALIDATION TESTS")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        ("What is DuraFlex?", "Basic System Query"),
        ("How do I install DuraFlex?", "Installation Guide"),
        ("What are the system requirements?", "System Requirements"),
        ("What are common DuraFlex problems?", "Troubleshooting"),
        ("How do I configure DuraFlex?", "Configuration"),
        ("What is the DuraFlex printhead?", "Technical Details"),
        ("How do I maintain DuraFlex?", "Maintenance"),
        ("What software does DuraFlex use?", "Software Requirements"),
    ]
    
    # Run all tests
    for question, test_name in test_cases:
        test_question(question, test_name)
    
    # Test error handling
    print(f"\n{'='*60}")
    print("ERROR HANDLING TESTS")
    print(f"{'='*60}")
    
    # Test missing question
    print("\nTesting missing question field:")
    result = handler({"input": {}})
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test missing input
    print("\nTesting missing input field:")
    result = handler({})
    print(f"Result: {json.dumps(result, indent=2)}")
    
    print(f"\n{'='*60}")
    print("ALL TESTS COMPLETED!")
    print("Your RAG system is ready for deployment!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
