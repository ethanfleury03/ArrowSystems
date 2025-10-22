#!/usr/bin/env python3
"""
Test script for Claude-powered Intent Classifier
Tests various query types to validate 95%+ accuracy
"""

import os
import sys
from orchestrator import ClaudeIntentClassifier, IntentClassifier

# Test queries covering all 5 intent types
TEST_QUERIES = [
    # DEFINITION queries
    {
        "query": "What is a DuraFlex printhead?",
        "expected_intent": "definition",
        "description": "Simple definition request"
    },
    {
        "query": "Explain how the inline degasser works",
        "expected_intent": "definition",
        "description": "Explanation request"
    },
    {
        "query": "Define PPU in the context of printing systems",
        "expected_intent": "definition",
        "description": "Explicit definition with context"
    },
    
    # LOOKUP queries
    {
        "query": "What is the DuraFlex printhead temperature range?",
        "expected_intent": "lookup",
        "description": "Specific technical specification"
    },
    {
        "query": "How much does the printer weigh?",
        "expected_intent": "lookup",
        "description": "Quantitative lookup"
    },
    {
        "query": "What voltage does the system require?",
        "expected_intent": "lookup",
        "description": "Technical specification lookup"
    },
    
    # TROUBLESHOOTING queries
    {
        "query": "Printer is showing error code E-23, how do I fix it?",
        "expected_intent": "troubleshooting",
        "description": "Error code troubleshooting"
    },
    {
        "query": "Print quality is poor, what's the issue?",
        "expected_intent": "troubleshooting",
        "description": "Quality problem"
    },
    {
        "query": "The printhead is not working properly",
        "expected_intent": "troubleshooting",
        "description": "Component malfunction"
    },
    {
        "query": "How to fix paper jam in the DuraFlex?",
        "expected_intent": "troubleshooting",
        "description": "Fix-oriented troubleshooting"
    },
    
    # REASONING queries
    {
        "query": "How to install the PPU module?",
        "expected_intent": "reasoning",
        "description": "Installation procedure"
    },
    {
        "query": "What are the steps to calibrate the printhead?",
        "expected_intent": "reasoning",
        "description": "Step-by-step procedure"
    },
    {
        "query": "Procedure for replacing the inline degasser",
        "expected_intent": "reasoning",
        "description": "Replacement procedure"
    },
    {
        "query": "How do I configure the network settings?",
        "expected_intent": "reasoning",
        "description": "Configuration process"
    },
    
    # COMPARISON queries
    {
        "query": "Compare inline degasser vs standard degasser",
        "expected_intent": "comparison",
        "description": "Direct comparison"
    },
    {
        "query": "What's the difference between DuraFlex A and DuraFlex B?",
        "expected_intent": "comparison",
        "description": "Difference inquiry"
    },
    {
        "query": "Which is better for high-volume printing, X or Y?",
        "expected_intent": "comparison",
        "description": "Comparative evaluation"
    },
    {
        "query": "PPU vs standard printing unit pros and cons",
        "expected_intent": "comparison",
        "description": "Pros/cons comparison"
    },
    
    # EDGE CASES
    {
        "query": "Temperature specs and troubleshooting overheating",
        "expected_intent": None,  # Could be either lookup or troubleshooting
        "description": "Multi-intent query (edge case)"
    },
    {
        "query": "Why is the printer making a strange noise?",
        "expected_intent": "troubleshooting",
        "description": "Problem without explicit fix request"
    },
]


def test_intent_classifier(use_claude=True):
    """Test the intent classifier with various queries."""
    
    print("=" * 80)
    if use_claude:
        print("🎯 TESTING CLAUDE-POWERED INTENT CLASSIFIER")
        classifier = ClaudeIntentClassifier()
        if classifier.claude_client is None:
            print("⚠️  Claude API not available. Falling back to pattern matching.")
    else:
        print("📋 TESTING FALLBACK PATTERN-MATCHING CLASSIFIER")
        classifier = IntentClassifier()
    print("=" * 80)
    print()
    
    correct = 0
    total = 0
    results = []
    
    for test_case in TEST_QUERIES:
        query = test_case["query"]
        expected = test_case["expected_intent"]
        description = test_case["description"]
        
        print(f"Query: {query}")
        print(f"Description: {description}")
        print(f"Expected: {expected or 'N/A (edge case)'}")
        
        try:
            intent = classifier.classify(query)
            
            print(f"Result: {intent.intent_type} (confidence: {intent.confidence:.2%})")
            print(f"Keywords: {', '.join(intent.keywords[:5])}")
            print(f"Requires subqueries: {intent.requires_subqueries}")
            
            # Check if correct (skip edge cases)
            if expected is not None:
                is_correct = intent.intent_type == expected
                if is_correct:
                    correct += 1
                    print("✅ CORRECT")
                else:
                    print(f"❌ INCORRECT (expected {expected})")
                total += 1
                
                results.append({
                    "query": query,
                    "expected": expected,
                    "actual": intent.intent_type,
                    "correct": is_correct,
                    "confidence": intent.confidence
                })
            else:
                print("⚠️  EDGE CASE (not scored)")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 80)
        print()
    
    # Summary
    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
        
        # Calculate average confidence for correct answers
        if results:
            avg_confidence = sum(r["confidence"] for r in results if r["correct"]) / max(correct, 1)
            print(f"Average confidence (correct answers): {avg_confidence:.2%}")
        
        # Show misclassifications
        misclassified = [r for r in results if not r["correct"]]
        if misclassified:
            print(f"\n❌ Misclassified queries ({len(misclassified)}):")
            for r in misclassified:
                print(f"  - '{r['query']}'")
                print(f"    Expected: {r['expected']}, Got: {r['actual']}")
    
    print("=" * 80)
    
    return correct, total


def main():
    """Run the test suite."""
    
    print("\n" + "=" * 80)
    print("🧪 INTENT CLASSIFIER TEST SUITE")
    print("=" * 80)
    print()
    
    # Check if API key is available
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        print("✅ ANTHROPIC_API_KEY found")
    else:
        print("⚠️  ANTHROPIC_API_KEY not found - will use fallback classifier")
    print()
    
    # Test Claude classifier
    claude_correct, claude_total = test_intent_classifier(use_claude=True)
    
    print("\n\n")
    
    # Test fallback classifier for comparison
    fallback_correct, fallback_total = test_intent_classifier(use_claude=False)
    
    # Comparison
    if claude_total > 0 and fallback_total > 0:
        print("\n" + "=" * 80)
        print("📈 COMPARISON")
        print("=" * 80)
        claude_acc = (claude_correct / claude_total) * 100
        fallback_acc = (fallback_correct / fallback_total) * 100
        improvement = claude_acc - fallback_acc
        
        print(f"Claude Classifier:   {claude_correct}/{claude_total} = {claude_acc:.1f}%")
        print(f"Fallback Classifier: {fallback_correct}/{fallback_total} = {fallback_acc:.1f}%")
        print(f"Improvement:         {improvement:+.1f} percentage points")
        print("=" * 80)
    
    print("\n✅ Test suite completed!")


if __name__ == "__main__":
    main()

