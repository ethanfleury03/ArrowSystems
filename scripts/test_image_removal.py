#!/usr/bin/env python3
"""
Test script to verify image removal changes are present in the codebase.
Run this on both local and RunPod to compare results.
"""

import sys
import os
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

def test_extract_images_method():
    """Test that extract_images_from_pdf returns empty list."""
    print("=" * 70)
    print("TEST 1: extract_images_from_pdf() returns empty list")
    print("=" * 70)
    
    try:
        from backend.ingest import NonTextExtractor
        
        extractor = NonTextExtractor()
        result = extractor.extract_images_from_pdf("dummy.pdf")
        
        if result == []:
            print("[PASS] Method returns empty list")
        else:
            print(f"[FAIL] Method returned {result} (expected empty list)")
            return False
            
        if len(result) == 0:
            print("[PASS] Length is 0")
        else:
            print(f"[FAIL] Length is {len(result)} (expected 0)")
            return False
            
        return True
    except Exception as e:
        print(f"[FAIL] Error testing method: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_method_source_code():
    """Test that extract_images_from_pdf has hard-remove code."""
    print("\n" + "=" * 70)
    print("TEST 2: extract_images_from_pdf() source code contains hard-remove")
    print("=" * 70)
    
    ingest_file = repo_root / "backend" / "ingest.py"
    if not ingest_file.exists():
        print(f"❌ FAIL: File not found: {ingest_file}")
        return False
    
    with open(ingest_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for hard-remove indicators
    checks = {
        "HARD REMOVE": "HARD REMOVE comment",
        "return []": "Immediate return empty list",
        "permanently disabled": "Permanently disabled message",
    }
    
    all_passed = True
    for keyword, description in checks.items():
        if keyword in content:
            print(f"✅ PASS: Found '{description}'")
        else:
            print(f"❌ FAIL: Missing '{description}' (keyword: {keyword})")
            all_passed = False
    
    # Check that method doesn't have old extraction code
    old_code_indicators = [
        "page.get_images()",
        "fitz.Pixmap",
        "pil_image.save",
        "saved_path",
    ]
    
    # Find the method
    method_start = content.find("def extract_images_from_pdf")
    if method_start == -1:
        print("❌ FAIL: Method definition not found")
        return False
    
    # Find method end (next def or class)
    method_end = content.find("\n    def ", method_start + 1)
    if method_end == -1:
        method_end = content.find("\nclass ", method_start + 1)
    if method_end == -1:
        method_end = len(content)
    
    method_code = content[method_start:method_end]
    
    print("\nChecking for old extraction code in method...")
    for indicator in old_code_indicators:
        if indicator in method_code:
            print(f"❌ FAIL: Found old code indicator: {indicator}")
            all_passed = False
        else:
            print(f"✅ PASS: No old code indicator: {indicator}")
    
    return all_passed

def test_no_image_logs():
    """Test that there are no log statements about extracting images."""
    print("\n" + "=" * 70)
    print("TEST 3: No log statements for extracting images")
    print("=" * 70)
    
    ingest_file = repo_root / "backend" / "ingest.py"
    if not ingest_file.exists():
        print(f"❌ FAIL: File not found: {ingest_file}")
        return False
    
    with open(ingest_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Check for problematic log patterns
    problematic_patterns = [
        'Extracted.*images.*from',
        'extract_images_from_pdf.*images',
        'logger.info.*images.*from',
    ]
    
    import re
    found_issues = []
    
    for line_num, line in enumerate(lines, 1):
        for pattern in problematic_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                found_issues.append((line_num, line.strip(), pattern))
    
    if found_issues:
        print("❌ FAIL: Found problematic log statements:")
        for line_num, line, pattern in found_issues:
            print(f"   Line {line_num}: {line} (matches: {pattern})")
        return False
    else:
        print("✅ PASS: No problematic log statements found")
        return True

def test_process_non_text_content():
    """Test that process_non_text_content doesn't call image extraction."""
    print("\n" + "=" * 70)
    print("TEST 4: process_non_text_content() doesn't extract images")
    print("=" * 70)
    
    ingest_file = repo_root / "backend" / "ingest.py"
    if not ingest_file.exists():
        print(f"❌ FAIL: File not found: {ingest_file}")
        return False
    
    with open(ingest_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find process_non_text_content method
    method_start = content.find("def process_non_text_content")
    if method_start == -1:
        print("❌ FAIL: process_non_text_content method not found")
        return False
    
    # Find method end
    method_end = content.find("\n    def ", method_start + 1)
    if method_end == -1:
        method_end = content.find("\nclass ", method_start + 1)
    if method_end == -1:
        method_end = len(content)
    
    method_code = content[method_start:method_end]
    
    # Check that it doesn't call extract_images_from_pdf
    if "extract_images_from_pdf" in method_code:
        print("❌ FAIL: Method still calls extract_images_from_pdf()")
        # Show context
        lines = method_code.split('\n')
        for i, line in enumerate(lines, 1):
            if "extract_images_from_pdf" in line:
                print(f"   Line {i}: {line.strip()}")
        return False
    else:
        print("✅ PASS: Method does not call extract_images_from_pdf()")
    
    # Check that all_images is set to empty list
    if "all_images = []" in method_code:
        print("✅ PASS: all_images is initialized as empty list")
    else:
        print("⚠️  WARNING: all_images initialization not found (may be OK if using different pattern)")
    
    return True

def test_method_implementation():
    """Test the actual implementation by reading the method."""
    print("\n" + "=" * 70)
    print("TEST 5: extract_images_from_pdf() implementation details")
    print("=" * 70)
    
    ingest_file = repo_root / "backend" / "ingest.py"
    if not ingest_file.exists():
        print(f"❌ FAIL: File not found: {ingest_file}")
        return False
    
    with open(ingest_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find method
    method_start = None
    for i, line in enumerate(lines):
        if "def extract_images_from_pdf" in line:
            method_start = i
            break
    
    if method_start is None:
        print("❌ FAIL: Method definition not found")
        return False
    
    # Read method (next 30 lines should be enough)
    method_lines = lines[method_start:method_start + 30]
    method_text = ''.join(method_lines)
    
    # Check for immediate return
    if "return []" in method_text:
        return_line = None
        for i, line in enumerate(method_lines):
            if "return []" in line:
                return_line = method_start + i + 1
                break
        
        if return_line:
            # Check if return is early (within first 15 lines of method)
            if return_line - method_start <= 15:
                print(f"✅ PASS: Method returns [] early (line {return_line})")
            else:
                print(f"⚠️  WARNING: Method returns [] but not early (line {return_line})")
        else:
            print("❌ FAIL: Method does not return []")
            return False
    else:
        print("❌ FAIL: Method does not contain 'return []'")
        return False
    
    # Count lines of actual code (excluding comments/docstrings)
    code_lines = 0
    in_docstring = False
    for line in method_lines:
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if stripped and not stripped.startswith('#'):
            code_lines += 1
    
    if code_lines <= 5:  # Should be very short (just return statement and maybe a warning)
        print(f"✅ PASS: Method is minimal ({code_lines} code lines)")
    else:
        print(f"⚠️  WARNING: Method has {code_lines} code lines (expected <= 5)")
    
    return True

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("IMAGE REMOVAL VERIFICATION TEST")
    print("=" * 70)
    print(f"Repo root: {repo_root}")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    tests = [
        ("Method returns empty", test_extract_images_method),
        ("Source code check", test_method_source_code),
        ("No image logs", test_no_image_logs),
        ("process_non_text_content", test_process_non_text_content),
        ("Implementation details", test_method_implementation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ FAIL: Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - Image removal is correctly implemented")
        return 0
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED - Code may not be fully updated")
        return 1

if __name__ == "__main__":
    sys.exit(main())

