#!/usr/bin/env python3
"""
Test each import individually to find which one is failing
Run this in your RunPod terminal: python test_imports.py
"""

import sys
import traceback

def test_import(module_name, import_statement):
    """Test a single import and report results"""
    try:
        exec(import_statement)
        print(f"✅ {module_name}")
        return True
    except Exception as e:
        print(f"❌ {module_name}")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False

print("=" * 80)
print("Testing imports from app.py")
print("=" * 80)
print()

# Test basic imports
test_import("streamlit", "import streamlit as st")
test_import("sys", "import sys")
test_import("warnings", "import warnings")
test_import("pathlib", "from pathlib import Path")
test_import("logging", "import logging")
test_import("datetime", "from datetime import datetime")

print()
print("Testing component imports...")
test_import("components.auth", "from components.auth import AuthManager, render_login_page, render_user_profile_sidebar")
test_import("components.query_interface", "from components.query_interface import render_query_controls, render_query_input, render_query_history, add_to_query_history")
test_import("components.results_display", "from components.results_display import render_results")

print()
print("Testing utils imports...")
test_import("utils.session_manager", "from utils.session_manager import init_session_state, increment_query_count, get_session_stats")
test_import("utils.export_utils", "from utils.export_utils import render_export_options")

print()
print("Testing RAG system imports...")
test_import("query.EliteRAGQuery", "from query import EliteRAGQuery")

print()
print("=" * 80)
print("Import test complete!")
print("=" * 80)

