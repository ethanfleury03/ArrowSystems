#!/usr/bin/env python3
"""
Setup script for the RAG Streamlit app.
Run this to install dependencies and verify your setup.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Setting up RAG Streamlit App")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Please run this script from the rag_app.py directory")
        sys.exit(1)
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("💡 Try running: pip install --upgrade pip")
        sys.exit(1)
    
    # Check if ingest.py has been run
    if not os.path.exists("storage"):
        print("📝 Running ingest.py to create the index...")
        if not run_command("python ingest.py", "Creating vector index"):
            print("❌ Failed to create index. Please check your data directory.")
            sys.exit(1)
    else:
        print("✅ Vector index already exists!")
    
    print("\n🎉 Setup complete!")
    print("\nTo run the app:")
    print("  streamlit run app.py")
    print("\nThe app will open in your browser at http://localhost:8501")

if __name__ == "__main__":
    main()
