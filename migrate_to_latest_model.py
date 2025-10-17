#!/usr/bin/env python3
"""
Migration script: storage/ → latest_model/

Helps transition from old storage directory to new latest_model/ directory
for the two-pod workflow.
"""

import os
import shutil
import sys

def migrate_storage():
    """Migrate storage/ to latest_model/ directory."""
    
    old_paths = ["storage", "/workspace/storage"]
    new_path = "latest_model"
    
    print("="*70)
    print("🔄 MIGRATION: storage/ → latest_model/")
    print("="*70)
    print()
    
    # Find existing storage
    existing_storage = None
    for path in old_paths:
        if os.path.exists(path):
            existing_storage = path
            break
    
    if not existing_storage:
        print("ℹ️  No existing storage/ directory found.")
        print("   Nothing to migrate. You're good to go!")
        print()
        print("   Run: python ingest.py")
        print("   This will create a fresh latest_model/ directory.")
        return
    
    print(f"📁 Found existing storage at: {existing_storage}")
    print()
    
    # Check if latest_model already exists
    if os.path.exists(new_path):
        print(f"⚠️  {new_path}/ already exists!")
        response = input("   Overwrite it? (y/n): ").strip().lower()
        if response != 'y':
            print("❌ Migration cancelled.")
            return
        print(f"🗑️  Removing existing {new_path}/...")
        shutil.rmtree(new_path)
    
    # Copy storage to latest_model
    print(f"📦 Copying {existing_storage}/ → {new_path}/...")
    shutil.copytree(existing_storage, new_path)
    
    print()
    print("="*70)
    print("✅ MIGRATION COMPLETE!")
    print("="*70)
    print()
    print(f"✅ Index copied to: {new_path}/")
    print()
    print("📋 Next steps:")
    print("   1. Test the app: streamlit run app.py")
    print("   2. If it works, commit the new index:")
    print(f"      git add {new_path}/")
    print("      git commit -m 'Migrate to latest_model/ for two-pod workflow'")
    print("      git push")
    print()
    print(f"   3. Optional: Delete old {existing_storage}/ (it's in .gitignore)")
    print(f"      rm -rf {existing_storage}/")
    print()

if __name__ == "__main__":
    try:
        migrate_storage()
    except KeyboardInterrupt:
        print("\n❌ Migration cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)

