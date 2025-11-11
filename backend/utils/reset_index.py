"""
Reset Index Utility
Safely deletes vector index and extracted content to allow fresh ingestion
"""
import os
import shutil
from pathlib import Path
import sys


def reset_index(confirm=True, verbose=True):
    """
    Delete vector index and extracted content.
    
    Args:
        confirm: If True, ask for confirmation before deleting
        verbose: If True, print detailed progress
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Determine project root (go up from utils/ to project root)
    script_dir = Path(__file__).parent        # backend/utils
    project_root = script_dir.parent.parent   # repository root
    
    # Change to project root for consistent path resolution
    original_dir = os.getcwd()
    os.chdir(project_root)
    
    if verbose:
        print(f"📂 Working from: {project_root}")
    
    # Paths to delete (relative to project root)
    paths_to_delete = [
        "storage",
        "latest_model",  # Current default storage location
        "extracted_content",
        ".qdrant",  # If using Qdrant local
    ]
    
    # Add absolute workspace paths if on RunPod/cloud
    if os.path.exists("/workspace"):
        paths_to_delete.extend([
            "/workspace/storage",
            "/workspace/latest_model",
        ])
    
    deleted_items = []
    
    if verbose:
        print("\n" + "="*70)
        print("🗑️  INDEX RESET UTILITY")
        print("="*70)
        print("\nThis will delete:")
        
        for path in paths_to_delete:
            if os.path.exists(path):
                size = get_directory_size(path)
                print(f"   • {path}/ ({size})")
        
        print("\n⚠️  WARNING: This action cannot be undone!")
        print("   You will need to run 'python -m backend.ingest' again to rebuild.")
        print("")
    
    # Confirmation
    if confirm:
        response = input("Are you sure you want to delete the index? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("\n❌ Reset cancelled")
            return False
    
    # Delete each path
    if verbose:
        print("\n🔄 Deleting index and extracted content...")
        print("")
    
    for path in paths_to_delete:
        if os.path.exists(path):
            try:
                if verbose:
                    print(f"   Deleting: {path}/")
                
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                
                deleted_items.append(path)
                
                if verbose:
                    print(f"   ✅ Deleted: {path}/")
            
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Error deleting {path}: {e}")
        else:
            if verbose:
                print(f"   ⏭️  Skipped: {path}/ (doesn't exist)")
    
    # Restore original directory
    os.chdir(original_dir)
    
    if verbose:
        print("")
        print("="*70)
        if deleted_items:
            print(f"✅ RESET COMPLETE")
            print("="*70)
            print(f"📊 Deleted {len(deleted_items)} item(s):")
            for item in deleted_items:
                print(f"   • {item}/")
            print("")
            print("📝 Next steps:")
            print("   1. Run: python -m backend.ingest")
            print("   2. Wait for ingestion to complete (~10-30 minutes)")
            print("   3. Start app: streamlit run app.py")
        else:
            print("ℹ️  NO FILES TO DELETE")
            print("="*70)
            print("Index was already clean or never created.")
        print("="*70 + "\n")
    
    return True


def get_directory_size(path):
    """Get human-readable directory size."""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        
        # Convert to human readable
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024.0:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024.0
        
        return f"{total_size:.1f} TB"
    
    except Exception as e:
        return "unknown size"


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Reset RAG index and extracted content',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils/reset_index.py              # Interactive mode (asks for confirmation)
  python utils/reset_index.py --force      # Skip confirmation
  python utils/reset_index.py --quiet      # Minimal output
  python utils/reset_index.py --force --quiet  # Silent deletion

After resetting:
  python -m backend.ingest                 # Rebuild the index
        """
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )
    
    args = parser.parse_args()
    
    # Run reset
    success = reset_index(
        confirm=not args.force,
        verbose=not args.quiet
    )
    
    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

