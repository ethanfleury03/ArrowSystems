"""
Standalone tool to promote local index to GCS.

Safely backs up existing index and uploads new one.
Requires explicit --promote flag to prevent accidental uploads.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.ingest import promote_index_to_gcs, verify_local_index_artifact


def main():
    parser = argparse.ArgumentParser(
        description="Promote local index to GCS (safe backup + upload)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify only (no upload)
  python -m backend.tools.promote_index_to_gcs --index-dir ./latest_model --verify-only

  # Promote to GCS (requires --promote flag)
  python -m backend.tools.promote_index_to_gcs \\
    --index-dir ./latest_model \\
    --promote \\
    --bucket arrow-rag-support-prod-rag \\
    --prefix latest_model/
        """
    )
    parser.add_argument("--index-dir", required=True, help="Local index directory")
    parser.add_argument("--bucket", help="GCS bucket (default: RAG_INDEX_GCS_BUCKET env var)")
    parser.add_argument("--prefix", help="GCS prefix (default: RAG_INDEX_GCS_PREFIX env var)")
    parser.add_argument("--old-prefix", help="GCS prefix for backups (default: GCS_RAG_OLD_PREFIX or old_model/)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify, don't upload")
    parser.add_argument("--promote", action="store_true", help="REQUIRED: Explicitly enable promotion (safety check)")
    
    args = parser.parse_args()
    
    index_dir = Path(args.index_dir).resolve()
    if not index_dir.exists():
        print(f"❌ Index directory not found: {index_dir}", file=sys.stderr)
        return 1
    
    # Verify index
    print("🔍 Verifying index...")
    try:
        verification = verify_local_index_artifact(index_dir)
        print(f"✅ Index verified:")
        print(f"   - Index dir: {verification['index_dir']}")
        print(f"   - Num nodes: {verification['num_nodes']}")
        print(f"   - Num chunks: {verification['num_chunks']}")
    except Exception as e:
        print(f"❌ Index verification failed: {e}", file=sys.stderr)
        return 1
    
    if args.verify_only:
        print("\n✅ Verification complete (--verify-only, no upload)")
        return 0
    
    # Require explicit --promote flag
    if not args.promote:
        print("\n❌ Promotion requires explicit --promote flag (safety check)", file=sys.stderr)
        print("   Add --promote to enable upload.", file=sys.stderr)
        return 1
    
    # Get GCS config
    bucket = args.bucket or os.getenv("RAG_INDEX_GCS_BUCKET") or os.getenv("GCS_RAG_BUCKET") or "arrow-rag-support-prod-rag"
    prefix = args.prefix or os.getenv("RAG_INDEX_GCS_PREFIX") or os.getenv("GCS_RAG_LATEST_PREFIX") or "latest_model/"
    old_prefix = args.old_prefix or os.getenv("GCS_RAG_OLD_PREFIX") or "old_model/"
    
    print(f"\n📤 Promoting index:")
    print(f"   - Local: {index_dir}")
    print(f"   - Bucket: {bucket}")
    print(f"   - Prefix: {prefix}")
    print(f"   - Backup prefix: {old_prefix}")
    
    # Promote
    try:
        result = promote_index_to_gcs(
            local_index_dir=index_dir,
            rag_bucket=bucket,
            latest_prefix=prefix,
            old_prefix=old_prefix,
        )
        
        print(f"\n✅ Promotion complete:")
        print(f"   - Backup: gs://{bucket}/{result['backup_prefix']}")
        print(f"   - Uploaded: {result['uploaded']} objects")
        print(f"   - Latest objects: {result['latest_objects']}")
        return 0
    except Exception as e:
        print(f"\n❌ Promotion failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

