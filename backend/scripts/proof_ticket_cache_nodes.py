#!/usr/bin/env python3
"""
Proof script: Count ticket_cache nodes in production index.

This script can run locally and downloads the index from GCS (same as Cloud Run does),
or can point at a local index directory after promotion.

Usage:
    # Download from GCS and check (matches prod behavior)
    python backend/scripts/proof_ticket_cache_nodes.py

    # Check local index directory
    python backend/scripts/proof_ticket_cache_nodes.py --index-dir ./latest_model

    # Download from specific GCS bucket/prefix
    python backend/scripts/proof_ticket_cache_nodes.py --bucket arrow-rag-support-prod-rag --prefix latest_model/
"""

import argparse
import os
import sys
from pathlib import Path

# Add repo root to path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from llama_index.core import load_index_from_storage, StorageContext
from backend.config.env import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Count ticket_cache nodes in RAG index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from GCS and check (production index)
  python backend/scripts/proof_ticket_cache_nodes.py

  # Check local index directory
  python backend/scripts/proof_ticket_cache_nodes.py --index-dir ./latest_model

  # Use specific GCS bucket/prefix
  python backend/scripts/proof_ticket_cache_nodes.py --bucket arrow-rag-support-prod-rag --prefix latest_model/
        """
    )
    parser.add_argument(
        "--index-dir",
        help="Local index directory (if not provided, downloads from GCS)"
    )
    parser.add_argument(
        "--bucket",
        help="GCS bucket (default: RAG_INDEX_GCS_BUCKET env var or arrow-rag-support-prod-rag)"
    )
    parser.add_argument(
        "--prefix",
        help="GCS prefix (default: RAG_INDEX_GCS_PREFIX env var or latest_model/)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of ticket_cache node IDs to sample (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Determine index directory
    if args.index_dir:
        index_dir = Path(args.index_dir).resolve()
        if not index_dir.exists():
            print(f"❌ Index directory not found: {index_dir}", file=sys.stderr)
            return 1
        if not (index_dir / "docstore.json").exists():
            print(f"❌ Index directory does not contain docstore.json: {index_dir}", file=sys.stderr)
            return 1
        print(f"📁 Using local index directory: {index_dir}")
    else:
        # Download from GCS (matches production behavior)
        bucket = args.bucket or os.getenv("RAG_INDEX_GCS_BUCKET") or "arrow-rag-support-prod-rag"
        prefix = args.prefix or os.getenv("RAG_INDEX_GCS_PREFIX") or "latest_model/"
        
        # Normalize prefix
        prefix = prefix.rstrip("/")
        if prefix:
            prefix = f"{prefix}/"
        
        print(f"📥 Downloading index from GCS: gs://{bucket}/{prefix}")
        
        # Set env vars for downloader
        os.environ["RAG_INDEX_GCS_BUCKET"] = bucket
        os.environ["RAG_INDEX_GCS_PREFIX"] = prefix
        os.environ["RAG_INDEX_LOCAL_DIR"] = "/tmp/proof_index"
        
        from backend.rag.startup_downloader import download_index_from_gcs
        
        if not download_index_from_gcs():
            from backend.rag.startup_downloader import get_last_download_error
            error = get_last_download_error() or "Unknown error"
            print(f"❌ Failed to download index from GCS: {error}", file=sys.stderr)
            return 1
        
        index_dir = Path("/tmp/proof_index")
        print(f"✅ Index downloaded to: {index_dir}")
    
    # Load index
    print(f"\n📖 Loading index from {index_dir}...")
    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
        index = load_index_from_storage(storage_context)
        print("✅ Index loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load index: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    # Count nodes
    print(f"\n🔍 Analyzing nodes...")
    docstore = index.storage_context.docstore
    
    total_nodes = 0
    ticket_cache_nodes = []
    ticket_cache_ids = []
    
    try:
        # Iterate through all nodes in docstore
        for node_id in docstore.docs.keys():
            try:
                node = docstore.get_document(node_id)
                total_nodes += 1
                
                # Extract metadata
                metadata = {}
                if hasattr(node, 'metadata') and node.metadata:
                    metadata = node.metadata
                elif hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                    metadata = node.node.metadata or {}
                
                # Check content_type
                content_type = metadata.get('content_type', '')
                if content_type == 'ticket_cache':
                    ticket_cache_nodes.append(node)
                    ticket_id = metadata.get('ticket_id', 'unknown')
                    ticket_cache_ids.append((node_id, ticket_id))
                    
            except Exception as e:
                logger.debug(f"Failed to process node {node_id}: {e}")
                continue
        
        # Print results
        print(f"\n📊 Results:")
        print(f"   Total nodes: {total_nodes:,}")
        print(f"   Ticket cache nodes: {len(ticket_cache_nodes):,}")
        
        if total_nodes > 0:
            pct = (len(ticket_cache_nodes) / total_nodes) * 100.0
            print(f"   Percentage: {pct:.2f}%")
        
        if ticket_cache_ids:
            print(f"\n📋 Sample ticket_cache node IDs (first {min(args.sample_size, len(ticket_cache_ids))}):")
            for node_id, ticket_id in ticket_cache_ids[:args.sample_size]:
                print(f"   - node_id: {node_id}, ticket_id: {ticket_id}")
        else:
            print(f"\n⚠️  No ticket_cache nodes found in index")
            print(f"   This means ticket artifacts have not been ingested yet.")
            print(f"   Run the promotion workflow to ingest ticket cache artifacts.")
            return 1
        
        print(f"\n✅ Proof complete: Index contains {len(ticket_cache_nodes)} ticket_cache nodes")
        return 0
        
    except Exception as e:
        print(f"❌ Failed to analyze nodes: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
