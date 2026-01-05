#!/usr/bin/env python3
"""
Smoke test to validate no image ingestion and RAG doesn't expect images.

This script:
1. Builds a fresh index from test documents
2. Inspects docstore.json for node counts by content_type
3. Hard-fails if any node has content_type == "image"
4. Runs a query and hard-fails if retrieval returns any image nodes

Usage:
    python scripts/smoke_no_images.py <test_docs_dir> [--config config.yaml] [--persist-dir /tmp/test_index] [--query "your query"]

Example:
    python scripts/smoke_no_images.py test_docs/ --query "temperature regulation"
"""

import sys
import os

# Set environment variables before any backend imports to avoid database requirements
os.environ["INGEST_NO_DB"] = "true"
# Set dummy DATABASE_URL to satisfy settings initialization (won't be used with INGEST_NO_DB)
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import json
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from backend.ingest import TechnicalRAGPipeline
    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.core.schema import NodeWithScore
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("   Make sure you're running from the repo root and dependencies are installed.")
    sys.exit(1)


def count_nodes_by_content_type(docstore_path: Path) -> Tuple[Dict[str, int], List[Dict]]:
    """
    Count nodes by content_type from docstore.json and return all nodes for inspection.
    
    Returns (counts_dict, nodes_list) where nodes_list contains all node data.
    """
    if not docstore_path.exists():
        return {}, []
    
    try:
        with open(docstore_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # LlamaIndex docstore format: {"docstore/data": {...}}
        docstore_data = data.get("docstore/data", {})
        
        counts: Dict[str, int] = {}
        total = 0
        nodes_list = []
        
        for node_id, node_data in docstore_data.items():
            if isinstance(node_data, dict):
                metadata = node_data.get("metadata", {})
                content_type = metadata.get("content_type", "text")
                counts[content_type] = counts.get(content_type, 0) + 1
                total += 1
                # Store full node data for inspection
                nodes_list.append({
                    "node_id": node_id,
                    "metadata": metadata,
                    "text": node_data.get("text", ""),
                    "node_data": node_data
                })
        
        counts["_total"] = total
        return counts, nodes_list
        
    except Exception as e:
        print(f"❌ Error reading docstore.json: {e}")
        return {}, []


def unwrap_node(node_or_nodewithscore) -> Any:
    """Safely unwrap NodeWithScore to get the underlying node."""
    if isinstance(node_or_nodewithscore, NodeWithScore):
        if hasattr(node_or_nodewithscore, 'node'):
            return node_or_nodewithscore.node
        return node_or_nodewithscore
    return node_or_nodewithscore


def get_content_type(node_or_nodewithscore) -> str:
    """Safely extract content_type from a node."""
    node = unwrap_node(node_or_nodewithscore)
    
    if hasattr(node, 'metadata') and node.metadata:
        return node.metadata.get('content_type', 'text')
    
    return 'text'


def check_config_extract_images(config_path: str) -> bool:
    """Check if extract_images is disabled in config."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        extract_images = config.get("non_text", {}).get("extract_images", True)
        return not extract_images  # Return True if disabled (False)
    except Exception as e:
        print(f"⚠️  Warning: Could not parse config to check extract_images: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test to validate no image ingestion"
    )
    parser.add_argument(
        "test_docs_dir",
        type=str,
        help="Directory containing test PDF/DOCX files"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help="Directory to persist index (default: temporary directory)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="temperature regulation",
        help="Test query to run against the index (default: 'temperature regulation')"
    )
    
    args = parser.parse_args()
    
    test_docs_dir = Path(args.test_docs_dir)
    if not test_docs_dir.exists():
        print(f"❌ Test docs directory does not exist: {test_docs_dir}")
        sys.exit(1)
    
    # Check config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"⚠️  Warning: Config file not found: {config_path}")
        print("   Continuing anyway, but extract_images check will be skipped.")
    else:
        if not check_config_extract_images(str(config_path)):
            print(f"❌ FAIL: Config has extract_images enabled (should be false)")
            print(f"   Edit {config_path} and set non_text.extract_images: false")
            sys.exit(1)
        print(f"✅ Config check passed: extract_images is disabled")
    
    # Setup persist directory
    use_temp = args.persist_dir is None
    if use_temp:
        persist_dir = Path(tempfile.mkdtemp(prefix="smoke_test_index_"))
        print(f"📁 Using temporary directory: {persist_dir}")
    else:
        persist_dir = Path(args.persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Using persist directory: {persist_dir}")
    
    try:
        # Step 1: Build index
        print("\n" + "=" * 60)
        print("STEP 1: Building index from test documents")
        print("=" * 60)
        
        pipeline = TechnicalRAGPipeline(config_path=str(config_path))
        
        # Initialize models if needed
        if hasattr(pipeline, 'initialize_models'):
            print("🔧 Initializing models...")
            try:
                pipeline.initialize_models()
            except Exception as e:
                print(f"⚠️  Warning: Model initialization failed: {e}")
                print("   Continuing anyway (may fail later if models are required)")
        
        # Build index
        print(f"📚 Building index from: {test_docs_dir}")
        pipeline.build_index(
            data_dir=str(test_docs_dir),
            storage_dir=str(persist_dir),
            use_qdrant=False,
            dry_run=False
        )
        
        print("✅ Index built successfully")
        
        # Step 2a: Check for extracted image artifacts
        print("\n" + "=" * 60)
        print("STEP 2a: Checking for extracted image artifacts")
        print("=" * 60)
        
        # Check extracted_content directory for image files
        extracted_content_dir = Path("extracted_content")
        image_artifacts = []
        if extracted_content_dir.exists():
            for img_file in extracted_content_dir.rglob("*_img*.png"):
                image_artifacts.append(img_file)
        
        if image_artifacts:
            print(f"❌ FAIL: Found {len(image_artifacts)} extracted image file(s):")
            for img_path in image_artifacts[:10]:  # Show first 10
                print(f"   {img_path}")
            if len(image_artifacts) > 10:
                print(f"   ... and {len(image_artifacts) - 10} more")
            print("   Expected: 0 image files when extract_images is disabled")
            sys.exit(1)
        
        print("✅ No extracted image artifacts found")
        
        # Step 2b: Inspect docstore.json
        print("\n" + "=" * 60)
        print("STEP 2b: Inspecting docstore.json for image nodes")
        print("=" * 60)
        
        docstore_path = persist_dir / "docstore.json"
        if not docstore_path.exists():
            print(f"❌ FAIL: docstore.json not found at {docstore_path}")
            sys.exit(1)
        
        counts, nodes_list = count_nodes_by_content_type(docstore_path)
        
        if not counts:
            print("❌ FAIL: Could not read node counts from docstore.json")
            sys.exit(1)
        
        total = counts.pop("_total", 0)
        print(f"📊 Total nodes: {total}")
        print(f"📊 Nodes by content_type:")
        for content_type, count in sorted(counts.items()):
            print(f"   {content_type}: {count}")
        
        # Check 1: content_type == "image"
        image_count = counts.get("image", 0)
        if image_count > 0:
            print(f"\n❌ FAIL: Found {image_count} node(s) with content_type='image'")
            print("   Expected: 0 image nodes")
            sys.exit(1)
        
        # Check 2: Metadata signature (image_index, width, height)
        image_metadata_nodes = []
        for node in nodes_list:
            metadata = node.get("metadata", {})
            # Check for image metadata signatures
            has_image_index = "image_index" in metadata
            has_image_dims = "width" in metadata and "height" in metadata and "image_index" in metadata
            
            if has_image_index or has_image_dims:
                image_metadata_nodes.append({
                    "node_id": node.get("node_id", "unknown"),
                    "metadata_keys": list(metadata.keys())
                })
        
        if image_metadata_nodes:
            print(f"\n❌ FAIL: Found {len(image_metadata_nodes)} node(s) with image metadata signatures:")
            for img_node in image_metadata_nodes[:5]:  # Show first 5
                print(f"   node_id={img_node['node_id']}, keys={img_node['metadata_keys']}")
            if len(image_metadata_nodes) > 5:
                print(f"   ... and {len(image_metadata_nodes) - 5} more")
            print("   Expected: 0 nodes with image_index/width/height metadata")
            sys.exit(1)
        
        # Check 3: Text signature ("Image from ")
        image_text_nodes = []
        for node in nodes_list:
            text = node.get("text", "")
            if text.startswith("Image from "):
                image_text_nodes.append({
                    "node_id": node.get("node_id", "unknown"),
                    "text_preview": text[:100]
                })
        
        if image_text_nodes:
            print(f"\n❌ FAIL: Found {len(image_text_nodes)} node(s) with image text signature:")
            for img_node in image_text_nodes[:5]:  # Show first 5
                print(f"   node_id={img_node['node_id']}, text='{img_node['text_preview']}...'")
            if len(image_text_nodes) > 5:
                print(f"   ... and {len(image_text_nodes) - 5} more")
            print("   Expected: 0 nodes with text starting with 'Image from '")
            sys.exit(1)
        
        print("✅ No image nodes found in docstore.json (content_type, metadata, or text signatures)")
        
        # Step 3: Run query and check results
        print("\n" + "=" * 60)
        print("STEP 3: Running query and checking retrieval results")
        print("=" * 60)
        
        print(f"🔍 Query: '{args.query}'")
        
        # Try to use pipeline's hybrid_search if available
        if hasattr(pipeline, 'hybrid_search') and pipeline.index:
            print("   Using TechnicalRAGPipeline.hybrid_search()")
            try:
                results = pipeline.hybrid_search(args.query, top_k=10)
            except Exception as e:
                print(f"❌ FAIL: Query failed: {e}")
                sys.exit(1)
        else:
            # Fallback: load index directly
            print("   Loading index via StorageContext...")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
                index = load_index_from_storage(storage_context)
                retriever = index.as_retriever(similarity_top_k=10)
                results = retriever.retrieve(args.query)
            except Exception as e:
                print(f"❌ FAIL: Failed to load index or run query: {e}")
                sys.exit(1)
        
        print(f"📊 Retrieved {len(results)} results")
        
        # Check each result for image nodes
        image_results = []
        for i, result in enumerate(results):
            content_type = get_content_type(result)
            if content_type == "image":
                image_results.append((i, result))
        
        if image_results:
            print(f"\n❌ FAIL: Query returned {len(image_results)} image node(s):")
            for idx, result in image_results:
                node = unwrap_node(result)
                node_id = getattr(node, 'node_id', 'unknown')
                print(f"   Result {idx}: node_id={node_id}, content_type=image")
            sys.exit(1)
        
        print("✅ No image nodes in query results")
        
        # Success!
        print("\n" + "=" * 60)
        print("✅ PASS: All checks passed")
        print("=" * 60)
        print(f"   - Config: extract_images disabled")
        print(f"   - Docstore: {total} total nodes, 0 image nodes")
        print(f"   - Query: {len(results)} results, 0 image nodes")
        print("=" * 60)
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ FAIL: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Cleanup temp directory
        if use_temp and persist_dir.exists():
            print(f"\n🧹 Cleaning up temporary directory: {persist_dir}")
            shutil.rmtree(persist_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

