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

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        # Fallback: use ASCII-safe characters
        pass

# Set environment variables before any backend imports to avoid database requirements
os.environ["INGEST_NO_DB"] = "true"
os.environ["DISABLE_METADATA_UPDATE"] = "1"
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
        
        # Step 4: Test orchestrator retrieval path (defense-in-depth)
        print("\n" + "=" * 60)
        print("STEP 4: Testing orchestrator retrieval path (defense-in-depth)")
        print("=" * 60)
        
        # This test MUST pass - fail hard if it can't run
        try:
            from backend.orchestrator import HybridRetriever
            from backend.config.env import settings
            from llama_index.core import Settings
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            
            print("   Initializing HybridRetriever directly (no database required)...")
            
            # Load the index we just built
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
            index = load_index_from_storage(storage_context)
            
            if index is None:
                print("   ❌ FAIL: Could not load index from persist_dir")
                print(f"   Persist dir: {persist_dir}")
                sys.exit(1)
            
            # Initialize embedding model (required for HybridRetriever)
            # Use cache_dir from pipeline if available, otherwise from settings or default
            cache_dir = getattr(pipeline, 'cache_dir', None)
            if not cache_dir:
                cache_dir = settings.HF_HOME if hasattr(settings, 'HF_HOME') else "/root/.cache/huggingface/hub"
            embed_model_name = pipeline.config.get("models", {}).get("embedding", "BAAI/bge-large-en-v1.5")
            embed_model = HuggingFaceEmbedding(
                model_name=embed_model_name,
                cache_folder=cache_dir
            )
            Settings.embed_model = embed_model
            
            # Initialize reranker if available
            reranker = None
            if hasattr(pipeline, 'reranker') and pipeline.reranker:
                reranker = pipeline.reranker
            
            # Create HybridRetriever directly (no database needed)
            retriever = HybridRetriever(
                index=index,
                embed_model=embed_model,
                reranker=reranker,
                document_evaluator=None
            )
            
            print("   ✅ HybridRetriever initialized")
            
            # Test hybrid_search via orchestrator
            print(f"   Running query via HybridRetriever: '{args.query}'")
            orchestrator_results = retriever.hybrid_search(
                query=args.query,
                top_k=10,
                alpha=0.5
            )
            
            print(f"   📊 HybridRetriever returned {len(orchestrator_results)} results")
            
            if len(orchestrator_results) == 0:
                print("   ⚠️  WARNING: HybridRetriever returned 0 results (may indicate index issue)")
            
            # Check for image nodes
            orchestrator_image_results = []
            for i, result in enumerate(orchestrator_results):
                content_type = get_content_type(result)
                if content_type == "image":
                    orchestrator_image_results.append((i, result))
            
            if orchestrator_image_results:
                print(f"\n❌ FAIL: HybridRetriever returned {len(orchestrator_image_results)} image node(s):")
                for idx, result in orchestrator_image_results:
                    node = unwrap_node(result)
                    node_id = getattr(node, 'node_id', 'unknown')
                    print(f"   Result {idx}: node_id={node_id}, content_type=image")
                sys.exit(1)
            
            print("   ✅ No image nodes in HybridRetriever results (defense-in-depth working)")
            
            # Test filtering helper directly with a fake image node
            print("   Testing image filtering helper with synthetic image node...")
            from backend.orchestrator import _is_image_node, _filter_image_nodes
            from llama_index.core.schema import TextNode
            
            # Create a fake image node
            fake_image_node = TextNode(
                text="Image from test.pdf",
                metadata={"content_type": "image", "file_name": "test.pdf"}
            )
            fake_image_node_wrapped = NodeWithScore(node=fake_image_node, score=0.5)
            
            # Test _is_image_node
            if not _is_image_node(fake_image_node_wrapped):
                print("   ❌ FAIL: _is_image_node() did not detect synthetic image node")
                sys.exit(1)
            
            # Test _filter_image_nodes
            test_nodes = [
                NodeWithScore(node=TextNode(text="Text node", metadata={"content_type": "text"}), score=0.8),
                fake_image_node_wrapped,
                NodeWithScore(node=TextNode(text="Table node", metadata={"content_type": "table"}), score=0.7)
            ]
            filtered = _filter_image_nodes(test_nodes)
            
            if len(filtered) != 2:
                print(f"   ❌ FAIL: _filter_image_nodes() should return 2 nodes, got {len(filtered)}")
                sys.exit(1)
            
            if any(_is_image_node(node) for node in filtered):
                print("   ❌ FAIL: _filter_image_nodes() did not remove image node")
                sys.exit(1)
            
            print("   ✅ Image filtering helpers work correctly")
            
        except ImportError as e:
            print(f"\n❌ FAIL: Could not import orchestrator modules: {e}")
            print("   This test is REQUIRED and cannot be skipped.")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ FAIL: Orchestrator test failed: {e}")
            print("   This test is REQUIRED and cannot be skipped.")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Step 5: Test single-file ingestion doesn't crash
        print("\n" + "=" * 60)
        print("STEP 5: Testing single-file ingestion path")
        print("=" * 60)
        
        # This test MUST pass - fail hard if it can't run
        try:
            from backend.utils.single_file_ingestion import ingest_single_file
            
            # Find a test PDF file
            test_pdf = None
            for pdf_file in Path(args.test_docs_dir).glob("*.pdf"):
                test_pdf = pdf_file
                break
            
            if not test_pdf:
                print(f"\n❌ FAIL: No PDF found in test_docs_dir: {args.test_docs_dir}")
                print("   This test requires at least one PDF file to test single-file ingestion.")
                sys.exit(1)
            
            print(f"   Testing single-file ingestion with: {test_pdf.name}")
            
            # Use the persist_dir we built (index must exist for single-file ingestion)
            # Single-file ingestion adds to an existing index, so we use the one we just built
            print(f"   Using existing index at: {persist_dir}")
            
            # Get cache_dir from pipeline or use default
            cache_dir = getattr(pipeline, 'cache_dir', "/root/.cache/huggingface/hub")
            
            # This should not crash and should not extract images
            result = ingest_single_file(
                file_path=str(test_pdf),
                storage_dir=str(persist_dir),  # Use the index we built
                config_path=str(config_path),
                cache_dir=cache_dir
            )
            
            if not result.get("success"):
                print(f"\n❌ FAIL: Single-file ingestion returned success=False")
                print(f"   Error: {result.get('error')}")
                print("   This test is REQUIRED and cannot be skipped.")
                sys.exit(1)
            
            print(f"   ✅ Single-file ingestion completed: {result.get('chunk_count', 0)} chunks")
            
            # Check that no image artifacts were created in extracted_content directory
            extracted_content_dir = Path("extracted_content")
            image_artifacts = []
            if extracted_content_dir.exists():
                for img_file in extracted_content_dir.rglob("*_img*.png"):
                    image_artifacts.append(img_file)
            
            if image_artifacts:
                print(f"\n❌ FAIL: Single-file ingestion created {len(image_artifacts)} image artifact(s):")
                for img_path in image_artifacts[:10]:  # Show first 10
                    print(f"   {img_path}")
                if len(image_artifacts) > 10:
                    print(f"   ... and {len(image_artifacts) - 10} more")
                print("   Expected: 0 image files when extract_images is disabled")
                sys.exit(1)
            
            print("   ✅ No image artifacts created by single-file ingestion")
            
        except ImportError as e:
            print(f"\n❌ FAIL: Could not import single_file_ingestion: {e}")
            print("   This test is REQUIRED and cannot be skipped.")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ FAIL: Single-file ingestion test failed: {e}")
            print("   This test is REQUIRED and cannot be skipped.")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Success!
        print("\n" + "=" * 60)
        print("✅ PASS: All checks passed")
        print("=" * 60)
        print(f"   - Config: extract_images disabled")
        print(f"   - Docstore: {total} total nodes, 0 image nodes")
        print(f"   - Query: {len(results)} results, 0 image nodes")
        print(f"   - Orchestrator: Defense-in-depth filtering verified")
        print(f"   - Single-file ingestion: No crashes, no image artifacts")
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

