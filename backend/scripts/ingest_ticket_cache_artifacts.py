#!/usr/bin/env python3
"""
Ingest ticket cache artifacts into LlamaIndex RAG index.

Reads JSONL file of TicketCacheArtifact objects and inserts them as TextNodes
into the existing RAG index.

Usage:
    python -m backend.scripts.ingest_ticket_cache_artifacts --jsonl out/cache_artifacts.jsonl
    python -m backend.scripts.ingest_ticket_cache_artifacts --jsonl out/cache_artifacts.jsonl --dry-run
    python -m backend.scripts.ingest_ticket_cache_artifacts --jsonl out/cache_artifacts.jsonl --limit 10 --skip-existing
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Set

# Add repo root to path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core.schema import TextNode
from backend.utils.ticket_cache_artifacts import TicketCacheArtifact, validate_ticket_cache_artifact
from backend.utils.test_mode import get_index_dir
from backend.logging_config import get_logger

logger = get_logger(__name__)


def ensure_ticket_embed_model():
    """
    Ensure a non-OpenAI embed model is configured before creating VectorStoreIndex.
    
    IMPORTANT: Do NOT read Settings.embed_model directly as it triggers lazy resolution
    that tries to import OpenAI embeddings. Instead, check Settings._embed_model (private field).
    """
    # Check private field to avoid triggering lazy resolution
    if getattr(Settings, "_embed_model", None) is not None:
        logger.debug(
            "[TICKET_REINDEX] Embed model already configured",
            embed_model=type(Settings._embed_model).__name__
        )
        return
    
    # Preferred: reuse backend's embed model factory
    try:
        from backend.utils.embedding_utils import build_offline_embedding
        
        # Use same model as production: BAAI/bge-large-en-v1.5 (1024 dim)
        model_name = os.getenv("TICKET_EMBED_MODEL_NAME", "BAAI/bge-large-en-v1.5")
        
        # Determine cache directory (same logic as production)
        cache_dir = (
            os.getenv("HF_HOME") or
            os.getenv("SENTENCE_TRANSFORMERS_HOME") or
            "/app/.cache/huggingface"
        )
        
        # Build offline embedding model (no OpenAI required)
        embed_model = build_offline_embedding(
            model_name=model_name,
            cache_dir=cache_dir,
            device="cpu"  # Cloud Run Jobs run on CPU
        )
        
        # Set embed model (this is safe - assignment doesn't trigger resolution)
        Settings.embed_model = embed_model
        
        logger.info(
            "[TICKET_REINDEX] Using embed model (repo factory)",
            embed_model=type(Settings._embed_model).__name__,
            model_name=model_name,
            cache_dir=cache_dir
        )
        return
        
    except Exception as e:
        logger.warning(
            "[TICKET_REINDEX] Repo embed model factory not available, trying HF fallback",
            error=str(e),
            error_type=type(e).__name__
        )
    
    # Fallback: HuggingFaceEmbedding directly (only if module is available)
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        model_name = os.getenv("TICKET_EMBED_MODEL_NAME", "BAAI/bge-large-en-v1.5")
        
        # Determine cache directory
        cache_dir = (
            os.getenv("HF_HOME") or
            os.getenv("SENTENCE_TRANSFORMERS_HOME") or
            "/app/.cache/huggingface"
        )
        
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            cache_folder=cache_dir
        )
        
        logger.info(
            "[TICKET_REINDEX] Using embed model (HF)",
            model_name=model_name,
            embed_model=type(Settings._embed_model).__name__,
            cache_dir=cache_dir
        )
        return
        
    except ImportError as e:
        raise RuntimeError(
            "Ticket reindex requires a non-OpenAI embed model, but no embedder is configured and "
            "llama_index.embeddings.huggingface is unavailable. Either wire this to the backend's embed "
            "model factory or add llama-index-embeddings-huggingface to dependencies. "
            f"ImportError: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize HuggingFaceEmbedding: {type(e).__name__}: {str(e)}"
        ) from e


def load_artifacts_from_jsonl(jsonl_path: str, limit: int = None) -> list[TicketCacheArtifact]:
    """
    Load TicketCacheArtifact objects from JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
        limit: Optional limit on number of artifacts to load
        
    Returns:
        List of TicketCacheArtifact objects
        
    Raises:
        FileNotFoundError: If JSONL file doesn't exist
        ValueError: If JSONL is invalid
    """
    artifacts = []
    errors = []
    
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if limit and len(artifacts) >= limit:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                artifact = TicketCacheArtifact(**data)
                validate_ticket_cache_artifact(artifact)
                artifacts.append(artifact)
            except Exception as e:
                errors.append((line_num, str(e)))
                logger.warning(f"Failed to parse line {line_num}: {e}")
    
    if errors:
        logger.warning(f"Encountered {len(errors)} errors while parsing JSONL")
        if len(errors) > 10:
            logger.warning(f"Showing first 10 errors:")
        for line_num, error in errors[:10]:
            logger.warning(f"  Line {line_num}: {error}")
    
    return artifacts


def get_existing_node_ids(index: VectorStoreIndex) -> Set[str]:
    """
    Get set of existing node IDs from index.
    
    Args:
        index: VectorStoreIndex to query
        
    Returns:
        Set of node IDs (strings)
    """
    existing_ids = set()
    try:
        # Try to get all nodes from docstore
        docstore = index.storage_context.docstore
        if hasattr(docstore, 'docs'):
            for node_id in docstore.docs.keys():
                existing_ids.add(node_id)
        elif hasattr(docstore, 'get_all_document_hashes'):
            # Alternative method
            for doc_hash in docstore.get_all_document_hashes():
                existing_ids.add(doc_hash)
    except Exception as e:
        logger.warning(f"Could not enumerate existing node IDs: {e}")
    return existing_ids


def delete_node_by_id(index: VectorStoreIndex, node_id: str) -> bool:
    """
    Delete a node from the index by ID (for overwrite semantics).
    
    Args:
        index: VectorStoreIndex to modify
        node_id: Node ID to delete
        
    Returns:
        True if node was deleted, False otherwise
    """
    try:
        docstore = index.storage_context.docstore
        
        # Try to delete from docstore
        if hasattr(docstore, 'delete_document'):
            docstore.delete_document(node_id)
            return True
        elif hasattr(docstore, 'docs') and node_id in docstore.docs:
            # Direct deletion from docs dict
            del docstore.docs[node_id]
            return True
        
        # Also try to delete from vector store if possible
        vector_store = index.storage_context.vector_store
        if hasattr(vector_store, 'delete'):
            try:
                vector_store.delete([node_id])
            except Exception:
                pass  # Vector store deletion may not be supported
        
        return True
    except Exception as e:
        logger.warning(f"Could not delete node {node_id}: {e}")
        return False


def artifact_to_text_node(artifact: TicketCacheArtifact) -> TextNode:
    """
    Convert TicketCacheArtifact to LlamaIndex TextNode.
    
    Args:
        artifact: TicketCacheArtifact to convert
        
    Returns:
        TextNode ready for indexing
    """
    return TextNode(
        text=artifact.text,
        metadata=artifact.metadata,
        id_=artifact.id  # Use artifact.id as node_id for deduplication
    )


def ingest_artifacts(
    jsonl_path: str,
    index_dir: str = None,
    dry_run: bool = False,
    limit: int = None,
    skip_existing: bool = False
) -> Dict[str, Any]:
    """
    Ingest ticket cache artifacts into RAG index.
    
    Args:
        jsonl_path: Path to JSONL file with artifacts
        index_dir: Directory containing RAG index (defaults to get_index_dir())
        dry_run: If True, validate but don't insert
        limit: Optional limit on number of artifacts to process
        skip_existing: If True, skip artifacts with IDs that already exist in index
        
    Returns:
        Dict with counts and status
    """
    # CRITICAL: Ensure embed model is set BEFORE any VectorStoreIndex operations
    # This prevents LlamaIndex from trying to use OpenAI embeddings (which aren't available)
    # IMPORTANT: Do NOT read Settings.embed_model here - use ensure_ticket_embed_model() instead
    ensure_ticket_embed_model()
    
    # Set LLM to None to avoid OpenAI initialization
    Settings.llm = None
    
    if index_dir is None:
        index_dir = get_index_dir()
    
    index_dir = Path(index_dir)
    
    logger.info(f"Loading artifacts from {jsonl_path}")
    artifacts = load_artifacts_from_jsonl(jsonl_path, limit=limit)
    logger.info(f"Loaded {len(artifacts)} artifacts")
    
    if not artifacts:
        logger.warning("No artifacts to ingest")
        return {
            "total": 0,
            "inserted": 0,
            "skipped": 0,
            "failed": 0,
            "errors": []
        }
    
    # Load or create index
    index = None
    existing_ids = set()
    
    if index_dir.exists():
        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
            index = load_index_from_storage(storage_context)
            logger.info(f"Loaded existing index from {index_dir}")
            
            if skip_existing:
                existing_ids = get_existing_node_ids(index)
                logger.info(f"Found {len(existing_ids)} existing node IDs")
        except Exception as e:
            logger.warning(f"Failed to load index from {index_dir}: {e}")
            logger.info("Creating new index")
            index = VectorStoreIndex(nodes=[], show_progress=False)
    else:
        logger.info(f"Index directory {index_dir} does not exist, creating new index")
        index_dir.mkdir(parents=True, exist_ok=True)
        index = VectorStoreIndex(nodes=[], show_progress=False)
    
    # Process artifacts
    nodes_to_insert = []
    skipped = 0
    failed = 0
    overwritten = 0
    errors = []
    
    for artifact in artifacts:
        try:
            # Check if already exists
            if skip_existing and artifact.id in existing_ids:
                skipped += 1
                logger.debug(f"Skipping existing artifact: {artifact.id}")
                continue
            
            # For deterministic overwrite: delete existing node if present
            if artifact.id in existing_ids:
                logger.info(f"[TICKET_INGEST] Overwriting existing node id={artifact.id}")
                deleted = delete_node_by_id(index, artifact.id)
                if deleted:
                    overwritten += 1
                    # Remove from existing_ids set to avoid double-counting
                    existing_ids.discard(artifact.id)
                else:
                    logger.warning(f"[TICKET_INGEST] Could not delete existing node {artifact.id}, will attempt insert anyway")
            
            # Convert to TextNode
            node = artifact_to_text_node(artifact)
            nodes_to_insert.append(node)
            
        except Exception as e:
            failed += 1
            error_msg = f"Failed to process artifact {artifact.id}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    logger.info(f"Prepared {len(nodes_to_insert)} nodes for insertion (skipped: {skipped}, overwritten: {overwritten}, failed: {failed})")
    
    if dry_run:
        logger.info("DRY RUN: Would insert nodes but skipping actual insertion")
        return {
            "total": len(artifacts),
            "inserted": 0,
            "skipped": skipped,
            "overwritten": overwritten,
            "failed": failed,
            "errors": errors,
            "dry_run": True
        }
    
    # Insert nodes in batches
    if nodes_to_insert:
        batch_size = 50
        inserted = 0
        
        for i in range(0, len(nodes_to_insert), batch_size):
            batch = nodes_to_insert[i:i + batch_size]
            try:
                index.insert_nodes(batch)
                inserted += len(batch)
                logger.info(f"Inserted batch {i // batch_size + 1} ({inserted}/{len(nodes_to_insert)} nodes)")
            except Exception as e:
                logger.error(f"Failed to insert batch {i // batch_size + 1}: {e}")
                failed += len(batch)
                errors.append(f"Batch {i // batch_size + 1} insertion failed: {e}")
        
        # Verify overwrites: check that nodes were actually inserted/updated
        # This is a post-insert verification step for deterministic behavior
        try:
            final_existing_ids = get_existing_node_ids(index)
            verified_overwritten = 0
            for artifact in artifacts:
                if artifact.id in final_existing_ids:
                    # Node exists in final index (either newly inserted or overwritten)
                    if artifact.id in existing_ids:
                        verified_overwritten += 1
            if verified_overwritten > 0:
                logger.info(f"[TICKET_INGEST] Verified {verified_overwritten} nodes exist in final index")
        except Exception as e:
            logger.debug(f"Could not verify final node state: {e}")
        
        # Persist index
        try:
            index.storage_context.persist(persist_dir=str(index_dir))
            logger.info(f"Persisted index to {index_dir}")
        except Exception as e:
            logger.error(f"Failed to persist index: {e}")
            errors.append(f"Index persistence failed: {e}")
    else:
        inserted = 0
    
    logger.info(f"[TICKET_INGEST] Ingestion complete: total={len(artifacts)}, inserted={inserted}, overwritten={overwritten}, skipped={skipped}, failed={failed}")
    
    return {
        "total": len(artifacts),
        "inserted": inserted,
        "skipped": skipped,
        "overwritten": overwritten,
        "failed": failed,
        "errors": errors
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ingest ticket cache artifacts into RAG index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest all artifacts
  python -m backend.scripts.ingest_ticket_cache_artifacts --jsonl out/cache_artifacts.jsonl
  
  # Dry run (validate only)
  python -m backend.scripts.ingest_ticket_cache_artifacts --jsonl out/cache_artifacts.jsonl --dry-run
  
  # Limit to first 10 artifacts
  python -m backend.scripts.ingest_ticket_cache_artifacts --jsonl out/cache_artifacts.jsonl --limit 10
  
  # Skip artifacts that already exist
  python -m backend.scripts.ingest_ticket_cache_artifacts --jsonl out/cache_artifacts.jsonl --skip-existing
        """
    )
    
    parser.add_argument(
        "--jsonl",
        required=True,
        help="Path to JSONL file containing TicketCacheArtifact objects"
    )
    
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Directory containing RAG index (defaults to latest_model or latest_model_test)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate artifacts but don't insert into index"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of artifacts to process"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip artifacts with IDs that already exist in index"
    )
    
    args = parser.parse_args()
    
    try:
        result = ingest_artifacts(
            jsonl_path=args.jsonl,
            index_dir=args.index_dir,
            dry_run=args.dry_run,
            limit=args.limit,
            skip_existing=args.skip_existing
        )
        
        print("\n" + "=" * 70)
        print("INGESTION SUMMARY")
        print("=" * 70)
        print(f"Total artifacts: {result['total']}")
        print(f"Inserted: {result['inserted']}")
        print(f"Overwritten: {result.get('overwritten', 0)}")
        print(f"Skipped: {result['skipped']}")
        print(f"Failed: {result['failed']}")
        
        if result.get('dry_run'):
            print("\n[DRY RUN] No changes were made to the index")
        
        if result['errors']:
            print(f"\nErrors ({len(result['errors'])}):")
            for error in result['errors'][:10]:
                print(f"  - {error}")
            if len(result['errors']) > 10:
                print(f"  ... and {len(result['errors']) - 10} more errors")
        
        print("=" * 70)
        
        # Exit with error code if failures occurred
        if result['failed'] > 0:
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
