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

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode
from backend.utils.ticket_cache_artifacts import TicketCacheArtifact, validate_ticket_cache_artifact
from backend.utils.test_mode import get_index_dir
from backend.logging_config import get_logger

logger = get_logger(__name__)


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
    errors = []
    
    for artifact in artifacts:
        try:
            # Check if already exists
            if skip_existing and artifact.id in existing_ids:
                skipped += 1
                logger.debug(f"Skipping existing artifact: {artifact.id}")
                continue
            
            # Convert to TextNode
            node = artifact_to_text_node(artifact)
            nodes_to_insert.append(node)
            
        except Exception as e:
            failed += 1
            error_msg = f"Failed to process artifact {artifact.id}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    logger.info(f"Prepared {len(nodes_to_insert)} nodes for insertion (skipped: {skipped}, failed: {failed})")
    
    if dry_run:
        logger.info("DRY RUN: Would insert nodes but skipping actual insertion")
        return {
            "total": len(artifacts),
            "inserted": 0,
            "skipped": skipped,
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
        
        # Persist index
        try:
            index.storage_context.persist(persist_dir=str(index_dir))
            logger.info(f"Persisted index to {index_dir}")
        except Exception as e:
            logger.error(f"Failed to persist index: {e}")
            errors.append(f"Index persistence failed: {e}")
    else:
        inserted = 0
    
    return {
        "total": len(artifacts),
        "inserted": inserted,
        "skipped": skipped,
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
