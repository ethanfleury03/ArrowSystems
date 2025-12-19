#!/usr/bin/env python3
"""
Verify metadata in the index.
Checks that document_id, machine_model_ids, machine_model_names are present.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Path to index (adjust if needed)
INDEX_DIR = Path("/workspace/ingest_work/index_artifact")
# Or if downloaded from GCS:
# INDEX_DIR = Path("/tmp/index_artifact")

# Allow override via environment variable
import os
if os.getenv("INDEX_DIR"):
    INDEX_DIR = Path(os.getenv("INDEX_DIR"))

DOCSTORE_FILE = INDEX_DIR / "docstore.json"

if not DOCSTORE_FILE.exists():
    print(f"❌ Index not found at {DOCSTORE_FILE}")
    print("   Options:")
    print("   1. Set INDEX_DIR environment variable:")
    print("      INDEX_DIR=/path/to/index python verify_index_metadata.py")
    print("   2. Download from GCS first:")
    print("      gsutil -m cp -r gs://arrow-rag-support-prod-rag/latest_model/ /tmp/index_artifact/")
    print("      INDEX_DIR=/tmp/index_artifact python verify_index_metadata.py")
    sys.exit(1)

print("=" * 70)
print("Index Metadata Verification")
print("=" * 70)
print(f"Index directory: {INDEX_DIR}")
print()

# Load docstore
print("Loading docstore.json...")
with open(DOCSTORE_FILE, 'r') as f:
    docstore = json.load(f)

nodes = docstore.get("docstore/data", {})
print(f"📊 Total nodes in index: {len(nodes)}")
print()

# Check metadata fields
metadata_stats = {
    "has_document_id": 0,
    "has_machine_model_ids": 0,
    "has_machine_model_names": 0,
    "has_source_gcs": 0,
    "missing_document_id": 0,
    "missing_machine_models": 0,
}

document_ids = set()
machine_models_by_doc = defaultdict(set)
source_gcs_paths = set()
file_names = set()

for node_id, node_data in nodes.items():
    metadata = node_data.get("metadata", {})
    
    # Check document_id
    doc_id = metadata.get("document_id")
    if doc_id:
        metadata_stats["has_document_id"] += 1
        document_ids.add(str(doc_id))
    else:
        metadata_stats["missing_document_id"] += 1
    
    # Check machine_model_ids
    mm_ids = metadata.get("machine_model_ids")
    if mm_ids:
        metadata_stats["has_machine_model_ids"] += 1
        if doc_id:
            if isinstance(mm_ids, list):
                machine_models_by_doc[str(doc_id)].update([str(x) for x in mm_ids])
            else:
                machine_models_by_doc[str(doc_id)].add(str(mm_ids))
    else:
        metadata_stats["missing_machine_models"] += 1
    
    # Check machine_model_names
    mm_names = metadata.get("machine_model_names")
    if mm_names:
        metadata_stats["has_machine_model_names"] += 1
    
    # Check source_gcs
    source_gcs = metadata.get("source_gcs") or metadata.get("gcs_path")
    if source_gcs:
        metadata_stats["has_source_gcs"] += 1
        source_gcs_paths.add(source_gcs)
    
    # Track file names
    file_name = metadata.get("file_name")
    if file_name:
        file_names.add(file_name)

print("📋 Metadata Coverage:")
total = len(nodes)
if total > 0:
    print(f"   Nodes with document_id: {metadata_stats['has_document_id']} ({metadata_stats['has_document_id']/total*100:.1f}%)")
    print(f"   Nodes with machine_model_ids: {metadata_stats['has_machine_model_ids']} ({metadata_stats['has_machine_model_ids']/total*100:.1f}%)")
    print(f"   Nodes with machine_model_names: {metadata_stats['has_machine_model_names']} ({metadata_stats['has_machine_model_names']/total*100:.1f}%)")
    print(f"   Nodes with source_gcs: {metadata_stats['has_source_gcs']} ({metadata_stats['has_source_gcs']/total*100:.1f}%)")
else:
    print("   No nodes found!")
print()

if metadata_stats["missing_document_id"] > 0:
    print(f"⚠️  WARNING: {metadata_stats['missing_document_id']} nodes missing document_id")
    print()

if metadata_stats["missing_machine_models"] > 0:
    print(f"⚠️  WARNING: {metadata_stats['missing_machine_models']} nodes missing machine_model_ids")
    print()

print(f"📄 Unique document_ids: {len(document_ids)}")
print(f"📄 Unique source_gcs paths: {len(source_gcs_paths)}")
print(f"📄 Unique file names: {len(file_names)}")
print()

# Show document_id distribution
if document_ids:
    print("📊 Document ID distribution (top 10 by node count):")
    doc_id_counts = defaultdict(int)
    for node_id, node_data in nodes.items():
        metadata = node_data.get("metadata", {})
        doc_id = metadata.get("document_id")
        if doc_id:
            doc_id_counts[str(doc_id)] += 1
    
    for doc_id, count in sorted(doc_id_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        mm_ids = list(machine_models_by_doc.get(doc_id, set()))
        print(f"   document_id={doc_id}: {count} nodes, machine_models={mm_ids}")
    print()

# Sample some nodes to show metadata
print("📝 Sample node metadata (first 3 nodes):")
for i, (node_id, node_data) in enumerate(list(nodes.items())[:3]):
    metadata = node_data.get("metadata", {})
    print(f"\n   Node {i+1} (id: {node_id[:20]}...):")
    print(f"     document_id: {metadata.get('document_id', 'MISSING')}")
    print(f"     file_name: {metadata.get('file_name', 'MISSING')}")
    print(f"     source_gcs: {(metadata.get('source_gcs') or metadata.get('gcs_path') or 'MISSING')[:60]}...")
    print(f"     machine_model_ids: {metadata.get('machine_model_ids', 'MISSING')}")
    print(f"     machine_model_names: {metadata.get('machine_model_names', 'MISSING')}")
    print(f"     page_label: {metadata.get('page_label', 'N/A')}")
    print(f"     chunk_index: {metadata.get('chunk_index', 'N/A')}")

print()
print("=" * 70)

# Final verdict
if metadata_stats["has_document_id"] / total > 0.95 and metadata_stats["has_machine_model_ids"] / total > 0.95:
    print("✅ Metadata verification PASSED")
    print(f"   >95% of nodes have required metadata fields")
else:
    print("⚠️  Metadata verification WARNING")
    print(f"   Some nodes missing required metadata fields")
    print(f"   document_id coverage: {metadata_stats['has_document_id']/total*100:.1f}%")
    print(f"   machine_model_ids coverage: {metadata_stats['has_machine_model_ids']/total*100:.1f}%")

print("=" * 70)

