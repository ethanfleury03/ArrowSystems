#!/usr/bin/env python3
"""
Measure runtime memory footprint for deployment sizing.
Loads models and index, runs sample queries, reports memory at checkpoints.

This script uses the same initialization and query paths as production,
providing accurate memory measurements for Cloud Run sizing decisions.
"""

import os
import sys
import argparse
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

# Set environment for no-db mode (if needed)
os.environ.setdefault("INGEST_NO_DB", "true")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from backend.utils.resource_monitor import log_resource_checkpoint, get_memory_mb, get_elapsed_seconds


def main():
    parser = argparse.ArgumentParser(
        description="Measure RAG pipeline memory footprint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic measurement (no queries)
  python scripts/measure_runtime_footprint.py --storage-dir latest_model

  # With test queries
  python scripts/measure_runtime_footprint.py --storage-dir latest_model --run-queries

  # Custom threshold
  python scripts/measure_runtime_footprint.py --storage-dir latest_model --max-mb 2048
        """
    )
    parser.add_argument(
        "--storage-dir",
        default="latest_model",
        help="Index storage directory (default: latest_model)"
    )
    parser.add_argument(
        "--run-queries",
        action="store_true",
        help="Run sample queries to measure query-time memory"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results per query (default: 5)"
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=4096,
        help="Fail if peak memory exceeds this threshold in MB (default: 4096)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("RAG PIPELINE MEMORY FOOTPRINT MEASUREMENT")
    print("=" * 70)
    print(f"Storage dir: {args.storage_dir}")
    print(f"Max memory threshold: {args.max_mb} MB")
    print(f"Run queries: {args.run_queries}")
    if args.run_queries:
        print(f"Top-K per query: {args.top_k}")
    print()
    
    checkpoints = []
    
    # Checkpoint 1: Process start
    log_resource_checkpoint("process_start")
    checkpoints.append(("process_start", get_memory_mb(), get_elapsed_seconds()))
    
    # Checkpoint 2: After imports
    log_resource_checkpoint("imports_complete")
    checkpoints.append(("imports_complete", get_memory_mb(), get_elapsed_seconds()))
    
    # Checkpoint 3: Initialize pipeline
    print("\n[STEP] Initializing RAG pipeline...")
    try:
        from backend.rag_pipeline import initialize_rag_pipeline
        
        pipeline, success = initialize_rag_pipeline(storage_dir=args.storage_dir)
        
        if not success:
            print(f"❌ FAIL: Pipeline initialization failed")
            debug_status = pipeline.debug_status() if pipeline else {}
            last_error = debug_status.get("last_error", "Unknown error")
            print(f"   Error: {last_error}")
            return 1
        
        log_resource_checkpoint("pipeline_initialized")
        checkpoints.append(("pipeline_initialized", get_memory_mb(), get_elapsed_seconds()))
        
        # Checkpoint 4: After queries (if requested)
        if args.run_queries:
            print("\n[STEP] Running sample queries...")
            test_queries = [
                "What is the temperature regulation range?",
                "How do I configure the printhead?",
                "What are the maintenance requirements?"
            ]
            
            for i, query in enumerate(test_queries, 1):
                print(f"  Query {i}/{len(test_queries)}: {query[:50]}...")
                try:
                    response = pipeline.query(
                        query=query,
                        top_k=args.top_k,
                        alpha=0.5,
                        dynamic_windowing=True
                    )
                    log_resource_checkpoint(f"query_{i}_complete")
                    checkpoints.append((f"query_{i}_complete", get_memory_mb(), get_elapsed_seconds()))
                    print(f"    ✅ Retrieved {len(response.sources)} chunks")
                except Exception as e:
                    print(f"    ⚠️  Query {i} failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Final checkpoint
        log_resource_checkpoint("measurement_complete")
        checkpoints.append(("measurement_complete", get_memory_mb(), get_elapsed_seconds()))
        
    except Exception as e:
        print(f"❌ FAIL: Error during measurement: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print summary table
    print("\n" + "=" * 70)
    print("MEMORY FOOTPRINT SUMMARY")
    print("=" * 70)
    print(f"{'Checkpoint':<30} {'Elapsed (s)':<15} {'RSS (MB)':<15} {'Delta (MB)':<15}")
    print("-" * 70)
    
    prev_memory = None
    for name, memory, elapsed in checkpoints:
        if prev_memory is not None and memory is not None:
            delta = f"+{memory - prev_memory:.1f}"
        else:
            delta = "—"
        memory_str = f"{memory:.1f}" if memory is not None else "unknown"
        print(f"{name:<30} {elapsed:<15.2f} {memory_str:<15} {delta:<15}")
        if memory is not None:
            prev_memory = memory
    
    # Check threshold
    memory_values = [m for _, m, _ in checkpoints if m is not None]
    if memory_values:
        max_memory = max(memory_values)
        print(f"\n{'='*70}")
        if max_memory > args.max_mb:
            print(f"❌ FAIL: Peak memory {max_memory:.1f} MB exceeds threshold {args.max_mb} MB")
            return 1
        else:
            print(f"✅ PASS: Peak memory {max_memory:.1f} MB is within threshold {args.max_mb} MB")
            print(f"\nRecommendation: Cloud Run memory should be at least {max_memory * 1.5:.0f} MB")
            print(f"  (1.5x headroom for safety: {max_memory:.1f} MB × 1.5 = {max_memory * 1.5:.0f} MB)")
    else:
        print("⚠️  WARNING: Memory measurement unavailable (psutil not installed)")
        print("   Install psutil for accurate measurements: pip install psutil")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

