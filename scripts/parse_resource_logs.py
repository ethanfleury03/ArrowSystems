#!/usr/bin/env python3
"""
Parse [RESOURCE] checkpoint logs and print summary table.

Reads from stdin or file, extracts [RESOURCE] checkpoint lines,
and prints a summary table with peak RSS.

Usage:
  # From stdin (Cloud Run logs)
  gcloud logging read "resource.type=cloud_run_revision" --format=json | \
    python scripts/parse_resource_logs.py

  # From log file
  python scripts/parse_resource_logs.py < logs/app.log

  # From file argument
  python scripts/parse_resource_logs.py logs/app.log
"""

import sys
import re
from typing import List, Tuple, Optional
from collections import defaultdict


def parse_checkpoint_line(line: str) -> Optional[Tuple[str, float, Optional[float]]]:
    """
    Parse a [RESOURCE] checkpoint line.
    
    Format: [RESOURCE] {name} elapsed={elapsed:.2f}s rss_mb={rss:.1f}MB
    Or:     [RESOURCE] {name} elapsed={elapsed:.2f}s rss_mb=unknown
    
    Returns:
        (checkpoint_name, elapsed_seconds, rss_mb) or None if not a checkpoint line
    """
    pattern = r'\[RESOURCE\]\s+(\S+)\s+elapsed=([\d.]+)s\s+rss_mb=([\d.]+|unknown)MB?'
    match = re.search(pattern, line)
    if not match:
        return None
    
    name = match.group(1)
    elapsed = float(match.group(2))
    rss_str = match.group(3)
    rss = float(rss_str) if rss_str != "unknown" else None
    
    return (name, elapsed, rss)


def main():
    # Read input
    if len(sys.argv) > 1:
        # Read from file
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            lines = f.readlines()
    else:
        # Read from stdin
        lines = sys.stdin.readlines()
    
    # Parse checkpoints
    checkpoints: List[Tuple[str, float, Optional[float]]] = []
    for line in lines:
        parsed = parse_checkpoint_line(line)
        if parsed:
            checkpoints.append(parsed)
    
    if not checkpoints:
        print("No [RESOURCE] checkpoints found in input.", file=sys.stderr)
        print("Expected format: [RESOURCE] {name} elapsed={seconds}s rss_mb={mb}MB", file=sys.stderr)
        return 1
    
    # Print summary table
    print("=" * 70)
    print("RESOURCE CHECKPOINT SUMMARY")
    print("=" * 70)
    print(f"{'Checkpoint':<30} {'Elapsed (s)':<15} {'RSS (MB)':<15} {'Delta (MB)':<15}")
    print("-" * 70)
    
    prev_memory = None
    memory_values = []
    
    for name, elapsed, memory in checkpoints:
        if prev_memory is not None and memory is not None:
            delta = f"+{memory - prev_memory:.1f}"
        else:
            delta = "—"
        
        memory_str = f"{memory:.1f}" if memory is not None else "unknown"
        print(f"{name:<30} {elapsed:<15.2f} {memory_str:<15} {delta:<15}")
        
        if memory is not None:
            memory_values.append(memory)
            prev_memory = memory
    
    # Print peak RSS
    print("=" * 70)
    if memory_values:
        peak_rss = max(memory_values)
        peak_checkpoint = next(
            (name for name, _, m in checkpoints if m == peak_rss),
            "unknown"
        )
        print(f"Peak RSS: {peak_rss:.1f} MB (at checkpoint: {peak_checkpoint})")
        print(f"\nRecommendation: Cloud Run memory should be at least {peak_rss * 1.5:.0f} MB")
        print(f"  (1.5x headroom: {peak_rss:.1f} MB × 1.5 = {peak_rss * 1.5:.0f} MB)")
    else:
        print("⚠️  No RSS measurements available (all checkpoints showed 'unknown')")
        print("   Install psutil in production for accurate measurements")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())




