#!/usr/bin/env python3
"""
Export machine models from Cloud SQL to JSON file.

Usage:
    python scripts/export_machine_models.py --output machine_models.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.machine_models_loader import load_from_cloudsql, export_to_json


def main():
    parser = argparse.ArgumentParser(description="Export machine models to JSON")
    parser.add_argument(
        "--output",
        type=str,
        default="machine_models.json",
        help="Output JSON file path (default: machine_models.json)"
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="PostgreSQL connection string (defaults to DATABASE_URL env var)"
    )
    
    args = parser.parse_args()
    
    database_url = args.database_url or os.getenv("DATABASE_URL")
    if not database_url:
        print("[ERROR] DATABASE_URL environment variable not set.")
        print("Set DATABASE_URL or use --database-url parameter.")
        sys.exit(1)
    
    print("Loading machine models from Cloud SQL...")
    try:
        models = load_from_cloudsql(database_url)
        print(f"Loaded {len(models)} machine models")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        sys.exit(1)
    
    output_path = Path(args.output)
    export_to_json(models, str(output_path))
    print(f"Exported to: {output_path}")
    print(f"\nFirst 5 models:")
    for model in models[:5]:
        print(f"  {model.id}: {model.name} ({len(model.aliases)} aliases)")


if __name__ == "__main__":
    main()
