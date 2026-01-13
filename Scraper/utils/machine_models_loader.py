"""
Load machine models from Cloud SQL or JSON file.

Supports two sources:
1. Cloud SQL: Direct connection to machine_models table
2. JSON file: Exported machine models data
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


class MachineModel:
    """Represents a machine model with its aliases."""
    
    def __init__(self, model_id: int, name: str, machine_kind: str = ""):
        self.id = model_id
        self.name = name
        self.machine_kind = machine_kind
        self.aliases = self._generate_aliases(name)
    
    def _generate_aliases(self, name: str) -> List[str]:
        """
        Generate alias variants for a machine model name.
        
        Examples:
        - "DuraFlex" → ["duraflex", "DURAFLEX", "Dura Flex", "dura flex"]
        - "EZCut 330" → ["ezcut 330", "EZ-Cut 330", "ez-cut 330"]
        - "2800" → ["2800"]
        """
        aliases = set()
        
        # Original name (normalized)
        aliases.add(name.strip())
        
        # Case variants
        aliases.add(name.lower())
        aliases.add(name.upper())
        aliases.add(name.title())
        
        # Spacing variants (if contains camelCase or numbers)
        # "DuraFlex" → "Dura Flex"
        import re
        spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        if spaced != name:
            aliases.add(spaced)
            aliases.add(spaced.lower())
            aliases.add(spaced.upper())
        
        # Hyphen variants (if contains camelCase)
        hyphenated = re.sub(r'([a-z])([A-Z])', r'\1-\2', name)
        if hyphenated != name:
            aliases.add(hyphenated)
            aliases.add(hyphenated.lower())
            aliases.add(hyphenated.upper())
        
        # Remove empty strings
        aliases.discard("")
        
        return sorted(list(aliases))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "machine_kind": self.machine_kind,
            "aliases": self.aliases
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MachineModel":
        """Create from dictionary."""
        model = cls(
            model_id=data["id"],
            name=data["name"],
            machine_kind=data.get("machine_kind", "")
        )
        # Use provided aliases if available, otherwise generate
        if "aliases" in data and isinstance(data["aliases"], list):
            model.aliases = list(set(model.aliases + data["aliases"]))
        return model


def _find_database_url() -> Optional[str]:
    """
    Find DATABASE_URL from multiple sources:
    1. Environment variable DATABASE_URL (already set)
    2. Root .env file (ArrowSystems/.env) - checked first since it's most likely
    3. Scraper/.env file
    
    Returns:
        DATABASE_URL string or None
    """
    # Check environment variable first (might already be set)
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url
    
    # Try loading from dotenv (checks multiple .env files)
    try:
        from dotenv import load_dotenv
        
        # Check root .env file first (project root, 2 levels up from utils/)
        root_env = Path(__file__).parent.parent.parent / ".env"
        if root_env.exists():
            load_dotenv(root_env, override=False)
            database_url = os.getenv("DATABASE_URL")
            if database_url:
                return database_url
        
        # Check Scraper/.env file
        scraper_env = Path(__file__).parent.parent / ".env"
        if scraper_env.exists():
            load_dotenv(scraper_env, override=False)
            database_url = os.getenv("DATABASE_URL")
            if database_url:
                return database_url
    except ImportError:
        pass
    
    return None


def load_from_cloudsql(database_url: Optional[str] = None) -> List[MachineModel]:
    """
    Load machine models from Cloud SQL database.
    
    Args:
        database_url: PostgreSQL connection string (defaults to DATABASE_URL from env/.env files)
        
    Returns:
        List of MachineModel objects
        
    Raises:
        ImportError: If psycopg2 not available
        Exception: If database connection fails
    """
    if not PSYCOPG2_AVAILABLE:
        raise ImportError(
            "psycopg2 not installed. Install with: pip install psycopg2-binary"
        )
    
    if not database_url:
        database_url = _find_database_url()
        if not database_url:
            raise ValueError(
                "DATABASE_URL not found. Check:\n"
                "  1. DATABASE_URL environment variable\n"
                "  2. Scraper/.env file\n"
                "  3. Root .env file\n"
                "Or provide database_url parameter."
            )
    
    # Connect to database
    # Handle SQLAlchemy-style URLs (postgresql+psycopg2://) by removing the +psycopg2 part
    # psycopg2.connect() expects postgresql:// format
    if database_url.startswith("postgresql+psycopg2://"):
        database_url = database_url.replace("postgresql+psycopg2://", "postgresql://", 1)
    
    # Try connecting, with fallback to port 5433 if 5432 fails (common Cloud SQL Proxy setup)
    try:
        conn = psycopg2.connect(database_url)
    except psycopg2.OperationalError as e:
        # If connection fails and URL uses localhost/127.0.0.1:5432, try 5433
        if ("localhost" in database_url or "127.0.0.1" in database_url) and ":5432" in database_url:
            fallback_url = database_url.replace(":5432", ":5433")
            try:
                print(f"[INFO] Connection to port 5432 failed, trying port 5433...")
                conn = psycopg2.connect(fallback_url)
            except psycopg2.OperationalError:
                # Re-raise original error if fallback also fails
                raise e
        else:
            raise
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT id, name, machine_kind
            FROM machine_models
            ORDER BY name ASC
        """)
        rows = cursor.fetchall()
        
        models = []
        for row in rows:
            model = MachineModel(
                model_id=row["id"],
                name=row["name"],
                machine_kind=row.get("machine_kind", "")
            )
            models.append(model)
        
        return models
    finally:
        conn.close()


def load_from_json(json_path: str) -> List[MachineModel]:
    """
    Load machine models from JSON file.
    
    Expected JSON format:
    [
        {
            "id": 1,
            "name": "DuraFlex",
            "machine_kind": "Print Engine",
            "aliases": ["duraflex", "Dura Flex"]  // optional
        },
        ...
    ]
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        List of MachineModel objects
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON is invalid
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Machine models JSON not found: {json_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data)}")
    
    models = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "id" not in item or "name" not in item:
            continue
        
        model = MachineModel.from_dict(item)
        models.append(model)
    
    return models


def export_to_json(models: List[MachineModel], output_path: str) -> None:
    """
    Export machine models to JSON file.
    
    Args:
        models: List of MachineModel objects
        output_path: Path to output JSON file
    """
    data = [model.to_dict() for model in models]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_machine_models(
    source: str = "cloudsql",
    database_url: Optional[str] = None,
    json_path: Optional[str] = None
) -> List[MachineModel]:
    """
    Load machine models from specified source.
    
    Args:
        source: "cloudsql" or "json"
        database_url: PostgreSQL connection string (for cloudsql)
        json_path: Path to JSON file (for json)
        
    Returns:
        List of MachineModel objects
    """
    if source == "cloudsql":
        return load_from_cloudsql(database_url)
    elif source == "json":
        if not json_path:
            raise ValueError("json_path required when source='json'")
        return load_from_json(json_path)
    else:
        raise ValueError(f"Unknown source: {source}. Use 'cloudsql' or 'json'")


if __name__ == "__main__":
    # Self-test: try loading models
    import sys
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        print(f"Loading models from JSON: {json_path}")
        models = load_from_json(json_path)
    else:
        print("Loading models from Cloud SQL...")
        try:
            models = load_from_cloudsql()
        except Exception as e:
            print(f"Failed to load from Cloud SQL: {e}")
            print("Usage: python machine_models_loader.py [path/to/machine_models.json]")
            sys.exit(1)
    
    print(f"Loaded {len(models)} machine models:")
    for model in models[:10]:  # Show first 10
        print(f"  {model.id}: {model.name} (aliases: {len(model.aliases)})")
    if len(models) > 10:
        print(f"  ... and {len(models) - 10} more")
