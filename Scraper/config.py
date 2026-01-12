"""
Centralized configuration for Anthropic API and other settings.

Loads environment variables from Scraper/.env file.
Provides single source of truth for API keys, model selection, and defaults.

Example usage:
    from config import get_anthropic_api_key, get_anthropic_model, get_anthropic_client
    
    api_key = get_anthropic_api_key()
    model = get_anthropic_model()
    client = get_anthropic_client()
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: python-dotenv package not installed. Run: pip install python-dotenv", file=sys.stderr)
    sys.exit(1)

# Load .env file from Scraper directory
SCRAPER_DIR = Path(__file__).resolve().parent
ENV_FILE = SCRAPER_DIR / ".env"
load_dotenv(ENV_FILE)

# Default model (Claude Sonnet 4)
DEFAULT_MODEL = "claude-sonnet-4-20250514"

# Default max tokens for different operations
DEFAULT_MAX_TOKENS_EXTRACTOR = 800
DEFAULT_MAX_TOKENS_VERIFIER = 600

# Default temperature (0 for deterministic outputs)
DEFAULT_TEMPERATURE = 0


def get_anthropic_api_key() -> str:
    """
    Get Anthropic API key from environment variable.
    
    Returns:
        API key string
        
    Raises:
        ValueError: If API key is not set
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            f"ANTHROPIC_API_KEY environment variable not set.\n"
            f"Please set it in {ENV_FILE} or as an environment variable."
        )
    return api_key


def get_anthropic_model() -> str:
    """
    Get Anthropic model name from environment variable, or return default.
    
    Returns:
        Model name string (default: claude-sonnet-4-20250514)
    """
    return os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)


def get_anthropic_client():
    """
    Get configured Anthropic client instance.
    
    Returns:
        Anthropic client instance
        
    Raises:
        ValueError: If API key is not set
        ImportError: If anthropic package is not installed
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not installed. Run: pip install anthropic"
        )
    
    api_key = get_anthropic_api_key()
    return Anthropic(api_key=api_key)


def log_config():
    """
    Log the current configuration (model only, not API key).
    Call this at startup of scripts that use Anthropic.
    """
    model = get_anthropic_model()
    print(f"Using Anthropic model: {model}", file=sys.stderr)


def check_config() -> dict:
    """
    Self-check function to verify configuration.
    
    Returns:
        Dict with status information
    """
    api_key_loaded = False
    api_key_error = None
    
    try:
        get_anthropic_api_key()
        api_key_loaded = True
    except ValueError as e:
        api_key_error = str(e)
    
    model = get_anthropic_model()
    
    return {
        "model": model,
        "api_key_loaded": api_key_loaded,
        "api_key_error": api_key_error,
        "env_file": str(ENV_FILE)
    }


if __name__ == "__main__":
    """
    CLI self-check: Print configuration status.
    
    Usage:
        python -m Scraper.config
    """
    status = check_config()
    
    print("Anthropic Configuration Check")
    print("=" * 60)
    print(f"Model: {status['model']}")
    print(f"API key loaded: {'yes' if status['api_key_loaded'] else 'no'}")
    if not status['api_key_loaded']:
        print(f"Error: {status['api_key_error']}")
    print(f"Environment file: {status['env_file']}")
    
    if not status['api_key_loaded']:
        sys.exit(1)

