"""Configuration loader with fallback."""
import os
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:
    yaml = None

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML config with environment variable override.
    
    Args:
        path: Optional path to config.yaml. Defaults to project root.
    
    Returns:
        Dict containing configuration.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return {}
    if yaml is None:
        # minimal fallback parsing not implemented -> return empty
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    # Override with .env if present (simple)
    env_path = cfg_path.parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv, dotenv_values
            load_dotenv(env_path)
        except ImportError:
            pass
    return config

def get_nested(config: Dict, dotted_key: str, default=None):
    keys = dotted_key.split(".")
    cur = config
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur
