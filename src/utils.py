"""Shared utilities for the Medical Insurance Claim Fraud Detection project.

This module provides configuration loading, logging setup, path helpers and
generic helpers used across the traditional ML, deep learning and agent AI
approaches. Keeping shared code in one place enforces separation of concerns.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import yaml

# Root of the repository (three levels up from this file's parent).
ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = "config/config.yaml") -> dict:
    """Load a YAML configuration file into a nested dictionary.

    Args:
        path: Path to the YAML configuration file (relative or absolute).

    Returns:
        dict: Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> logging.Logger:
    """Configure a root logger with console (and optional file) handlers.

    Args:
        level: Logging verbosity level.
        log_file: Optional path to write logs to.

    Returns:
        logging.Logger: The configured root logger.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("fraud")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Path: The created/existing directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path, indent: int = 2) -> None:
    """Serialise an object to a JSON file.

    Args:
        obj: JSON-serialisable object.
        path: Output file path.
        indent: JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)


def load_json(path: str | Path):
    """Load a JSON file into a Python object.

    Args:
        path: Input file path.

    Returns:
        object: Parsed JSON content.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the repository root.

    Args:
        path: A path.

    Returns:
        Path: The absolute path.
    """
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
