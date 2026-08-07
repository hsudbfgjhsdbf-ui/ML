"""Shared configuration, logging, reproducibility, and safe file utilities.

This module deliberately has no modelling logic. Keeping project plumbing here makes
all training and reporting entry points use identical paths, seed handling and logs.
"""
from __future__ import annotations

import json
import logging
import random
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectJSONEncoder(json.JSONEncoder):
    """Encode common scientific and path objects in machine-readable project metadata."""

    def default(self, value: Any) -> Any:
        """Return a JSON-compatible representation or delegate to the base encoder."""
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return super().default(value)


def load_config(config_path: str | Path = "configs/traditional_ml.yaml") -> dict[str, Any]:
    """Load YAML configuration and resolve all configured paths against project root.

    Args:
        config_path: Repository-relative or absolute YAML path.

    Returns:
        Nested configuration dictionary with a ``Path`` object for every ``paths`` value.

    Raises:
        FileNotFoundError: If the requested configuration file is absent.
        ValueError: If the YAML root is not a mapping.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    for key, value in config.get("paths", {}).items():
        resolved = Path(value)
        config["paths"][key] = resolved if resolved.is_absolute() else PROJECT_ROOT / resolved
    return config


def ensure_directories(config: Mapping[str, Any]) -> None:
    """Create all configured artifact directories if they do not yet exist.

    Args:
        config: Loaded project configuration containing the ``paths`` mapping.
    """
    for key, path in config["paths"].items():
        # The raw workbook is a file; all remaining configured paths are directories
        # except the synthetic-data output, whose parent must exist.
        target = path.parent if key in {"raw_workbook", "synthetic_data"} else path
        Path(target).mkdir(parents=True, exist_ok=True)


def configure_logging(log_name: str = "pipeline", level: int = logging.INFO) -> logging.Logger:
    """Configure idempotent console and file logging for a project execution.

    Args:
        log_name: Human-readable run identifier used in the log filename.
        level: Minimum standard-library logging level.

    Returns:
        Configured named logger.
    """
    logger = logging.getLogger(f"fraud_detection.{log_name}")
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / f"{log_name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for Python and NumPy.

    Args:
        seed: Non-negative integer seed recorded in output metadata.
    """
    if seed < 0:
        raise ValueError("Random seed must be non-negative.")
    random.seed(seed)
    np.random.seed(seed)


def write_json(path: str | Path, payload: Any) -> None:
    """Write complete JSON content atomically enough for normal local project use.

    Args:
        path: Destination JSON file.
        payload: JSON-serialisable data, including supported NumPy/Path values.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, cls=ProjectJSONEncoder, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: str | Path) -> Any:
    """Read JSON metadata with a useful missing-file error.

    Args:
        path: JSON file to read.

    Returns:
        Decoded JSON object.

    Raises:
        FileNotFoundError: If metadata is missing.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Required metadata file not found: {source}")
    return json.loads(source.read_text(encoding="utf-8"))
