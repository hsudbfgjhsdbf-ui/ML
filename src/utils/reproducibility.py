"""Reproducibility, hashing, and environment snapshot helpers.

Fraud detection results are only useful when the exact data snapshot,
configuration, software environment, and random seed can be recovered.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy stochastic sources.

    Args:
        seed: Non-negative deterministic seed.
    Returns:
        None.
    Raises:
        ValueError: If the seed is negative.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: Path) -> str:
    """Return a SHA-256 checksum for a file without loading it all at once.

    Args:
        path: File to hash.
    Returns:
        Lowercase hexadecimal SHA-256 digest.
    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    """Hash a JSON-serializable value using canonical key ordering.

    Args:
        value: JSON-compatible object.
    Returns:
        SHA-256 digest of its canonical representation.
    """
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def environment_snapshot() -> dict[str, Any]:
    """Collect a small, serializable runtime and hardware snapshot.

    Returns:
        Dictionary suitable for a run manifest or JSON artifact.
    """
    packages: dict[str, str] = {}
    for name in ["numpy", "pandas", "sklearn", "scipy", "matplotlib", "joblib"]:
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # pragma: no cover - optional package import failure
            packages[name] = "unavailable"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "packages": packages,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON artifact with stable formatting.

    Args:
        path: Destination path.
        payload: JSON-compatible mapping.
    Returns:
        None.
    Raises:
        OSError: If the destination cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)
