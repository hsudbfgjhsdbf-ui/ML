"""Deep-learning utilities for deterministic seeds, device selection, and I/O."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch sources for a repeatable run.

    Args:
        seed: Non-negative random seed.
    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CPU operations are the default target. CUDA determinism is
    # requested where possible but may reduce throughput on some kernels.
    torch.use_deterministic_algorithms(False)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` into a PyTorch device.

    Args:
        requested: Device policy from YAML.
    Returns:
        A usable torch.device.
    Raises:
        RuntimeError: If CUDA was explicitly requested but unavailable.
    """
    requested = requested.lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a temporary file and rename operation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def parameter_count(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
