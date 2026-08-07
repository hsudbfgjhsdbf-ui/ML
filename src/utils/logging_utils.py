"""Consistent console and file logging for reproducible pipeline runs."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_path: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure an idempotent project logger.

    Args:
        log_path: File where timestamped events are written.
        level: Logging threshold for console and file handlers.
    Returns:
        Configured ``medical_fraud`` logger.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("medical_fraud")
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
