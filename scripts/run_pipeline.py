"""User-facing entry point for the Approach 1 pipeline.

Run from the repository root with ``python scripts/run_pipeline.py``. The
script deliberately delegates to ``src.pipeline`` so one implementation owns
all stage ordering, logging, and artifact contracts.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repository importable when this file is invoked directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
