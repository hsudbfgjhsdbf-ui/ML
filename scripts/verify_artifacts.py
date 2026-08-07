"""Verify the lightweight integrity contract after a pipeline run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    """Check required files, leaderboard schema, and declared winner."""
    required = [
        ROOT / "data" / "metadata" / "raw_manifest.json",
        ROOT / "evaluation" / "leaderboard.csv",
        ROOT / "evaluation" / "evaluation.md",
        ROOT / "evaluation" / "test_results.json",
        ROOT / "evaluation" / "run_manifest.json",
        ROOT / "artifacts" / "models" / "best_model.joblib",
        ROOT / "presentation" / "approach_1_traditional_ml.pptx",
        ROOT / "reports" / "approach_1_project_report.pdf",
        ROOT / "reports" / "approach_1_ieee_paper.pdf",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        print("Missing artifacts:", *missing, sep="\n- ")
        return 1
    leaderboard = pd.read_csv(ROOT / "evaluation" / "leaderboard.csv")
    for column in ["rank", "key", "status", "val_f2", "val_pr_auc", "test_f2"]:
        if column not in leaderboard.columns:
            print(f"Missing leaderboard column: {column}")
            return 1
    manifest = json.loads((ROOT / "evaluation" / "run_manifest.json").read_text(encoding="utf-8"))
    winner = manifest.get("winner_key")
    if winner is None:
        winner_value = manifest.get("winner")
        winner = winner_value.get("key") if isinstance(winner_value, dict) else winner_value
    if winner not in set(leaderboard["key"].astype(str)):
        print(f"Winner {winner!r} is absent from leaderboard")
        return 1
    print(f"Artifact verification passed for {manifest['run_id']} with winner {winner}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
