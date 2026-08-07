"""Verify the dedicated Approach 2 artifact contract."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Check five models, XAI packs, reports, and manifest consistency."""
    required = [
        ROOT / "evaluation2" / "leaderboard.csv",
        ROOT / "evaluation2" / "evaluation.md",
        ROOT / "evaluation2" / "run_manifest.json",
        ROOT / "presentation2" / "approach_2_deep_learning_xai.pptx",
        ROOT / "presentation2" / "slide_manifest.json",
        ROOT / "reports2" / "approach_2_project_report.pdf",
        ROOT / "reports2" / "approach_2_ieee_paper.pdf",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        print("Missing:", *missing, sep="\n- ")
        return 1
    leaderboard = pd.read_csv(ROOT / "evaluation2" / "leaderboard.csv")
    if len(leaderboard) != 5 or leaderboard["key"].nunique() != 5:
        print("Expected exactly five unique deep architecture rows")
        return 1
    manifest = json.loads((ROOT / "evaluation2" / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["winner_key"] not in set(leaderboard["key"]):
        print("Manifest winner is absent from deep leaderboard")
        return 1
    for key in leaderboard["key"]:
        for suffix in ["faithfulness.json", "stability.json", "local_dossier.json", "occlusion_importance.csv"]:
            if not (ROOT / "evaluation2" / "xai" / f"{key}_{suffix}").exists():
                print(f"Missing XAI artifact for {key}: {suffix}")
                return 1
    print(f"Approach 2 verification passed for {manifest['run_id']} with winner {manifest['winner_key']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
