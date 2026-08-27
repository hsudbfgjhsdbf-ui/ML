#!/usr/bin/env python
"""Build all presentation / report / research deliverables.

Assumes the three pipelines have been run (so results files exist). Generates:
  - PPT presentations (ml, dl, agent)
  - PDF reports (ml, dl, agent)
  - Research paper (PDF + MD)
  - Consolidated README/evaluation index

Usage:
    python make_deliverables.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.build_ppt import build_all as build_ppt
from src.build_pdf import build_all as build_pdf
from src.build_paper import build_all as build_paper
from src.shap_analysis import run as run_shap
from src.utils import ROOT, setup_logging

logger = setup_logging()


def build_index(eval_dir: Path, report_dir: Path, pres_dir: Path) -> None:
    """Write a deliverables index markdown summarising outputs."""
    lines = [
        "# Project Deliverables Index\n",
        "## Presentations",
        "- Approach 1 (Traditional ML): `presentation/presentation_ml.pptx`",
        "- Approach 2 (Deep Learning): `presentation/presentation_dl.pptx`",
        "- Approach 3 (Agent AI): `presentation/presentation_agent.pptx`",
        "",
        "## PDF Reports",
        "- Approach 1: `reports/report_ml.pdf`",
        "- Approach 2: `reports/report_dl.pdf`",
        "- Approach 3: `reports/report_agent.pdf`",
        "- Research paper (IEEE): `reports/research_paper.pdf`",
        "",
        "## Evaluation & Visualisations",
        "- `evaluation/ml_evaluation.md`",
        "- `evaluation/dl_evaluation.md`",
        "- `evaluation/agent_evaluation.md`",
        "- `visualizations/` (ML, DL plots)",
        "- `assets/` (AI-generated imagery)",
        "",
        "## Reproduce",
        "```bash",
        "pip install -r requirements.txt",
        "python run_ml_pipeline.py",
        "python run_dl_pipeline.py",
        "python run_agent_pipeline.py",
        "python make_deliverables.py",
        "```",
    ]
    (ROOT / "DELIVERABLES.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote DELIVERABLES.md")


def main() -> None:
    run_shap(ROOT / "visualizations" / "ml")
    build_ppt(ROOT / "presentation")
    build_pdf(ROOT / "reports")
    build_paper(ROOT / "reports")
    build_index(ROOT / "evaluation", ROOT / "reports", ROOT / "presentation")
    logger.info("All deliverables generated.")


if __name__ == "__main__":
    main()
