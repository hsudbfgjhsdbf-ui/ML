#!/usr/bin/env python
"""Approach 3 - Agent AI multi-agent system pipeline.

Builds the reference knowledge base from the claims data, runs the multi-agent
workflow on the held-out claims, scores it against ground truth, and produces
an explainable decision report plus the evaluation markdown.

Usage:
    python run_agent_pipeline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.agent.core import Coordinator, ReferenceDatabase
from src.data_loader import load_raw_data
from src.utils import ensure_dir, load_config, resolve, setup_logging

logger = setup_logging()


def run(cfg_path: str = "config/config.yaml") -> None:
    cfg = load_config(cfg_path)
    eval_dir = resolve("evaluation")
    reports_dir = resolve("reports")
    ensure_dir(eval_dir); ensure_dir(reports_dir)

    df = load_raw_data()
    s = cfg["data"]["split"]
    y = (df["ClaimLegitimacy"] == "Fraud").astype(int).values
    _, df_test = train_test_split(df, test_size=s["test"], stratify=y,
                                  random_state=s["random_state"])
    _, y_test = train_test_split(y, test_size=s["test"], stratify=y,
                                 random_state=s["random_state"])

    # reference DB built on full data (typical-cost baselines)
    db = ReferenceDatabase(df)
    coordinator = Coordinator(db, verbose=False)

    claims = df_test.to_dict("records")
    report = coordinator.process_batch(claims, labels=y_test)

    # ---- write evaluation markdown ----------------------------------------
    metrics = report["metrics"]
    lines = [
        "# Approach 3 - Agent AI Multi-Agent System Report\n",
        "## Overview", "",
        "- Coordinator orchestrates 5 specialised agents (eligibility, policy,",
        "  anomaly, historical, reasoning).",
        "- Each agent emits structured findings with confidence & evidence.",
        "- Reasoning agent synthesises findings into Approved/Flagged/Rejected",
        "  verdict with a natural-language explanation.",
        f"- Tested on {len(claims)} held-out claims.\n",
        "## Aggregate Performance", "",
        f"| Accuracy | Precision | Recall | F2 |",
        f"|---|---|---|---|",
        f"| {metrics['accuracy']:.3f} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f2']:.3f} |",
        "",
        "## Verdict Distribution", "",
    ]
    vc = pd.Series([r["state"].verdict for r in report["results"]]).value_counts()
    for v, c in vc.items():
        lines.append(f"- {v}: {c}")
    (eval_dir / "agent_evaluation.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- sample explainable decision ---------------------------------------
    sample = report["results"][0]["state"]
    with open(reports_dir / "agent_sample_decision.json", "w") as f:
        json.dump({
            "claim_id": sample.claim.get("ClaimID"),
            "verdict": sample.verdict,
            "risk_score": sample.risk_score,
            "explanation": sample.explanation,
            "findings": sample.findings,
        }, f, indent=2, default=str)

    logger.info("Agent pipeline complete. F2=%.3f | Recall=%.3f",
                metrics["f2"], metrics["recall"])


if __name__ == "__main__":
    run()
