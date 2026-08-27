"""Helpers to read pipeline results and format them for deliverables."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import pandas as pd

from src.utils import ROOT, setup_logging

logger = setup_logging()

# Project / institutional metadata (used across all deliverables).
PROJECT = {
    "title": "Medical Insurance Claim Fraud Detection",
    "subtitle": "A Multi-Approach AI System for the Indian Healthcare & Insurance Ecosystem",
    "institution": "IIIT Dharwad",
    "department": "Department of Data Science and AI",
    "adviser": "Ramesh Athe",
    "team": ["B Varshith", "M Jagadeshwar", "J Ganesh"],
    "date": "2026",
}


def load_ml_results() -> list[dict]:
    """Load Approach-1 ML results (ranked by F2) from models/ml/results.pkl."""
    p = ROOT / "models" / "ml" / "results.pkl"
    if not p.exists():
        # fall back to cache json
        cp = ROOT / "models" / "ml" / "results_cache.json"
        if cp.exists():
            entries = json.loads(cp.read_text())
            return sorted(entries, key=lambda r: r["metrics"]["f2"], reverse=True)
        return []
    with open(p, "rb") as f:
        obj = pickle.load(f)
    results = obj["results"]
    rows = []
    for r in results:
        rows.append({
            "name": r.name, "metrics": r.metrics,
            "train_time": r.train_time, "pred_time_ms": r.pred_time_ms,
            "model_size_kb": r.model_size_kb, "n_hyperparams": r.n_hyperparams,
            "threshold": r.optimal_threshold,
        })
    return sorted(rows, key=lambda r: r["metrics"]["f2"], reverse=True)


def load_dl_results() -> list[dict]:
    """Load Approach-2 DL results (ranked by F2) from models/dl/results.json."""
    p = ROOT / "models" / "dl" / "results.json"
    if not p.exists():
        return []
    rows = json.loads(p.read_text())
    return sorted(rows, key=lambda r: r["metrics"]["f2"], reverse=True)


def load_agent_metrics() -> dict | None:
    """Load Approach-3 agent aggregate metrics from evaluation/agent_evaluation.md."""
    p = ROOT / "evaluation" / "agent_evaluation.md"
    if not p.exists():
        return None
    text = p.read_text()
    try:
        row = next(line for line in text.splitlines() if line.startswith("| "))
        parts = [x.strip() for x in row.split("|")[1:-1]]
        return {"accuracy": float(parts[0]), "precision": float(parts[1]),
                "recall": float(parts[2]), "f2": float(parts[3])}
    except Exception:
        return None


def fmt(m):
    """Safely format a metric value."""
    return f"{m:.3f}" if isinstance(m, float) else f"{m}"


def metric_table(rows, key_fn=None):
    """Build a markdown table string from result rows."""
    if not rows:
        return "_(results pending — run the corresponding pipeline first)_"
    lines = ["| Model | Acc | Prec | Recall | F1 | F2 | AUC-ROC | MCC |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r["metrics"]
        lines.append(
            f"| {r['name']} | {fmt(m.get('accuracy'))} | {fmt(m.get('precision'))} | "
            f"{fmt(m.get('recall'))} | {fmt(m.get('f1'))} | **{fmt(m.get('f2'))}** | "
            f"{fmt(m.get('roc_auc', '-'))} | {fmt(m.get('mcc'))} |")
    return "\n".join(lines)
