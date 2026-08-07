"""Training, comparison, calibration, and XAI figures for Approach 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

NAVY = "#102A43"
TEAL = "#2A9D8F"
ORANGE = "#E76F51"
GOLD = "#E9C46A"
SLATE = "#486581"


def _save(fig: plt.Figure, path: Path) -> str:
    """Save a figure at publication resolution and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_learning_curves(telemetry: list[pd.DataFrame], key: str, output: Path) -> str:
    """Plot mean train/validation loss and validation PR-AUC over epochs."""
    frame = pd.concat(telemetry, ignore_index=True)
    grouped = (
        frame.groupby("epoch")
        .agg(
            train_loss=("train_loss", "mean"),
            validation_loss=("validation_loss", "mean"),
            pr_auc=("validation_pr_auc", "mean"),
            pr_std=("validation_pr_auc", "std"),
        )
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(grouped.epoch, grouped.train_loss, label="Train loss", color=TEAL)
    axis.plot(grouped.epoch, grouped.validation_loss, label="Validation loss", color=ORANGE)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("BCE / reconstruction loss")
    axis.set_title(f"Training dynamics — {key}")
    axis.legend(loc="upper left")
    right = axis.twinx()
    right.plot(grouped.epoch, grouped.pr_auc, color=NAVY, label="Validation PR-AUC")
    right.set_ylabel("Validation PR-AUC")
    return _save(fig, output)


def plot_deep_leaderboard(frame: pd.DataFrame, output: Path) -> str:
    """Render mean validation PR-AUC and F2 by architecture."""
    ordered = frame.sort_values("val_pr_auc", ascending=True)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y - 0.18, ordered.val_pr_auc, height=0.34, color=TEAL, label="PR-AUC")
    ax.barh(y + 0.18, ordered.val_f2, height=0.34, color=ORANGE, label="F2")
    ax.set_yticks(y, ordered.display_name)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Validation score")
    ax.set_title("Deep-learning validation comparison")
    ax.legend()
    return _save(fig, output)


def plot_curves(rows: list[dict[str, Any]], output_dir: Path) -> tuple[str, str]:
    """Overlay validation ROC and PR curves across deep models."""
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    colors = sns.color_palette("tab10", len(rows))
    for color, row in zip(colors, rows):
        fpr, tpr, _ = roc_curve(row["y_validation"], row["validation_probabilities"])
        recall, precision, _ = precision_recall_curve(row["y_validation"], row["validation_probabilities"])
        ax1.plot(fpr, tpr, label=f"{row['key']} ({row['val_roc_auc']:.3f})", color=color)
        ax2.plot(recall, precision, label=f"{row['key']} ({row['val_pr_auc']:.3f})", color=color)
    for ax, title, xlabel, ylabel in [
        (ax1, "Deep validation ROC curves", "False-positive rate", "True-positive rate"),
        (ax2, "Deep validation precision-recall curves", "Recall", "Precision"),
    ]:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    return _save(fig1, output_dir / "roc_curves_validation.png"), _save(fig2, output_dir / "pr_curves_validation.png")


def plot_xai(importance: pd.DataFrame, key: str, output: Path) -> str:
    """Render top occlusion features."""
    table = importance.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(table.feature, table.importance, color=ORANGE)
    ax.set_xlabel("Mean probability change after occlusion")
    ax.set_title(f"Occlusion importance — {key}")
    return _save(fig, output)


def plot_calibration(y: np.ndarray, before: np.ndarray, after: np.ndarray, output: Path) -> str:
    """Render validation reliability curves before and after isotonic mapping."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for probs, label, color in [(before, "Before", SLATE), (after, "After", ORANGE)]:
        observed, predicted = calibration_curve(y, probs, n_bins=10, strategy="uniform")
        ax.plot(predicted, observed, "o-", label=label, color=color)
    ax.plot([0, 1], [0, 1], "--", color="#999999", label="Perfect")
    ax.set_title("Deep-model reliability diagram")
    ax.set_xlabel("Predicted fraud probability")
    ax.set_ylabel("Observed fraud frequency")
    ax.legend()
    return _save(fig, output)
