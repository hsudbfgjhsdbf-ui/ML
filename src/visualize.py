"""Visualisation generation for the Traditional ML approach.

Produces all plots required by the spec: dataset distributions, correlation
analysis, ROC/PR curves, feature importance, model comparison, radar charts
and data-relationship scatter plots. Consistent colour scheme: fraud = red,
legitimate = blue/green.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from src.utils import ensure_dir, setup_logging

logger = setup_logging()

FRAUD_C = "#c0392b"
LEGIT_C = "#2980b9"

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 9})


def _save(fig, out_dir: Path, name: str) -> None:
    ensure_dir(out_dir)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot %s", path)


def plot_class_distribution(raw: pd.DataFrame, out_dir: Path) -> None:
    """Pie + bar of class distribution and categorical distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = raw["ClaimLegitimacy"].value_counts()
    axes[0].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=[LEGIT_C, FRAUD_C], startangle=90)
    axes[0].set_title("Claim Class Distribution")
    # fraud rate by claim type
    ct = pd.crosstab(raw["ClaimType"], raw["ClaimLegitimacy"], normalize="index")
    ct.plot(kind="barh", ax=axes[1], color=[LEGIT_C, FRAUD_C])
    axes[1].set_title("Fraud Rate by Claim Type")
    axes[1].legend(title="Class")
    _save(fig, out_dir, "01_class_distribution.png")


def plot_numeric_distributions(raw: pd.DataFrame, num_cols, out_dir: Path) -> None:
    """Histograms of numeric features by class."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()
    for ax, c in zip(axes, num_cols[:6]):
        for cls, color in [("Fraud", FRAUD_C), ("Legitimate", LEGIT_C)]:
            vals = raw.loc[raw["ClaimLegitimacy"] == cls, c]
            ax.hist(vals, bins=30, alpha=0.5, color=color, label=cls)
        ax.set_title(c)
        ax.legend(fontsize=7)
    fig.suptitle("Numeric Feature Distributions by Class", y=1.02)
    _save(fig, out_dir, "02_numeric_distributions.png")


def plot_correlation_heatmap(data, out_dir: Path) -> None:
    """Correlation heatmap of numeric features."""
    df = data["X_test"]
    num = df.select_dtypes(include=[np.number])
    corr = num.corr().abs()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Numeric Feature Correlation Heatmap")
    _save(fig, out_dir, "03_correlation_heatmap.png")


def plot_roc_curves(results, out_dir: Path) -> None:
    """Overlay ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        y_true, y_prob = np.asarray(r.y_true), np.asarray(r.probabilities)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, lw=1.5,
                label=f"{r.name} (AUC={r.metrics.get('roc_auc', 0):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - All Algorithms")
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, out_dir, "04_roc_curves.png")


def plot_pr_curves(results, out_dir: Path) -> None:
    """Overlay precision-recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        y_true, y_prob = np.asarray(r.y_true), np.asarray(r.probabilities)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(rec, prec, lw=1.5,
                label=f"{r.name} (AP={r.metrics.get('pr_auc', 0):.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves - All Algorithms")
    ax.legend(fontsize=7, loc="lower left")
    _save(fig, out_dir, "05_pr_curves.png")


def plot_metric_comparison(results, out_dir: Path) -> None:
    """Grouped bar chart comparing key metrics across models."""
    metrics = ["precision", "recall", "f1", "f2", "roc_auc"]
    fig, ax = plt.subplots(figsize=(11, 6))
    names = [r.name for r in results]
    x = np.arange(len(names)); width = 0.15
    for i, m in enumerate(metrics):
        vals = [r.metrics.get(m, 0) for r in results]
        ax.bar(x + (i - 2) * width, vals, width, label=m)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison by Metric")
    ax.legend(ncol=5, fontsize=8)
    _save(fig, out_dir, "06_metric_comparison.png")


def plot_radar(results, out_dir: Path, top_n: int = 5) -> None:
    """Radar chart for the top-N models across metrics."""
    top = results[:top_n]
    metrics = ["precision", "recall", "f1", "f2", "roc_auc", "pr_auc"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for r in top:
        vals = [r.metrics.get(m, 0) for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, lw=2, label=r.name)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1); ax.set_title("Performance Profile - Top 5 Models")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.15), fontsize=8)
    _save(fig, out_dir, "07_radar_chart.png")


def plot_feature_importance(data, out_dir: Path) -> None:
    """Bar chart of top features by mutual information."""
    imp = data["mi_importance"].head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.sort_values().plot(kind="barh", ax=ax, color="#16a085")
    ax.set_title("Top 20 Features by Mutual Information")
    ax.set_xlabel("Mutual Information")
    _save(fig, out_dir, "08_feature_importance.png")


def plot_confusion_heatmaps(results, out_dir: Path, top_n: int = 6) -> None:
    """Confusion matrix heatmaps for the top-N models."""
    n = min(top_n, len(results))
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 6))
    axes = np.array(axes).flatten()
    for ax, r in zip(axes, results[:n]):
        cm = r.confusion_matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"],
                    cbar=False)
        ax.set_title(f"{r.name}\nThr={r.optimal_threshold:.2f}")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Confusion Matrices (Test Set)", y=1.02)
    _save(fig, out_dir, "09_confusion_matrices.png")


def plot_train_time_vs_accuracy(results, out_dir: Path) -> None:
    """Scatter of training time vs accuracy."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        ax.scatter(r.train_time, r.metrics["accuracy"], s=80)
        ax.annotate(r.name, (r.train_time, r.metrics["accuracy"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Training Time (s)"); ax.set_ylabel("Accuracy")
    ax.set_title("Training Time vs Accuracy Trade-off")
    _save(fig, out_dir, "10_train_time_vs_accuracy.png")


def generate_all_ml_plots(data, results, impact_rows, out_dir: Path) -> None:
    """Generate every ML visualisation into `out_dir`."""
    out_dir = ensure_dir(out_dir)
    raw = pd.read_excel("data/raw/Health Insurance Fraud Claims.xlsx")
    plot_class_distribution(raw, out_dir)
    plot_numeric_distributions(raw, ["ClaimAmount", "PatientAge", "PatientIncome"], out_dir)
    plot_correlation_heatmap(data, out_dir)
    plot_roc_curves(results, out_dir)
    plot_pr_curves(results, out_dir)
    plot_metric_comparison(results, out_dir)
    plot_radar(results, out_dir)
    plot_feature_importance(data, out_dir)
    plot_confusion_heatmaps(results, out_dir)
    plot_train_time_vs_accuracy(results, out_dir)
    logger.info("All ML visualisations generated in %s", out_dir)
