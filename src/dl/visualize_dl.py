"""Visualisation generation for the Deep Learning approach."""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.manifold import TSNE

from src.utils import ensure_dir, setup_logging

logger = setup_logging()


def _save(fig, out_dir, name):
    ensure_dir(out_dir)
    fig.tight_layout()
    fig.savefig(out_dir / name, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(results, out_dir):
    """Loss and F2 curves over epochs for all architectures."""
    n = len(results)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    axes = np.array(axes).flatten()
    for ax, r in zip(axes, results):
        h = r["history"]
        if not h["train_loss"]:
            ax.axis("off"); continue
        ax.plot(h["train_loss"], label="train loss")
        ax.plot(h["val_loss"], label="val loss")
        ax2 = ax.twinx()
        ax2.plot(h["val_f2"], "--", color="green", label="val F2")
        ax.set_title(r["name"]); ax.legend(fontsize=6, loc="upper right")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("DL Training Dynamics", y=1.02)
    _save(fig, out_dir, "dl_learning_curves.png")


def plot_roc_pr(results, out_dir):
    """ROC and PR curves overlay for all DL architectures."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for r in results:
        yt = np.asarray(r["y_true"]); pr = np.asarray(r["prob"])
        fpr, tpr, _ = roc_curve(yt, pr)
        prec, rec, _ = precision_recall_curve(yt, pr)
        axes[0].plot(fpr, tpr, lw=1.2, label=f"{r['name']} ({r['metrics'].get('roc_auc',0):.3f})")
        axes[1].plot(rec, prec, lw=1.2, label=f"{r['name']} ({r['metrics'].get('pr_auc',0):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--"); axes[0].set_title("DL ROC Curves")
    axes[1].set_title("DL PR Curves")
    axes[0].legend(fontsize=6, loc="lower right"); axes[1].legend(fontsize=6, loc="lower left")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    _save(fig, out_dir, "dl_roc_pr_curves.png")


def plot_metric_compare_ml(dl_results, ml_eval_path, out_dir):
    """Compare DL best against the Approach-1 ML baseline."""
    import re
    try:
        txt = open(ml_eval_path).read()
    except Exception:
        return
    # parse best ML F2 from ranking table (approx)
    best_ml = None
    for line in txt.splitlines():
        if "F2=" in line and line.strip()[0].isdigit():
            m = re.search(r"F2=([0-9.]+)", line)
            if m:
                best_ml = float(m.group(1)); break
    if best_ml is None:
        return
    names = [r["name"] for r in dl_results]
    f2s = [r["metrics"]["f2"] for r in dl_results]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(names, f2s, color="#16a085")
    ax.axhline(best_ml, color="#c0392b", ls="--", lw=2, label=f"Best ML baseline F2={best_ml:.3f}")
    ax.set_ylabel("F2 Score"); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title("DL Architectures vs Traditional ML Baseline (F2)")
    ax.legend()
    _save(fig, out_dir, "dl_vs_ml_baseline.png")


def generate_dl_plots(results, best_dl, out_dir):
    ensure_dir(out_dir)
    plot_learning_curves(results, out_dir)
    plot_roc_pr(results, out_dir)
    try:
        from src.utils import ROOT
        plot_metric_compare_ml(results, ROOT / "evaluation" / "ml_evaluation.md", out_dir)
    except Exception as e:
        logger.warning("Skipping ML comparison plot: %s", e)
    logger.info("DL visualisations generated in %s", out_dir)
