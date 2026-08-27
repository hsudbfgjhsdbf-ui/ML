"""Evaluation utilities: metrics, threshold optimisation, statistical tests.

Implements the full evaluation methodology: comprehensive metrics for an
imbalanced dataset, confusion-matrix-based business impact, optimal-threshold
selection maximising F2, McNemar's test, Wilcoxon signed-rank test and
bootstrap confidence intervals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    fbeta_score, matthews_corrcoef, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

from src.utils import setup_logging

logger = setup_logging()


@dataclass
class EvalResult:
    """Container for the full evaluation of a single model."""

    name: str
    metrics: dict = field(default_factory=dict)
    confusion_matrix: np.ndarray | None = None
    optimal_threshold: float = 0.5
    probabilities: np.ndarray | None = None
    y_true: np.ndarray | None = None
    fpr: np.ndarray | None = None
    tpr: np.ndarray | None = None
    roc_auc: float | None = None
    precision_curve: np.ndarray | None = None
    recall_curve: np.ndarray | None = None
    avg_precision: float | None = None
    train_time: float = 0.0
    pred_time_ms: float = 0.0
    model_size_kb: float = 0.0
    n_hyperparams: int = 0
    model: object = None  # fitted estimator


def compute_metrics(y_true, y_pred, y_prob=None, beta: float = 2.0) -> dict:
    """Compute the full set of primary evaluation metrics.

    Args:
        y_true: Ground-truth labels (binary, 1 = fraud).
        y_pred: Predicted labels.
        y_prob: Predicted fraud probabilities (optional).
        beta: F-beta weighting (2 emphasises recall).

    Returns:
        dict: Computed metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"f{int(beta)}": float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return metrics


def optimal_threshold_f2(y_true, y_prob, beta: float = 2.0) -> tuple[float, float]:
    """Find the probability threshold that maximises the F2 score.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted probabilities.
        beta: F-beta parameter.

    Returns:
        tuple: (best threshold, best F2 score).
    """
    thresholds = np.linspace(0.01, 0.99, 199)
    best_t, best_f = 0.5, -1
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t), float(best_f)


def business_impact(cm: np.ndarray, avg_claim: float) -> dict:
    """Quantify business impact in INR from the confusion matrix.

    Args:
        cm: 2x2 confusion matrix [[TN, FP], [FN, TP]].
        avg_claim: Average legitimate claim amount used to estimate losses.

    Returns:
        dict: Impact estimates.
    """
    tn, fp, fn, tp = cm.ravel()
    # Each false negative = fraudulent claim approved -> direct financial loss
    est_loss = fn * avg_claim
    # Each false positive = legitimate claim wrongly rejected -> customer/regulatory cost
    est_customer_cost = fp * (avg_claim * 0.25)  # estimated processing/regulatory overhead
    return {
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
        "fraud_loss_inr": float(est_loss),
        "customer_cost_inr": float(est_customer_cost),
    }


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot: int = 1000, seed: int = 42) -> tuple:
    """Compute a 95% bootstrap confidence interval for a metric.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted probabilities.
        metric_fn: Function (y_true, y_prob) -> float.
        n_boot: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        tuple: (mean, lower, upper).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(metric_fn(y_true[idx], y_prob[idx]))
        except Exception:
            continue
    if not vals:
        return 0.0, 0.0, 0.0
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar_test(y_true, pred_a, pred_b) -> tuple[float, float]:
    """McNemar's test for paired model comparison on the same test set.

    Args:
        y_true: Ground-truth labels.
        pred_a: Predictions of model A.
        pred_b: Predictions of model B.

    Returns:
        tuple: (chi2 statistic, p-value).
    """
    diff_a = (pred_a != y_true).astype(int)
    diff_b = (pred_b != y_true).astype(int)
    b = int(((diff_a == 1) & (diff_b == 0)).sum())   # A wrong, B right
    c = int(((diff_a == 0) & (diff_b == 1)).sum())   # A right, B wrong
    chi2 = (abs(b - c) - 1) ** 2 / (b + c + 1e-9)
    p = sp_stats.chi2.sf(chi2, 1)
    return float(chi2), float(p)


def wilcoxon_test(scores_a, scores_b) -> tuple[float, float]:
    """Wilcoxon signed-rank test for paired cross-validation score lists.

    Args:
        scores_a: Fold scores of model A.
        scores_b: Fold scores of model B.

    Returns:
        tuple: (statistic, p-value).
    """
    stat, p = sp_stats.wilcoxon(scores_a, scores_b)
    return float(stat), float(p)
