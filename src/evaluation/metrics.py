"""Canonical classification metrics, threshold selection, and uncertainty tools."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ThresholdResult:
    """Validation threshold selected without looking at the test partition."""

    threshold: float
    metric_name: str
    metric_value: float
    precision: float
    recall: float
    f1: float
    f2: float


def probabilities_from_estimator(estimator: Any, x: np.ndarray) -> np.ndarray:
    """Return clipped fraud probabilities from either probability or score APIs.

    Args:
        estimator: Fitted estimator implementing ``predict_proba`` or
            ``decision_function``.
        x: Numeric feature matrix.
    Returns:
        One-dimensional probabilities in the closed interval [0, 1].
    Raises:
        AttributeError: If the estimator exposes neither supported API.
    """
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(x))[:, 1]
    elif hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(x), dtype=float)
        scores = np.clip(scores, -60.0, 60.0)
        probabilities = 1.0 / (1.0 + np.exp(-scores))
    else:
        raise AttributeError(f"{type(estimator).__name__} has no probability or score method")
    return np.clip(probabilities.astype(float), 1e-7, 1.0 - 1e-7)


def threshold_predictions(probabilities: Iterable[float], threshold: float) -> np.ndarray:
    """Convert probabilities to the canonical fraud-positive labels."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between zero and one")
    return (np.asarray(list(probabilities), dtype=float) >= threshold).astype(int)


def compute_metrics(y_true: Iterable[int], probabilities: Iterable[float], threshold: float = 0.5) -> dict[str, Any]:
    """Compute the complete canonical metric suite for one operating point.

    Args:
        y_true: Binary labels where one is fraud.
        probabilities: Fraud probabilities.
        threshold: Decision threshold used for discrete metrics.
    Returns:
        JSON-friendly metric dictionary including the confusion matrix.
    """
    actual = np.asarray(list(y_true), dtype=int)
    probs = np.clip(np.asarray(list(probabilities), dtype=float), 1e-7, 1.0 - 1e-7)
    if len(actual) != len(probs):
        raise ValueError("y_true and probabilities must have equal length")
    predicted = threshold_predictions(probs, threshold)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "f2": float(fbeta_score(actual, predicted, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probs)) if len(np.unique(actual)) == 2 else 0.5,
        "pr_auc": float(average_precision_score(actual, probs)) if actual.sum() else 0.0,
        "mcc": float(matthews_corrcoef(actual, predicted)),
        "cohen_kappa": float(cohen_kappa_score(actual, predicted)),
        "brier": float(brier_score_loss(actual, probs)),
        "log_loss": float(log_loss(actual, probs, labels=[0, 1])),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "support_negative": int((actual == 0).sum()),
        "support_positive": int((actual == 1).sum()),
    }


def threshold_sweep(y_true: Iterable[int], probabilities: Iterable[float], points: int = 99) -> pd.DataFrame:
    """Evaluate a deterministic threshold grid for validation analysis.

    Args:
        y_true: Binary validation labels.
        probabilities: Validation fraud probabilities.
        points: Number of thresholds between 0.01 and 0.99 inclusive.
    Returns:
        Dataframe with precision, recall, F1, and F2 for each threshold.
    """
    if points < 3:
        raise ValueError("threshold grid requires at least three points")
    rows = []
    for threshold in np.linspace(0.01, 0.99, points):
        values = compute_metrics(y_true, probabilities, float(threshold))
        rows.append(
            {
                key: values[key]
                for key in [
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "f2",
                    "specificity",
                    "false_positive",
                    "false_negative",
                ]
            }
        )
    return pd.DataFrame(rows)


def select_threshold(
    y_true: Iterable[int], probabilities: Iterable[float], points: int = 99, precision_floor: float = 0.50
) -> tuple[ThresholdResult, pd.DataFrame]:
    """Select a validation-only threshold by F2 with a precision floor.

    The precision constraint avoids the degenerate result of maximizing recall
    at a threshold that flags nearly every claim. If no candidate satisfies the
    floor, the unconstrained F2 maximum is used and the fallback is recorded.

    Args:
        y_true: Validation labels.
        probabilities: Validation probabilities.
        points: Number of threshold candidates.
        precision_floor: Minimum preferred fraud precision.
    Returns:
        Selected ThresholdResult and complete sweep table.
    """
    sweep = threshold_sweep(y_true, probabilities, points)
    constrained = sweep[sweep["precision"] >= precision_floor]
    candidates = constrained if not constrained.empty else sweep
    best = candidates.sort_values(["f2", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    return ThresholdResult(
        threshold=float(best["threshold"]),
        metric_name="f2_with_precision_floor" if not constrained.empty else "f2_fallback_no_precision_candidate",
        metric_value=float(best["f2"]),
        precision=float(best["precision"]),
        recall=float(best["recall"]),
        f1=float(best["f1"]),
        f2=float(best["f2"]),
    ), sweep


def bootstrap_intervals(
    y_true: Iterable[int], probabilities: Iterable[float], threshold: float, seed: int = 42, replicates: int = 300
) -> dict[str, dict[str, float]]:
    """Estimate percentile confidence intervals for headline metrics.

    Args:
        y_true: Test labels.
        probabilities: Frozen-model test probabilities.
        threshold: Validation-selected threshold.
        seed: Bootstrap RNG seed.
        replicates: Number of resamples.
    Returns:
        Mapping metric name to estimate, 2.5th percentile, and 97.5th percentile.
    """
    actual = np.asarray(list(y_true), dtype=int)
    probs = np.asarray(list(probabilities), dtype=float)
    rng = np.random.default_rng(seed)
    keys = ["accuracy", "precision", "recall", "f1", "f2", "roc_auc", "pr_auc", "mcc", "brier"]
    samples = {key: [] for key in keys}
    for _ in range(max(20, replicates)):
        indices = rng.integers(0, len(actual), len(actual))
        if len(np.unique(actual[indices])) < 2:
            continue
        metrics = compute_metrics(actual[indices], probs[indices], threshold)
        for key in keys:
            samples[key].append(metrics[key])
    original = compute_metrics(actual, probs, threshold)
    return {
        key: {
            "estimate": float(original[key]),
            "lower_2_5": float(np.percentile(values, 2.5)) if values else float(original[key]),
            "upper_97_5": float(np.percentile(values, 97.5)) if values else float(original[key]),
        }
        for key, values in samples.items()
    }


def fairness_metrics(
    frame: pd.DataFrame,
    labels: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
    columns: list[str],
    min_slice_size: int = 20,
) -> pd.DataFrame:
    """Compute fraud-catch and false-alarm metrics by demographic slice.

    Sensitive columns are used only for auditing, never as model inputs in the
    shipped feature matrix.

    Args:
        frame: Raw rows aligned with labels and probabilities.
        labels: Binary ground truth.
        probabilities: Fraud probabilities.
        threshold: Operating threshold selected on validation data.
        columns: Slice columns to evaluate.
        min_slice_size: Minimum row count for a stable slice annotation.
    Returns:
        Long-form slice metric table with stability flags.
    """
    actual = np.asarray(list(labels), dtype=int)
    probs = np.asarray(list(probabilities), dtype=float)
    predicted = threshold_predictions(probs, threshold)
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in frame:
            continue
        for value, positions in frame.groupby(column, dropna=False).groups.items():
            indices = np.asarray(list(positions), dtype=int)
            y = actual[indices]
            p = predicted[indices]
            matrix = confusion_matrix(y, p, labels=[0, 1])
            tn, fp, fn, tp = matrix.ravel()
            rows.append(
                {
                    "slice_column": column,
                    "slice_value": str(value),
                    "n": int(len(indices)),
                    "positive_n": int(y.sum()),
                    "stable_for_comparison": bool(len(indices) >= min_slice_size and y.sum() >= 2),
                    "tpr_recall": float(tp / (tp + fn)) if tp + fn else math.nan,
                    "fpr": float(fp / (fp + tn)) if fp + tn else math.nan,
                    "precision": float(precision_score(y, p, zero_division=0)),
                    "accuracy": float(accuracy_score(y, p)),
                    "predicted_fraud_rate": float(p.mean()) if len(p) else math.nan,
                }
            )
    return pd.DataFrame(rows)
