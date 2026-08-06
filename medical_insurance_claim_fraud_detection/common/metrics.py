"""Metrics for imbalanced fraud detection."""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, confusion_matrix, brier_score_loss,
    precision_recall_curve, roc_curve
)

def compute_all_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    """Compute comprehensive metrics."""
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["specificity"] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics["sensitivity"] = metrics["recall"]
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["f2"] = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    except Exception:
        metrics["pr_auc"] = float("nan")
    try:
        metrics["brier_score"] = brier_score_loss(y_true, y_prob)
    except Exception:
        metrics["brier_score"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(set(y_true))>1 else (0,0,0,0)
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    metrics["fpr"] = float(fp / (fp+tn)) if (fp+tn)>0 else 0.0
    metrics["fnr"] = float(fn / (fn+tp)) if (fn+tp)>0 else 0.0
    return metrics

def precision_at_k(y_true, y_scores, k: int) -> float:
    """Precision at top k."""
    if k <=0:
        return 0.0
    order = np.argsort(y_scores)[::-1]
    topk = order[:k]
    return float(np.mean(np.array(y_true)[topk]))

def recall_at_k(y_true, y_scores, k: int) -> float:
    order = np.argsort(y_scores)[::-1]
    topk = order[:k]
    y_true_arr = np.array(y_true)
    total_pos = y_true_arr.sum()
    if total_pos == 0:
        return 0.0
    return float(y_true_arr[topk].sum() / total_pos)

def threshold_analysis(y_true, y_prob, thresholds: List[float] = None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17).tolist()
    rows = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        m = compute_all_metrics(y_true, y_pred, y_prob)
        rows.append({"threshold": thr, **m})
    return pd.DataFrame(rows)

def calibration_curve_data(y_true, y_prob, n_bins: int = 10):
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return prob_true, prob_pred
