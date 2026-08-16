"""
Utility Functions and Statistical Evaluation Engine.
Provides logging, metrics computation, statistical significance testing,
cost matrix calculation (in INR), and threshold optimization.
"""

import logging
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
from src.config import (
    config, COST_FALSE_NEGATIVE_INR, COST_FALSE_POSITIVE_INR,
    COST_TRUE_POSITIVE_SAVING_INR
)

# Configure comprehensive logger
def setup_logger(name: str = "FraudDetectionSystem", log_file: Optional[Path] = None) -> logging.Logger:
    """Sets up a structured logger writing to console and persistent log file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        # Console handler (INFO and above)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler (DEBUG and above)
        target_file = log_file or config.log_file
        fh = logging.FileHandler(target_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

logger = setup_logger()

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes exhaustive classification metrics including F2-score, MCC, and PR-AUC.
    """
    y_true = np.asarray(y_true, dtype=int)
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred, dtype=int)
        y_prob = y_pred.astype(float)
        
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float(np.mean(y_true))
        
    brier = float(brier_score_loss(y_true, y_prob))
    
    # Financial Cost Analysis in INR
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    financial_loss_inr = float(fn * COST_FALSE_NEGATIVE_INR + fp * COST_FALSE_POSITIVE_INR)
    fraud_savings_inr = float(tp * COST_TRUE_POSITIVE_SAVING_INR)
    
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "f2_score": round(f2, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "mcc": round(mcc, 4),
        "brier_score": round(brier, 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "financial_loss_inr": round(financial_loss_inr, 2),
        "fraud_savings_inr": round(fraud_savings_inr, 2),
        "threshold": round(threshold, 3)
    }

def find_optimal_threshold_f2(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_steps: int = 100,
    min_precision: float = 0.65
) -> Tuple[float, float, Dict[str, float]]:
    """
    Optimizes classification decision threshold to maximize F2-score (prioritizing fraud recall)
    while maintaining healthy precision balance (>= min_precision).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    
    thresholds = np.linspace(0.15, 0.85, num_steps)
    best_f2 = -1.0
    best_thresh = 0.5
    
    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        if prec >= min_precision or best_f2 < 0:
            f2 = fbeta_score(y_true, preds, beta=2.0, zero_division=0)
            if f2 > best_f2:
                best_f2 = f2
                best_thresh = float(th)
                
    best_metrics = compute_all_metrics(y_true, None, y_prob, threshold=best_thresh)
    return best_thresh, best_f2, best_metrics

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray
) -> Dict[str, Any]:
    """
    Executes McNemar's Test with continuity correction for pairwise model comparison.
    H0: Classifiers A and B have the same error rate.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_a = np.asarray(y_pred_a, dtype=int)
    y_pred_b = np.asarray(y_pred_b, dtype=int)
    
    # a_correct: Model A correct, b_correct: Model B correct
    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)
    
    n00 = int(np.sum(~a_correct & ~b_correct)) # both wrong
    n01 = int(np.sum(~a_correct & b_correct))  # A wrong, B right (b)
    n10 = int(np.sum(a_correct & ~b_correct))  # A right, B wrong (c)
    n11 = int(np.sum(a_correct & b_correct))   # both right
    
    b = n01
    c = n10
    
    if (b + c) == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        # Edwards continuity correction
        statistic = float(((abs(b - c) - 1.0) ** 2) / (b + c))
        p_value = float(1.0 - stats.chi2.cdf(statistic, df=1))
        
    significant = p_value < 0.05
    return {
        "contingency_table": [[n11, n10], [n01, n00]],
        "statistic": round(statistic, 4),
        "p_value": round(p_value, 6),
        "statistically_significant": significant,
        "interpretation": "Significant difference" if significant else "No significant difference"
    }

def wilcoxon_signed_rank_test(
    scores_a: List[float],
    scores_b: List[float]
) -> Dict[str, Any]:
    """
    Conducts Wilcoxon Signed-Rank Test across cross-validation folds.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diff = a - b
    
    if np.all(diff == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "statistically_significant": False,
            "mean_diff": 0.0
        }
        
    try:
        res = stats.wilcoxon(a, b, zero_method="pratt")
        stat = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception as e:
        stat, p_val = 0.0, 1.0
        
    return {
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "statistically_significant": p_val < 0.05,
        "mean_score_a": round(float(np.mean(a)), 4),
        "mean_score_b": round(float(np.mean(b)), 4),
        "mean_difference": round(float(np.mean(diff)), 4)
    }

def save_model_artifacts(
    model: Any,
    name: str,
    metrics: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    save_dir: Path = config.raw_data_path.parent.parent / "saved_models"
) -> Path:
    """Serializes model weights, metrics, and hyperparameter metadata."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model_file = save_dir / f"{name}.joblib"
    meta_file = save_dir / f"{name}_metadata.json"
    
    joblib.dump(model, model_file)
    
    full_meta = {
        "model_name": name,
        "metrics": metrics,
        "metadata": metadata or {}
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(full_meta, f, indent=2, default=str)
        
    logger.info(f"Saved model '{name}' to {model_file} with F2={metrics.get('f2_score', 0):.4f}")
    return model_file
