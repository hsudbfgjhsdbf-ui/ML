"""Threshold selection utilities."""
import numpy as np
from typing import Tuple
from sklearn.metrics import f1_score, fbeta_score, precision_recall_curve

def select_threshold(y_true, y_prob, strategy: str = "optimize_f2", recall_target: float = 0.8) -> Tuple[float, dict]:
    """Select best threshold based on strategy."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds length = precisions length -1
    best_thr = 0.5
    info = {}
    if strategy == "optimize_f1":
        f1s = []
        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(int)
            f1s.append(f1_score(y_true, y_pred, zero_division=0))
        idx = int(np.argmax(f1s))
        best_thr = float(thresholds[idx]) if len(thresholds)>0 else 0.5
        info["best_f1"] = float(f1s[idx]) if f1s else 0.0
    elif strategy == "optimize_f2":
        f2s = []
        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(int)
            f2s.append(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
        idx = int(np.argmax(f2s))
        best_thr = float(thresholds[idx]) if len(thresholds)>0 else 0.5
        info["best_f2"] = float(f2s[idx]) if f2s else 0.0
    elif strategy == "pr_auc_recall_target":
        # Find threshold where recall >= target with max precision
        feasible = [(p,r,t) for p,r,t in zip(precisions[:-1], recalls[:-1], thresholds) if r >= recall_target]
        if feasible:
            # max precision among feasible
            feasible_sorted = sorted(feasible, key=lambda x: x[0], reverse=True)
            best_thr = float(feasible_sorted[0][2])
            info["precision_at_target_recall"] = float(feasible_sorted[0][0])
        else:
            best_thr = 0.5
    else:
        best_thr = 0.5
    
    info["strategy"] = strategy
    info["threshold"] = best_thr
    # Ensure threshold in [0.05, 0.95]
    best_thr = float(np.clip(best_thr, 0.05, 0.95))
    info["threshold_clipped"] = best_thr
    return best_thr, info
