"""
Utility functions for Medical Insurance Claim Fraud Detection System.
Institution: IIIT Dharwad, Department of Data Science and AI
Faculty Adviser: Prof. Ramesh Athe
Team Members: B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)

This module provides common utilities for:
- Configuration management
- Logging setup
- Directory lifecycle management
- Indian currency (INR) formatting
- Statistical significance tests (McNemar's Test, Wilcoxon Signed-Rank Test)
- Bootstrap confidence intervals
"""

import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple, List


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Loads the master YAML configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_logger(name: str = "FraudDetectionLogger", log_file: str = "project_execution.log", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a dedicated logger for tracking execution steps.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def ensure_directories(dirs: List[str]) -> None:
    """
    Ensures that all specified directories exist.
    """
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def format_inr(amount: float) -> str:
    """
    Formats a numeric amount into Indian Rupee (INR / Rs.) format with Indian comma separators.
    Example: 250000 -> Rs. 2,50,000.00
    """
    try:
        amount_str = f"{amount:.2f}"
        parts = amount_str.split(".")
        integer_part = parts[0]
        decimal_part = parts[1]
        
        is_negative = False
        if integer_part.startswith("-"):
            is_negative = True
            integer_part = integer_part[1:]
            
        if len(integer_part) <= 3:
            formatted = integer_part
        else:
            last_three = integer_part[-3:]
            remaining = integer_part[:-3]
            groups = []
            while len(remaining) > 2:
                groups.append(remaining[-2:])
                remaining = remaining[:-2]
            if remaining:
                groups.append(remaining)
            groups.reverse()
            formatted = ",".join(groups) + "," + last_three
            
        res = f"Rs. {formatted}.{decimal_part}"
        return f"-{res}" if is_negative else res
    except Exception:
        return f"Rs. {amount:.2f}"


def calculate_mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Tuple[float, float, bool]:
    """
    Performs pairwise McNemar's test for statistical significance between two classification models.
    Returns: (chi2_statistic, p_value, is_significant_at_05)
    """
    b = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    c = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    
    if (b + c) == 0:
        return 0.0, 1.0, False
        
    chi2 = ((abs(b - c) - 1.0) ** 2) / float(b + c)
    p_value = float(stats.chi2.sf(chi2, 1))
    is_significant = p_value < 0.05
    return float(chi2), p_value, is_significant


def calculate_wilcoxon_test(scores1: List[float], scores2: List[float]) -> Tuple[float, float, bool]:
    """
    Performs Wilcoxon signed-rank test on cross-validation fold scores between two models.
    Returns: (statistic, p_value, is_significant_at_05)
    """
    try:
        stat, p_val = stats.wilcoxon(scores1, scores2)
        return float(stat), float(p_val), (p_val < 0.05)
    except Exception:
        return 0.0, 1.0, False


def bootstrap_metric_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_fn, n_bootstrap: int = 500, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes a bootstrap confidence interval for a given evaluation metric function.
    Returns: (mean_val, lower_bound, upper_bound)
    """
    scores = []
    n_samples = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        try:
            score = metric_fn(y_true[idx], y_pred[idx])
            scores.append(score)
        except Exception:
            continue
    if not scores:
        return 0.0, 0.0, 0.0
    scores = np.array(scores)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(scores, alpha * 100))
    upper = float(np.percentile(scores, (1.0 - alpha) * 100))
    mean_val = float(np.mean(scores))
    return mean_val, lower, upper
