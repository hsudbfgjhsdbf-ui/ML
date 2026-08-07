"""Evaluation helpers for deep models using Approach 1's metric canon."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import bootstrap_intervals, compute_metrics, fairness_metrics, select_threshold


def predict_probabilities(
    model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 1024
) -> np.ndarray:
    """Run deterministic sigmoid inference over a NumPy matrix.

    Args:
        model: Trained deep classifier returning logits.
        x: Float32 input matrix.
        device: CPU or CUDA device.
        batch_size: Inference batch size.
    Returns:
        One-dimensional NumPy fraud probabilities.
    """
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            output = model(batch)
            logits = output[0] if isinstance(output, tuple) else output
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.empty(0, dtype=float)


def evaluate_probabilities(
    y_true: np.ndarray, probabilities: np.ndarray, points: int = 99, precision_floor: float = 0.5
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select a validation threshold and compute the complete metric suite."""
    threshold, sweep = select_threshold(
        y_true.astype(int), probabilities, points=points, precision_floor=precision_floor
    )
    metrics = compute_metrics(y_true.astype(int), probabilities, threshold.threshold)
    return {**metrics, "threshold_metric": threshold.metric_name}, sweep


def evaluate_fairness(
    audit_frame: pd.DataFrame, y_true: np.ndarray, probabilities: np.ndarray, threshold: float, min_slice_size: int
) -> pd.DataFrame:
    """Compute the same demographic audit slices used by Approach 1."""
    return fairness_metrics(
        audit_frame,
        y_true.astype(int),
        probabilities,
        threshold,
        ["PatientGender", "age_band", "ClaimType", "PatientEmploymentStatus"],
        min_slice_size,
    )


def confidence_intervals(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float, seed: int, replicates: int
) -> dict[str, Any]:
    """Return bootstrap intervals for the final deep-model test output."""
    return bootstrap_intervals(y_true.astype(int), probabilities, threshold, seed=seed, replicates=replicates)
