"""Model-agnostic explainability, faithfulness, and stability for deep models.

SHAP and LIME remain optional extensions. The reference run uses deterministic
feature occlusion on the shared transformed matrix, which is dependency-light,
works for all five architectures, and gives directly comparable evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from src_dl.evaluation import predict_probabilities


def occlusion_importance(
    model: torch.nn.Module, x: np.ndarray, feature_names: list[str], device: torch.device, sample_size: int = 250
) -> pd.DataFrame:
    """Measure mean absolute probability change after zeroing each feature.

    Args:
        model: Fitted deep classifier.
        x: Validation feature matrix; zero is the scaled training mean.
        feature_names: Frozen transformed feature names.
        device: Inference device.
        sample_size: Maximum rows used for the importance estimate.
    Returns:
        Tidy feature-importance table with rank and signed direction.
    """
    sample = x[: min(sample_size, len(x))].copy()
    base = predict_probabilities(model, sample, device)
    rows = []
    for index, name in enumerate(feature_names):
        perturbed = sample.copy()
        perturbed[:, index] = 0.0
        changed = predict_probabilities(model, perturbed, device)
        delta = base - changed
        rows.append(
            {"feature": name, "importance": float(np.mean(np.abs(delta))), "signed_delta": float(np.mean(delta))}
        )
    frame = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame


def faithfulness_score(
    model: torch.nn.Module,
    x: np.ndarray,
    importance: pd.DataFrame,
    device: torch.device,
    ks: tuple[int, ...] = (1, 5, 10),
    sample_size: int = 250,
) -> dict[str, Any]:
    """Measure whether deleting top features changes the model score more than random deletion."""
    sample = x[: min(sample_size, len(x))].copy()
    base = predict_probabilities(model, sample, device)
    feature_to_index = {name: index for index, name in enumerate(importance["feature"])}
    top_scores = {}
    random_scores = {}
    rng = np.random.default_rng(42)
    for k in ks:
        top_names = importance.head(min(k, len(importance)))["feature"].tolist()
        top_indices = [feature_to_index[name] for name in top_names]
        top_x = sample.copy()
        top_x[:, top_indices] = 0.0
        random_x = sample.copy()
        random_indices = rng.choice(sample.shape[1], size=len(top_indices), replace=False)
        random_x[:, random_indices] = 0.0
        top_scores[str(k)] = float(np.mean(base - predict_probabilities(model, top_x, device)))
        random_scores[str(k)] = float(np.mean(base - predict_probabilities(model, random_x, device)))
    return {
        "top_deletion_score": top_scores,
        "random_deletion_score": random_scores,
        "faithfulness_at_5": top_scores.get("5", 0.0) - random_scores.get("5", 0.0),
    }


def stability_score(
    model: torch.nn.Module,
    x: np.ndarray,
    importance: pd.DataFrame,
    device: torch.device,
    sample_size: int = 250,
    repetitions: int = 3,
) -> dict[str, Any]:
    """Estimate top-10 ranking stability under small feature jitter."""
    reference = set(importance.head(10)["feature"])
    rankings = []
    for repetition in range(repetitions):
        rng = np.random.default_rng(100 + repetition)
        jittered = x[: min(sample_size, len(x))].copy()
        jittered += rng.normal(0.0, 0.01, size=jittered.shape).astype(np.float32)
        ranking = occlusion_importance(
            model, jittered, list(importance["feature"]), device, sample_size=sample_size
        ).head(10)["feature"]
        rankings.append(set(ranking))
    overlaps = []
    for ranking in rankings:
        union = reference | ranking
        overlaps.append(len(reference & ranking) / len(union) if union else 1.0)
    return {
        "top_k": 10,
        "repetitions": repetitions,
        "jaccard_mean": float(np.mean(overlaps)),
        "jaccard_values": overlaps,
    }


def local_dossier(
    model: torch.nn.Module,
    x: np.ndarray,
    feature_names: list[str],
    importance: pd.DataFrame,
    device: torch.device,
    row_index: int = 0,
) -> dict[str, Any]:
    """Create one neutral local explanation dossier for a validation row."""
    row = x[row_index : row_index + 1].copy()
    base = float(predict_probabilities(model, row, device)[0])
    entries = []
    for _, item in importance.head(10).iterrows():
        index = feature_names.index(item["feature"])
        altered = row.copy()
        altered[:, index] = 0.0
        changed = float(predict_probabilities(model, altered, device)[0])
        entries.append(
            {
                "feature": item["feature"],
                "occlusion_probability_change": base - changed,
                "global_rank": int(item["rank"]),
            }
        )
    entries.sort(key=lambda item: abs(item["occlusion_probability_change"]), reverse=True)
    return {
        "row_index": row_index,
        "base_probability": base,
        "drivers": entries[:5],
        "language": "Signals are associations used to prioritize review; they do not establish claimant intent.",
    }
