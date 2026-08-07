"""Approach 2 data contract adapter.

This module reuses Approach 1's validated loader, feature engineering, split,
and train-only transformer. The resulting matrices are float32 tensors so all
five deep architectures receive exactly the same rows and target semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.loading import load_claims, profile_dataset
from src.features.engineering import engineer_features
from src.features.preprocessing import fit_transform_matrices, stratified_three_way_split
from src.utils.reproducibility import sha256_file, stable_json_hash


@dataclass(frozen=True)
class DeepDataBundle:
    """Frozen numeric matrices and metadata consumed by the deep run."""

    raw: pd.DataFrame
    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    validation_audit: pd.DataFrame
    test_audit: pd.DataFrame
    feature_names: list[str]
    feature_lineage: pd.DataFrame
    input_sha256: str
    split_fingerprint: str
    feature_fingerprint: str
    profile: dict[str, Any]


def load_deep_data(root: Path, config: dict[str, Any]) -> DeepDataBundle:
    """Load and transform the same supplied data used by Approach 1.

    Args:
        root: Repository root.
        config: Deep configuration dictionary.
    Returns:
        DeepDataBundle with float matrices and audit frames.
    Raises:
        RuntimeError: If persisted Approach 1 split membership disagrees.
    """
    source = root / "data" / "raw" / "health_insurance_fraud_claims.xlsx"
    raw = load_claims(source)
    engineered_result = engineer_features(raw)
    target = raw["ClaimLegitimacy"].eq("Fraud").astype(int)
    split = stratified_three_way_split(
        engineered_result.features,
        target,
        seed=int(config["seed"]),
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
    )
    persisted = root / "evaluation" / "split_membership.csv"
    if config.get("splitting", {}).get("require_approach1_membership", True) and persisted.exists():
        membership = pd.read_csv(persisted)
        expected = pd.DataFrame(
            {
                "split": ["train"] * len(split.train_indices)
                + ["validation"] * len(split.validation_indices)
                + ["test"] * len(split.test_indices),
                "row_index": np.concatenate([split.train_indices, split.validation_indices, split.test_indices]),
            }
        )
        if not membership.equals(expected):
            raise RuntimeError("Approach 2 split membership does not match Approach 1's frozen split artifact")
    matrices = fit_transform_matrices(split)
    split_payload = {
        "train": split.train_indices.tolist(),
        "validation": split.validation_indices.tolist(),
        "test": split.test_indices.tolist(),
    }
    split_fingerprint = stable_json_hash(split_payload)
    feature_fingerprint = stable_json_hash(matrices.feature_names)
    validation_audit = raw.loc[split.validation_indices].reset_index(drop=True).copy()
    test_audit = raw.loc[split.test_indices].reset_index(drop=True).copy()
    for frame in [validation_audit, test_audit]:
        frame["age_band"] = pd.cut(
            frame["PatientAge"],
            bins=[-np.inf, 17, 30, 45, 60, 75, np.inf],
            labels=["0_17", "18_30", "31_45", "46_60", "61_75", "76_plus"],
        ).astype(str)
    return DeepDataBundle(
        raw=raw,
        x_train=matrices.x_train.astype(np.float32),
        x_validation=matrices.x_validation.astype(np.float32),
        x_test=matrices.x_test.astype(np.float32),
        y_train=split.y_train.to_numpy(dtype=np.float32),
        y_validation=split.y_validation.to_numpy(dtype=np.float32),
        y_test=split.y_test.to_numpy(dtype=np.float32),
        validation_audit=validation_audit,
        test_audit=test_audit,
        feature_names=matrices.feature_names,
        feature_lineage=engineered_result.lineage,
        input_sha256=sha256_file(source),
        split_fingerprint=split_fingerprint,
        feature_fingerprint=feature_fingerprint,
        profile=profile_dataset(raw).as_dict(),
    )


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    """Create a deterministic DataLoader for one split.

    Args:
        x: Float32 feature matrix.
        y: Float32 binary labels.
        batch_size: Batch size.
        shuffle: Whether training order is shuffled.
        seed: Generator seed.
    Returns:
        PyTorch DataLoader yielding feature and label tensors.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0, generator=generator
    )
