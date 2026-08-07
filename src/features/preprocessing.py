"""Leakage-safe preprocessing and stratified data splitting utilities.

All learned imputation, category, and scaling state is fitted on the training
partition through a scikit-learn pipeline. Validation and test rows are only
transformed with that frozen state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class SplitData:
    """Raw feature and target partitions with original row identifiers."""

    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray


@dataclass(frozen=True)
class MatrixBundle:
    """Transformed matrices, fitted transformer, and feature names."""

    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    transformer: ColumnTransformer
    feature_names: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]


def encode_target(labels: pd.Series, positive_label: str = "Fraud") -> pd.Series:
    """Map the supplied target to the canonical fraud-positive convention.

    Args:
        labels: Text labels containing Fraud and Legitimate.
        positive_label: Label mapped to one.
    Returns:
        Integer series with 1 meaning fraud and 0 meaning legitimate.
    Raises:
        ValueError: If an unexpected label appears.
    """
    normalized = labels.astype(str).str.strip().str.title()
    allowed = {positive_label.title(), "Legitimate"}
    unexpected = sorted(set(normalized).difference(allowed))
    if unexpected:
        raise ValueError(f"Unexpected target labels: {unexpected}")
    return normalized.eq(positive_label.title()).astype(int)


def stratified_three_way_split(
    features: pd.DataFrame,
    target: pd.Series,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> SplitData:
    """Create deterministic 70/15/15 stratified partitions.

    Args:
        features: Engineered, target-free features.
        target: Binary fraud target aligned to ``features``.
        seed: Random seed used for both split operations.
        train_fraction: Fraction assigned to training.
        validation_fraction: Fraction assigned to validation.
        test_fraction: Fraction assigned to test.
    Returns:
        SplitData preserving original dataframe indices.
    Raises:
        ValueError: If fractions or class counts are invalid.
    """
    total = train_fraction + validation_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError("train, validation, and test fractions must sum to one")
    if len(features) != len(target) or target.nunique() < 2:
        raise ValueError("Features and binary target must be aligned with two classes")
    indices = np.asarray(features.index)
    train_idx, remainder_idx = train_test_split(
        indices,
        test_size=(validation_fraction + test_fraction),
        stratify=target.loc[indices],
        random_state=seed,
    )
    remainder_target = target.loc[remainder_idx]
    test_share_of_remainder = test_fraction / (validation_fraction + test_fraction)
    validation_idx, test_idx = train_test_split(
        remainder_idx,
        test_size=test_share_of_remainder,
        stratify=remainder_target,
        random_state=seed,
    )
    return SplitData(
        x_train=features.loc[train_idx].copy(),
        x_validation=features.loc[validation_idx].copy(),
        x_test=features.loc[test_idx].copy(),
        y_train=target.loc[train_idx].copy(),
        y_validation=target.loc[validation_idx].copy(),
        y_test=target.loc[test_idx].copy(),
        train_indices=np.asarray(train_idx),
        validation_indices=np.asarray(validation_idx),
        test_indices=np.asarray(test_idx),
    )


def build_preprocessor(features: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build the common imputation, encoding, and scaling transformer.

    Args:
        features: Training feature dataframe.
    Returns:
        Tuple of transformer, numeric column names, categorical column names.
    """
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return transformer, numeric, categorical


def fit_transform_matrices(split: SplitData) -> MatrixBundle:
    """Fit preprocessing on training rows and transform all three partitions.

    Args:
        split: Raw feature partitions from ``stratified_three_way_split``.
    Returns:
        MatrixBundle with dense finite matrices and feature names.
    Raises:
        ValueError: If a transformed matrix contains non-finite values.
    """
    transformer, numeric, categorical = build_preprocessor(split.x_train)
    x_train = transformer.fit_transform(split.x_train)
    x_validation = transformer.transform(split.x_validation)
    x_test = transformer.transform(split.x_test)
    matrices = [np.asarray(x, dtype=np.float64) for x in [x_train, x_validation, x_test]]
    if any(not np.isfinite(matrix).all() for matrix in matrices):
        raise ValueError("Preprocessing produced NaN or infinite model inputs")
    try:
        names = transformer.get_feature_names_out().tolist()
    except AttributeError:  # pragma: no cover - compatibility fallback
        names = [f"feature_{index}" for index in range(matrices[0].shape[1])]
    return MatrixBundle(
        x_train=matrices[0],
        x_validation=matrices[1],
        x_test=matrices[2],
        transformer=transformer,
        feature_names=names,
        numeric_columns=numeric,
        categorical_columns=categorical,
    )


def simple_smote(x: np.ndarray, y: np.ndarray, seed: int, target_ratio: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Create a lightweight, dependency-free SMOTE-like training comparison.

    This comparison is intentionally separate from the default class-weighted
    experiments. It interpolates minority pairs and is never applied to
    validation or test rows.

    Args:
        x: Numeric training matrix.
        y: Binary training labels.
        seed: Reproducible interpolation seed.
        target_ratio: Desired minority proportion after synthesis.
    Returns:
        Resampled matrix and labels.
    Raises:
        ValueError: If input labels do not contain both classes.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    minority = x[y == 1]
    majority = x[y == 0]
    if len(minority) < 2 or len(majority) == 0:
        raise ValueError("SMOTE comparison requires at least two minority and one majority row")
    desired_minority = int(np.ceil(target_ratio * len(majority) / (1.0 - target_ratio)))
    needed = max(0, desired_minority - len(minority))
    left = rng.integers(0, len(minority), size=needed)
    right = rng.integers(0, len(minority), size=needed)
    weights = rng.random(needed)[:, None]
    synthetic = minority[left] + weights * (minority[right] - minority[left]) if needed else np.empty((0, x.shape[1]))
    output_x = np.vstack([x, synthetic])
    output_y = np.concatenate([y, np.ones(len(synthetic), dtype=int)])
    order = rng.permutation(len(output_y))
    return output_x[order], output_y[order]
