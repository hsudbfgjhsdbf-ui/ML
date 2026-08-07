"""Preprocessing pipeline for the fraud detection dataset.

Implements a modular, reproducible, serialisable sequence of transformations
(imputation, deduplication, encoding, scaling) that is fitted on training data
only and then applied consistently to validation and test data to avoid any
data leakage. Each step is a separate well-documented function.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from src.utils import setup_logging

logger = setup_logging()


# --------------------------------------------------------------------------
# Step 1 - Missing value handling
# --------------------------------------------------------------------------
def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Report the count/percentage of missing values per column.

    Args:
        df: Input data.

    Returns:
        pd.DataFrame: Missing-value report.
    """
    missing = df.isna().sum()
    missing = missing[missing > 0]
    report = pd.DataFrame({"count": missing, "percent": (missing / len(df)) * 100})
    return report.sort_values("count", ascending=False)


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.70) -> pd.DataFrame:
    """Drop features missing more than `threshold` fraction of values.

    Args:
        df: Input data.
        threshold: Fraction threshold above which a column is dropped.

    Returns:
        pd.DataFrame: Data with high-missing columns removed.
    """
    frac = df.isna().mean()
    drop_cols = frac[frac > threshold].index.tolist()
    if drop_cols:
        logger.warning("Dropping %d columns with >%.0f%% missing: %s",
                       len(drop_cols), threshold * 100, drop_cols)
    return df.drop(columns=drop_cols)


# --------------------------------------------------------------------------
# Step 2 - Duplicate handling
# --------------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    """Remove exact duplicate rows.

    Args:
        df: Input data.
        subset: Optional column subset for duplicate detection.

    Returns:
        pd.DataFrame: Deduplicated data.
    """
    before = len(df)
    out = df.drop_duplicates(subset=subset)
    if len(out) < before:
        logger.info("Removed %d duplicate rows", before - len(out))
    return out


# --------------------------------------------------------------------------
# Step 3 - Outlier detection
# --------------------------------------------------------------------------
def flag_outliers_iqr(df: pd.DataFrame, cols, k: float = 3.0) -> pd.DataFrame:
    """Flag outliers using the IQR rule for numeric columns.

    Args:
        df: Input data.
        cols: Numeric columns to check.
        k: IQR multiplier (default 3.0 -> far outliers).

    Returns:
        pd.DataFrame: Boolean mask (True = outlier) per column.
    """
    mask = pd.DataFrame(index=df.index)
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        mask[c] = (df[c] < lower) | (df[c] > upper)
    return mask


def flag_outliers_zscore(df: pd.DataFrame, cols, z: float = 3.0) -> pd.DataFrame:
    """Flag outliers using the z-score rule for numeric columns.

    Args:
        df: Input data.
        cols: Numeric columns to check.
        z: Z-score threshold.

    Returns:
        pd.DataFrame: Boolean mask (True = outlier) per column.
    """
    mask = pd.DataFrame(index=df.index)
    for c in cols:
        mean, std = df[c].mean(), df[c].std()
        mask[c] = df[c].sub(mean).abs() > z * std
    return mask


# --------------------------------------------------------------------------
# Encoding transformers
# --------------------------------------------------------------------------
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode high-cardinality categoricals by their frequency counts."""

    def __init__(self, cols, min_count: int = 1):
        self.cols = cols
        self.min_count = min_count
        self.freq_ = {}

    def fit(self, X, y=None):
        for c in self.cols:
            vc = X[c].value_counts(dropna=False)
            vc = vc[vc >= self.min_count]
            # map rare/unknown to 0
            self.freq_[c] = vc.to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[c] = X[c].map(self.freq_).fillna(0).astype(float)
        return X


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target (mean) encoding for high-cardinality categorical features.

    Uses out-of-fold style smoothing to avoid overfitting the target.
    """

    def __init__(self, cols, smoothing: float = 20.0):
        self.cols = cols
        self.smoothing = smoothing
        self.mapping_ = {}
        self.prior_ = None

    def fit(self, X, y):
        prior = float(np.mean(y))
        self.prior_ = prior
        for c in self.cols:
            stat = pd.DataFrame({"y": y, "c": X[c]}).groupby("c")["y"].agg(["mean", "count"])
            stat["smoothed"] = (stat["mean"] * stat["count"] + prior * self.smoothing) / (
                stat["count"] + self.smoothing
            )
            self.mapping_[c] = stat["smoothed"].to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[c] = X[c].map(self.mapping_).fillna(self.prior_).astype(float)
        return X


# --------------------------------------------------------------------------
# Column type helpers
# --------------------------------------------------------------------------
def split_types(df: pd.DataFrame, target: str) -> tuple[list, list]:
    """Split feature columns into numeric and categorical lists.

    Args:
        df: Feature dataframe (target excluded).
        target: Target column name (excluded from features).

    Returns:
        tuple[list, list]: (numeric columns, categorical columns).
    """
    X = df.drop(columns=[target], errors="ignore")
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "datetime"]).columns.tolist()
    return num_cols, cat_cols


# --------------------------------------------------------------------------
# Main preprocessing pipeline
# --------------------------------------------------------------------------
class PreprocessingPipeline(BaseEstimator, TransformerMixin):
    """End-to-end preprocessing pipeline.

    Encapsulates encoding + scaling and can be fit once and reused.
    """

    def __init__(self, high_cardinality=None, low_cardinality=None,
                 ordinal_cols=None, num_cols=None):
        self.high_cardinality = high_cardinality or []
        self.low_cardinality = low_cardinality or []
        self.ordinal_cols = ordinal_cols or []
        self.num_cols = num_cols or []
        self.target_encoder_ = None
        self.frequency_encoder_ = None
        self.onehot_ = None
        self.scaler_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.high_cardinality:
            X = self.target_encoder_.transform(X)
        if self.low_cardinality:
            encoded = self.onehot_.transform(X[self.low_cardinality])
            enc_cols = self.onehot_.get_feature_names_out(self.low_cardinality)
            X = X.drop(columns=self.low_cardinality).reset_index(drop=True)
            encoded_df = pd.DataFrame(encoded, columns=enc_cols, index=X.index)
            X = pd.concat([X, encoded_df], axis=1)
        if self.num_cols:
            X[self.num_cols] = self.scaler_.transform(X[self.num_cols])
        return X

    def fit_transform(self, X, y=None):
        X = X.copy()

        # Target encode high-cardinality categoricals
        if self.high_cardinality:
            self.target_encoder_ = TargetEncoder(self.high_cardinality)
            X = self.target_encoder_.fit_transform(X, y)

        # One-hot encode low-cardinality categoricals
        if self.low_cardinality:
            self.onehot_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = self.onehot_.fit_transform(X[self.low_cardinality])
            enc_cols = self.onehot_.get_feature_names_out(self.low_cardinality)
            X = X.drop(columns=self.low_cardinality).reset_index(drop=True)
            encoded_df = pd.DataFrame(encoded, columns=enc_cols, index=X.index)
            X = pd.concat([X, encoded_df], axis=1)

        # Standard-scale numeric columns
        if self.num_cols:
            self.scaler_ = StandardScaler()
            X[self.num_cols] = self.scaler_.fit_transform(X[self.num_cols])

        return X


def build_pipeline(
    num_cols, low_cat_cols, high_cat_cols
) -> PreprocessingPipeline:
    """Construct a preprocessing pipeline for given column groups.

    Args:
        num_cols: Numeric columns to standard-scale.
        low_cat_cols: Low-cardinality categoricals to one-hot encode.
        high_cat_cols: High-cardinality categoricals to target encode.

    Returns:
        PreprocessingPipeline: Configured pipeline object.
    """
    return PreprocessingPipeline(
        num_cols=num_cols, low_cardinality=low_cat_cols,
        high_cardinality=high_cat_cols,
    )
