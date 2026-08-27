"""Feature engineering for the fraud detection dataset.

Creates domain-specific, temporal, interaction and statistical features that
help the models separate fraudulent claims from legitimate ones. All features
are derived from a single claim row plus cross-feature interactions (the raw
dataset has one claim per patient/provider, so no longitudinal aggregation is
possible; we document this limitation explicitly).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils import setup_logging

logger = setup_logging()


def add_temporal_features(df: pd.DataFrame, date_col: str = "ClaimDate") -> pd.DataFrame:
    """Add calendar/time-based features derived from the claim date.

    Args:
        df: Input data.
        date_col: Name of the datetime claim-date column.

    Returns:
        pd.DataFrame: Data with added temporal features.
    """
    d = pd.to_datetime(df[date_col])
    df = df.copy()
    df["ClaimYear"] = d.dt.year
    df["ClaimMonth"] = d.dt.month
    df["ClaimDayOfWeek"] = d.dt.dayofweek
    df["ClaimDayOfYear"] = d.dt.dayofyear
    df["ClaimIsWeekend"] = (d.dt.dayofweek >= 5).astype(int)
    # seasonality (Indian context): 1=winter,2=summer,3=monsoon,4=post-monsoon
    df["ClaimSeason"] = np.select(
        [d.dt.month.isin([12, 1, 2]), d.dt.month.isin([3, 4, 5]),
         d.dt.month.isin([6, 7, 8])],
        [1, 2, 3], default=4,
    )
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific ratio / interaction features.

    Args:
        df: Input data with ClaimAmount, PatientIncome, PatientAge present.

    Returns:
        pd.DataFrame: Data with added ratio features.
    """
    df = df.copy()
    df["ClaimToIncome"] = df["ClaimAmount"] / (df["PatientIncome"] + 1e-6)
    df["ClaimAmountLog"] = np.log1p(df["ClaimAmount"])
    df["IncomeLog"] = np.log1p(df["PatientIncome"])
    # age group in Indian context
    bins = [-1, 17, 30, 45, 60, 100]
    labels = ["Child", "YoungAdult", "Adult", "Senior", "Elderly"]
    df["AgeGroup"] = pd.cut(df["PatientAge"], bins=bins, labels=labels)
    df["AgeGroup"] = df["AgeGroup"].astype(str)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple numeric interaction/polynomial features.

    Args:
        df: Input data.

    Returns:
        pd.DataFrame: Data with interaction features.
    """
    df = df.copy()
    df["AmountPerAge"] = df["ClaimAmount"] / (df["PatientAge"] + 1e-6)
    df["AmountPerIncome"] = df["ClaimAmount"] / (df["PatientIncome"] + 1e-6)
    df["AgeXIncome"] = df["PatientAge"] * df["PatientIncome"]
    df["AmountSq"] = df["ClaimAmount"] ** 2
    df["AgeSq"] = df["PatientAge"] ** 2
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering routine.

    Args:
        df: Raw dataset.

    Returns:
        pd.DataFrame: Dataset with engineered features.
    """
    out = df.copy()
    if "ClaimDate" in out.columns:
        out = add_temporal_features(out)
        out = out.drop(columns=["ClaimDate"])
    out = add_ratio_features(out)
    out = add_interaction_features(out)
    logger.info("Feature engineering complete: %d columns", out.shape[1])
    return out
