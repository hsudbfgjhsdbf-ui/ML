"""Domain-aware, leakage-conscious feature engineering for claim rows.

The supplied workbook has one record per claim and almost-unique identifiers.
Identifiers, near-unique codes, provider location strings, and post-decision
status are excluded from the model matrix. Date and amount signals are turned
into features that would be available when a claim is screened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureEngineeringResult:
    """Feature matrix and lineage information produced from raw claims."""

    features: pd.DataFrame
    lineage: pd.DataFrame
    dropped_columns: list[str]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Compute a stable ratio, returning zero when a denominator is zero.

    Args:
        numerator: Numerator values.
        denominator: Denominator values.
    Returns:
        Ratio series with finite values.
    """
    denominator_safe = denominator.replace(0, np.nan)
    result = numerator / denominator_safe
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _age_band(age: pd.Series) -> pd.Series:
    """Map age into stable, interpretable Indian insurance age bands."""
    return pd.cut(
        age,
        bins=[-np.inf, 17, 30, 45, 60, 75, np.inf],
        labels=["0_17", "18_30", "31_45", "46_60", "61_75", "76_plus"],
    ).astype(str)


def engineer_features(frame: pd.DataFrame) -> FeatureEngineeringResult:
    """Create screening-time features and a feature-lineage table.

    Args:
        frame: Validated raw claims dataframe containing ``ClaimLegitimacy``.
    Returns:
        FeatureEngineeringResult with model inputs, lineage, and exclusions.
    Raises:
        ValueError: If required source columns are absent.
    """
    required = {
        "ClaimAmount",
        "ClaimDate",
        "PatientAge",
        "PatientIncome",
        "Cluster",
        "PatientGender",
        "ProviderSpecialty",
        "PatientMaritalStatus",
        "PatientEmploymentStatus",
        "ClaimType",
        "ClaimSubmissionMethod",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Cannot engineer features; missing source columns: {missing}")

    source = frame.copy()
    result = pd.DataFrame(index=source.index)
    lineage: list[dict[str, Any]] = []

    def add(name: str, values: pd.Series, source_name: str, transform: str, rationale: str) -> None:
        """Add one feature and its provenance row."""
        result[name] = values
        lineage.append(
            {
                "feature": name,
                "source": source_name,
                "transform": transform,
                "rationale": rationale,
                "decision_time_available": True,
            }
        )

    numeric = {
        "claim_amount_inr": (
            source["ClaimAmount"].astype(float),
            "ClaimAmount",
            "renamed",
            "Claim value is the primary financial signal.",
        ),
        "patient_age_years": (
            source["PatientAge"].astype(float),
            "PatientAge",
            "renamed",
            "Age supports demographic slice auditing and risk context.",
        ),
        "patient_income_inr": (
            source["PatientIncome"].astype(float),
            "PatientIncome",
            "renamed",
            "Income contextualizes the requested amount.",
        ),
        "cluster_code": (
            source["Cluster"].astype(float),
            "Cluster",
            "numeric category code",
            "Supplied cluster is treated as a source feature, not a label.",
        ),
    }
    for name, (values, source_name, transform, rationale) in numeric.items():
        add(name, values, source_name, transform, rationale)

    claim_amount = source["ClaimAmount"].astype(float)
    income = source["PatientIncome"].astype(float)
    add(
        "log_claim_amount",
        np.log1p(claim_amount),
        "ClaimAmount",
        "log1p",
        "Reduces the effect of right-skewed claim values.",
    )
    add(
        "log_patient_income",
        np.log1p(income),
        "PatientIncome",
        "log1p",
        "Stabilizes income scale for linear and distance models.",
    )
    add(
        "claim_to_income_ratio",
        _safe_ratio(claim_amount, income),
        "ClaimAmount, PatientIncome",
        "safe ratio",
        "High claim relative to income can prioritize review.",
    )
    add(
        "claim_minus_income_scaled",
        _safe_ratio(claim_amount - income, income.abs() + 1.0),
        "ClaimAmount, PatientIncome",
        "scaled difference",
        "Captures financial mismatch without unbounded subtraction.",
    )
    add(
        "claim_amount_per_age",
        _safe_ratio(claim_amount, source["PatientAge"].astype(float) + 1.0),
        "ClaimAmount, PatientAge",
        "safe ratio",
        "Captures amount-age interaction without dividing by zero.",
    )
    add(
        "income_age_interaction",
        income / (source["PatientAge"].astype(float) + 1.0),
        "PatientIncome, PatientAge",
        "safe ratio",
        "Represents income context by age.",
    )

    claim_date = pd.to_datetime(source["ClaimDate"], errors="coerce")
    add("claim_year", claim_date.dt.year.astype(float), "ClaimDate", "calendar year", "Captures broad temporal drift.")
    add(
        "claim_month",
        claim_date.dt.month.astype(float),
        "ClaimDate",
        "calendar month",
        "Captures seasonal claim patterns.",
    )
    add(
        "claim_day_of_week",
        claim_date.dt.dayofweek.astype(float),
        "ClaimDate",
        "weekday index",
        "Captures operational timing patterns.",
    )
    add(
        "claim_day_of_month",
        claim_date.dt.day.astype(float),
        "ClaimDate",
        "day of month",
        "Captures billing-cycle timing.",
    )
    add(
        "claim_is_weekend",
        claim_date.dt.dayofweek.ge(5).astype(float),
        "ClaimDate",
        "binary calendar flag",
        "Weekend activity is an auditable process signal.",
    )
    add(
        "claim_month_sin",
        np.sin(2 * np.pi * claim_date.dt.month / 12.0),
        "ClaimDate",
        "cyclic month encoding",
        "Preserves December-to-January continuity.",
    )
    add(
        "claim_month_cos",
        np.cos(2 * np.pi * claim_date.dt.month / 12.0),
        "ClaimDate",
        "cyclic month encoding",
        "Preserves seasonal phase information.",
    )

    add(
        "age_band",
        _age_band(source["PatientAge"]),
        "PatientAge",
        "ordinal bins",
        "Provides interpretable age-group audits.",
    )
    add(
        "cluster_category",
        source["Cluster"].astype(str),
        "Cluster",
        "nominal category",
        "Allows nonlinear cluster effects without ordinal assumptions.",
    )

    categorical_columns = [
        "PatientGender",
        "ProviderSpecialty",
        "PatientMaritalStatus",
        "PatientEmploymentStatus",
        "ClaimType",
        "ClaimSubmissionMethod",
    ]
    for column in categorical_columns:
        add(
            column.lower(),
            source[column].astype(str).str.strip().replace("nan", "unknown"),
            column,
            "normalized category",
            "Retains source context with unknown-category safety.",
        )

    dropped = [
        "ClaimID",
        "PatientID",
        "ProviderID",
        "DiagnosisCode",
        "ProcedureCode",
        "ProviderLocation",
        "ClaimStatus",
        "ClaimDate",
        "ClaimLegitimacy",
    ]
    lineage_frame = pd.DataFrame(lineage)
    return FeatureEngineeringResult(
        features=result,
        lineage=lineage_frame,
        dropped_columns=dropped,
    )
