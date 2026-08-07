"""Transparent claim-level feature engineering for the traditional ML baseline.

Features are deterministic functions of information available at claim time. No target
label is used here; high-cardinality target encoding is applied later inside the
train-fitted preprocessing pipeline to avoid validation/test leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


ENGINEERED_FEATURES = [
    "claim_to_premium_ratio", "treatment_cost_deviation_z", "days_to_waiting_period_end",
    "claim_frequency_intensity", "amount_per_hospital_day_inr", "current_vs_historical_average",
    "distance_x_provider_risk", "age_x_claim_amount_lakh", "policy_utilisation_ratio",
    "claim_amount_log", "claim_amount_squared_lakh", "hospital_stay_squared",
]


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create documented domain, temporal, interaction and limited polynomial features.

    Args:
        frame: Raw synthetic claim records with required source fields.

    Returns:
        A copied DataFrame augmented with feature columns.

    Raises:
        KeyError: If a required source column is absent.
    """
    required = {
        "claim_amount_inr", "annual_premium_inr", "regional_treatment_baseline_inr",
        "policy_duration_days", "waiting_period_days", "claims_past_12_months",
        "hospitalization_days", "historical_average_claim_inr", "distance_to_hospital_km",
        "provider_rejection_rate", "age", "sum_insured_inr",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Cannot engineer features; missing source columns: {missing}")
    output = frame.copy()
    amount = output["claim_amount_inr"].clip(lower=1)
    expected = output["regional_treatment_baseline_inr"].clip(lower=1)
    premium = output["annual_premium_inr"].clip(lower=1)
    historical = output["historical_average_claim_inr"].clip(lower=1)
    output["claim_to_premium_ratio"] = amount / premium
    # A fixed relative dispersion is an explicit synthetic-data assumption, not a learned label signal.
    output["treatment_cost_deviation_z"] = (amount - expected) / (expected * 0.36)
    output["days_to_waiting_period_end"] = output["policy_duration_days"] - output["waiting_period_days"]
    output["claim_frequency_intensity"] = output["claims_past_12_months"] / 12.0
    output["amount_per_hospital_day_inr"] = amount / output["hospitalization_days"].clip(lower=1)
    output["current_vs_historical_average"] = amount / historical
    output["distance_x_provider_risk"] = output["distance_to_hospital_km"] * output["provider_rejection_rate"]
    output["age_x_claim_amount_lakh"] = output["age"] * (amount / 100_000.0)
    output["policy_utilisation_ratio"] = amount / output["sum_insured_inr"].clip(lower=1)
    output["claim_amount_log"] = np.log1p(amount)
    # Limited degree-two terms; broad polynomial expansion is intentionally avoided.
    output["claim_amount_squared_lakh"] = (amount / 100_000.0) ** 2
    output["hospital_stay_squared"] = output["hospitalization_days"] ** 2
    return output


def feature_dictionary() -> list[dict[str, str]]:
    """Return definitions for engineered variables to merge into the data dictionary."""
    return [
        {"feature": "claim_to_premium_ratio", "type": "numeric", "range": "0 to unbounded", "description": "Current claim divided by annual premium.", "fraud_relevance": "High values can indicate unusual policy utilisation."},
        {"feature": "treatment_cost_deviation_z", "type": "numeric", "range": "roughly -3 to >3", "description": "Claim deviation from regional treatment baseline using assumed 36% relative spread.", "fraud_relevance": "Large positive deviations may merit review after regional/tier context."},
        {"feature": "days_to_waiting_period_end", "type": "integer", "range": "negative to policy duration", "description": "Policy age minus waiting-period duration on claim date.", "fraud_relevance": "Near/negative values identify coverage-timing risk."},
        {"feature": "claim_frequency_intensity", "type": "numeric", "range": "0 to unbounded", "description": "Claims filed in previous 12 months divided by 12.", "fraud_relevance": "Frequency spikes can be a fraud indicator but are not proof."},
        {"feature": "amount_per_hospital_day_inr", "type": "numeric", "range": "positive INR", "description": "Claim amount divided by inpatient days; day-care uses one day.", "fraud_relevance": "Flags implausible per-day cost after tier/context controls."},
        {"feature": "current_vs_historical_average", "type": "numeric", "range": "positive", "description": "Current amount divided by policyholder historical mean.", "fraud_relevance": "Large departures from personal baseline may warrant review."},
        {"feature": "distance_x_provider_risk", "type": "numeric", "range": "non-negative", "description": "Travel distance multiplied by historical provider rejection-rate proxy.", "fraud_relevance": "Captures distance/provider interaction without a deterministic rule."},
        {"feature": "age_x_claim_amount_lakh", "type": "numeric", "range": "non-negative", "description": "Age multiplied by amount in lakhs.", "fraud_relevance": "Limited interaction used to model age-treatment cost patterns."},
        {"feature": "policy_utilisation_ratio", "type": "numeric", "range": "0 to unbounded", "description": "Current claim divided by sum insured.", "fraud_relevance": "Near-limit amounts may require additional validation."},
        {"feature": "claim_amount_log", "type": "numeric", "range": "positive", "description": "Natural log of one plus claim amount.", "fraud_relevance": "Stabilises right-skewed INR claim distributions."},
        {"feature": "claim_amount_squared_lakh", "type": "numeric", "range": "non-negative", "description": "Squared claim amount in lakh INR units.", "fraud_relevance": "Captures limited nonlinear large-claim risk."},
        {"feature": "hospital_stay_squared", "type": "numeric", "range": "non-negative", "description": "Square of hospitalization duration.", "fraud_relevance": "Captures nonlinear duration/cost relationships."},
    ]
