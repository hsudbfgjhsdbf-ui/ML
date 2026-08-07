"""Fast unit tests for the non-training contracts of Approach 1."""

import numpy as np
import pandas as pd

from src.data.loading import EXPECTED_COLUMNS, profile_dataset, validate_schema
from src.evaluation.metrics import compute_metrics, select_threshold
from src.features.engineering import engineer_features
from src.features.preprocessing import fit_transform_matrices, stratified_three_way_split


def sample_frame() -> pd.DataFrame:
    """Return a small valid claim frame for unit tests."""
    rows = []
    for index in range(100):
        rows.append(
            {
                "ClaimID": f"claim-{index}",
                "PatientID": f"patient-{index}",
                "ProviderID": f"provider-{index}",
                "ClaimAmount": float(1000 + index),
                "ClaimDate": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "DiagnosisCode": f"D{index:03d}",
                "ProcedureCode": f"P{index:03d}",
                "PatientAge": index % 90,
                "PatientGender": "F" if index % 2 else "M",
                "ProviderSpecialty": "Cardiology",
                "ClaimStatus": "Pending",
                "PatientIncome": float(20000 + 100 * index),
                "PatientMaritalStatus": "Married",
                "PatientEmploymentStatus": "Employed",
                "ProviderLocation": f"Location {index}",
                "ClaimType": "Inpatient",
                "ClaimSubmissionMethod": "Online",
                "Cluster": index % 4,
                "ClaimLegitimacy": "Fraud" if index % 5 == 0 else "Legitimate",
            }
        )
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def test_schema_profile_and_features() -> None:
    """The source gate and feature lineage should be deterministic."""
    frame = sample_frame()
    validate_schema(frame)
    profile = profile_dataset(frame)
    assert profile.rows == 100
    result = engineer_features(frame)
    assert "claim_to_income_ratio" in result.features
    assert "ClaimID" not in result.features.columns
    assert len(result.lineage) == result.features.shape[1]


def test_split_and_transform_are_finite() -> None:
    """The 70/15/15 split and fitted transformer preserve finite matrices."""
    frame = sample_frame()
    features = engineer_features(frame).features
    target = frame["ClaimLegitimacy"].eq("Fraud").astype(int)
    split = stratified_three_way_split(features, target, seed=42)
    matrices = fit_transform_matrices(split)
    assert matrices.x_train.shape[0] == 70
    assert matrices.x_validation.shape[0] == 15
    assert matrices.x_test.shape[0] == 15
    assert np.isfinite(matrices.x_train).all()


def test_fraud_positive_metrics_and_threshold() -> None:
    """Metric definitions use fraud as the positive class."""
    labels = [0, 0, 1, 1]
    probabilities = [0.05, 0.20, 0.80, 0.95]
    result = compute_metrics(labels, probabilities, threshold=0.5)
    assert result["true_positive"] == 2
    assert result["false_negative"] == 0
    threshold, sweep = select_threshold(labels, probabilities, points=9, precision_floor=0.5)
    assert 0.0 < threshold.threshold < 1.0
    assert not sweep.empty
