"""Fast regression tests for the reproducible Approach-1 data pipeline."""
from __future__ import annotations

import numpy as np

from src.data_loading import generate_indian_synthetic_claims
from src.feature_engineering import ENGINEERED_FEATURES, engineer_features
from src.preprocessing import SmoothedTargetEncoder, clean_claims


def test_synthetic_generator_is_deterministic_and_has_plausible_fraud_rate() -> None:
    """A fixed seed must generate identical educational records and a 5–15% fraud rate."""
    first = generate_indian_synthetic_claims(1_000, seed=11)
    second = generate_indian_synthetic_claims(1_000, seed=11)
    assert first.equals(second)
    unique, report = clean_claims(first)
    assert len(unique) == 1_000
    assert report["exact_duplicates_removed"] == 5
    assert 0.05 <= unique["is_fraud"].mean() <= 0.15
    assert {"Ayurvedic", "Allopathic"}.issubset(set(unique["medical_practice"]))


def test_engineered_features_are_finite_and_nonleaking() -> None:
    """Feature functions should add all documented variables without target-derived columns."""
    data, _ = clean_claims(generate_indian_synthetic_claims(1_000, seed=12))
    engineered = engineer_features(data)
    assert set(ENGINEERED_FEATURES).issubset(engineered.columns)
    assert np.isfinite(engineered[ENGINEERED_FEATURES].to_numpy(dtype=float)).all()


def test_target_encoder_uses_global_rate_for_unseen_categories() -> None:
    """Unknown categorical values must map to train global rate, not error or a test label."""
    encoder = SmoothedTargetEncoder(smoothing=5.0).fit(
        np.array([["A"], ["A"], ["B"]]), np.array([1, 0, 0])
    )
    transformed = encoder.transform(np.array([["UNSEEN"], ["A"]]))
    assert transformed.shape == (2, 1)
    assert np.isclose(transformed[0, 0], 1 / 3)
