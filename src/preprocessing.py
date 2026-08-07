"""Leakage-aware cleaning, encoding, selection and imbalance utilities.

All fitted transformations are learned from the training partition only. Validation and
test data are transformed solely by the fitted objects. This is essential in fraud
studies because imputation, target encoding and sampling can otherwise leak labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.feature_engineering import engineer_features
from src.utils import write_json

TARGET = "is_fraud"
ID_COLUMNS = {"claim_id", "policyholder_id"}
DATE_COLUMNS = {"claim_date", "policy_start_date"}


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """Train-fitted smoothed binary target encoder for high-cardinality categoricals.

    It substitutes a category's observed training fraud rate with a smoothed estimate.
    Unknown categories use the global training rate. It never sees validation/test labels.
    The estimator is intentionally simple, serialisable and suitable for audit.
    """

    def __init__(self, smoothing: float = 20.0) -> None:
        """Initialize encoder with pseudo-count smoothing strength."""
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame | np.ndarray, y: Iterable[int]) -> "SmoothedTargetEncoder":
        """Learn per-column category-to-smoothed-target mappings from training data.

        Args:
            X: Categorical training columns.
            y: Binary training targets aligned to ``X``.

        Returns:
            Fitted encoder.

        Raises:
            ValueError: If X and y have different lengths.
        """
        frame = self._frame(X)
        target = pd.Series(np.asarray(list(y)), index=frame.index, dtype=float)
        if len(frame) != len(target):
            raise ValueError("Target encoder X/y length mismatch.")
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.global_mean_ = float(target.mean())
        self.mappings_: dict[str, dict[str, float]] = {}
        for column in frame.columns:
            values = frame[column].fillna("__MISSING__").astype(str)
            grouped = pd.DataFrame({"value": values, "target": target}).groupby("value")["target"].agg(["mean", "count"])
            encoded = (grouped["mean"] * grouped["count"] + self.global_mean_ * self.smoothing) / (grouped["count"] + self.smoothing)
            self.mappings_[str(column)] = encoded.to_dict()
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Map categories to training-derived smoothed risk values.

        Args:
            X: New categorical rows with the fitted columns.

        Returns:
            Dense float matrix of encoded features.

        Raises:
            RuntimeError: If called before fitting.
        """
        if not hasattr(self, "mappings_"):
            raise RuntimeError("SmoothedTargetEncoder must be fit before transform.")
        frame = self._frame(X)
        encoded = []
        for column in self.feature_names_in_:
            mapping = self.mappings_[str(column)]
            values = frame[str(column)].fillna("__MISSING__").astype(str)
            encoded.append(values.map(mapping).fillna(self.global_mean_).to_numpy(dtype=float))
        return np.column_stack(encoded)

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        """Expose stable names compatible with scikit-learn ColumnTransformer."""
        names = list(input_features) if input_features is not None else list(self.feature_names_in_)
        return np.asarray([f"target_encoded_{name}" for name in names], dtype=object)

    @staticmethod
    def _frame(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Normalize scikit-learn transformer input into a named DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        array = np.asarray(X)
        return pd.DataFrame(array, columns=[f"feature_{i}" for i in range(array.shape[1])])


@dataclass
class DataBundle:
    """Fitted data-processing artifacts and transformed stratified partitions."""

    raw_train: pd.DataFrame
    raw_validation: pd.DataFrame
    raw_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    selector: SelectKBest
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    selected_feature_names: list[str]
    quality_report: dict[str, Any]


def clean_claims(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate labels, remove exact duplicates and create date-derived fields.

    Suspicious monetary outliers are deliberately retained: in a fraud application they
    may contain signal. The report identifies IQR/Z-score candidates rather than deleting
    them. Only impossible records receive defensive clipping/validation.

    Args:
        frame: Synthetic raw records.

    Returns:
        Cleaned DataFrame and a complete data-quality report.

    Raises:
        ValueError: If target is missing/non-binary or cleaning leaves too little data.
    """
    if TARGET not in frame:
        raise ValueError(f"Required binary target column is absent: {TARGET}")
    original_rows = len(frame)
    missingness = {column: round(float(value), 6) for column, value in frame.isna().mean().items()}
    duplicate_count = int(frame.duplicated().sum())
    cleaned = frame.drop_duplicates().copy()
    if cleaned[TARGET].isna().any() or set(cleaned[TARGET].dropna().unique()) - {0, 1}:
        raise ValueError("Target must contain only non-missing binary 0/1 labels.")
    if len(cleaned) < 1_000:
        raise ValueError("Insufficient rows remain after cleaning; need at least 1,000 claims.")
    for date_column in DATE_COLUMNS:
        cleaned[date_column] = pd.to_datetime(cleaned[date_column], errors="coerce")
        if cleaned[date_column].isna().any():
            raise ValueError(f"Unparseable date values in {date_column}.")
    cleaned["claim_month"] = cleaned["claim_date"].dt.month.astype(str)
    cleaned["claim_quarter"] = cleaned["claim_date"].dt.quarter.astype(str)
    cleaned["claim_weekday"] = cleaned["claim_date"].dt.dayofweek.astype(str)
    cleaned["policy_start_year"] = cleaned["policy_start_date"].dt.year.astype(str)
    numeric = cleaned.select_dtypes(include=[np.number]).drop(columns=[TARGET], errors="ignore")
    z_candidates = ((numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, np.nan)).abs().gt(3).sum()
    q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
    iqr = q3 - q1
    iqr_candidates = ((numeric.lt(q1 - 1.5 * iqr)) | (numeric.gt(q3 + 1.5 * iqr))).sum()
    report = {
        "rows_before_cleaning": original_rows,
        "rows_after_cleaning": int(len(cleaned)),
        "exact_duplicates_removed": duplicate_count,
        "missingness_fraction_before_imputation": missingness,
        "outlier_candidates_preserved": {
            "z_score_abs_gt_3": {k: int(v) for k, v in z_candidates.items() if v},
            "iqr_1_5": {k: int(v) for k, v in iqr_candidates.items() if v},
        },
        "outlier_policy": "Plausible high-value or unusual claims are retained because they can be fraud signal; no statistical outlier was deleted.",
        "target_distribution_after_cleaning": {str(k): int(v) for k, v in cleaned[TARGET].value_counts().items()},
    }
    return cleaned, report


def model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic features and remove target/identity/raw-date leakage columns.

    Args:
        frame: Cleaned claim DataFrame.

    Returns:
        Feature-only DataFrame suitable for fitting a preprocessor.
    """
    engineered = engineer_features(frame)
    forbidden = ID_COLUMNS | DATE_COLUMNS | {TARGET, "claim_legitimacy"}
    output = engineered.drop(columns=list(forbidden), errors="ignore")
    # Object categories are allowed; labels and direct identifiers are not.
    return output


def infer_feature_groups(features: pd.DataFrame, high_cardinality: list[str], low_max: int) -> tuple[list[str], list[str], list[str]]:
    """Partition model columns into numerical, low-cardinality and target-encoded groups.

    Args:
        features: Training-only feature DataFrame.
        high_cardinality: Configured category names requiring target encoding.
        low_max: Maximum unique categorical values for one-hot encoding.

    Returns:
        Tuple of numeric, low-cardinality categorical and high-cardinality categorical names.
    """
    numeric = list(features.select_dtypes(include=[np.number, "bool"]).columns)
    categoricals = [column for column in features.columns if column not in numeric]
    high = [column for column in categoricals if column in high_cardinality or features[column].nunique(dropna=False) > low_max]
    low = [column for column in categoricals if column not in high]
    return numeric, low, high


def build_preprocessor(features: pd.DataFrame, config: dict[str, Any]) -> tuple[ColumnTransformer, dict[str, list[str]]]:
    """Build an unfitted train-only serialisable preprocessing pipeline.

    Args:
        features: Training-only raw feature frame used solely to infer data groups.
        config: Loaded project configuration.

    Returns:
        Unfitted ColumnTransformer and feature-group metadata.
    """
    numeric, low, high = infer_feature_groups(
        features,
        list(config["preprocessing"]["high_cardinality_columns"]),
        int(config["preprocessing"]["low_cardinality_max"]),
    )
    numeric_pipe = Pipeline([
        ("impute_median", SimpleImputer(strategy="median")),
        ("robust_scale", RobustScaler()),
    ])
    low_pipe = Pipeline([
        ("impute_missing_category", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    high_pipe = Pipeline([
        ("target_encode", SmoothedTargetEncoder(smoothing=25.0)),
        ("robust_scale", RobustScaler()),
    ])
    transformers: list[tuple[str, Any, list[str]]] = [("numeric", numeric_pipe, numeric)]
    if low:
        transformers.append(("low_cardinality", low_pipe, low))
    if high:
        transformers.append(("high_cardinality", high_pipe, high))
    pipeline = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)
    return pipeline, {"numeric": numeric, "low_cardinality": low, "high_cardinality": high}


def create_data_bundle(frame: pd.DataFrame, config: dict[str, Any]) -> DataBundle:
    """Clean, stratify, fit train-only transforms and select mutual-information features.

    Args:
        frame: Raw synthetic study data (duplicates permitted).
        config: Loaded pipeline configuration.

    Returns:
        DataBundle used by all Approach-1 model training and evaluation stages.
    """
    cleaned, quality = clean_claims(frame)
    X_all, y_all = model_frame(cleaned), cleaned[TARGET].astype(int)
    test_plus_validation = float(config["split"]["validation_size"]) + float(config["split"]["test_size"])
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=test_plus_validation, stratify=y_all,
        random_state=int(config["project"]["random_seed"]),
    )
    test_fraction_of_temp = float(config["split"]["test_size"]) / test_plus_validation
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp, y_temp, test_size=test_fraction_of_temp, stratify=y_temp,
        random_state=int(config["project"]["random_seed"]) + 1,
    )
    preprocessor, group_meta = build_preprocessor(X_train, config)
    transformed_train = preprocessor.fit_transform(X_train, y_train)
    transformed_validation = preprocessor.transform(X_validation)
    transformed_test = preprocessor.transform(X_test)
    max_features = min(int(config["preprocessing"]["feature_selection_k"]), transformed_train.shape[1])
    selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
    selected_train = selector.fit_transform(transformed_train, y_train)
    selected_validation = selector.transform(transformed_validation)
    selected_test = selector.transform(transformed_test)
    feature_names = np.asarray(preprocessor.get_feature_names_out())
    selected_names = feature_names[selector.get_support()].tolist()
    quality["feature_groups"] = group_meta
    quality["feature_selection"] = {
        "method": "mutual information filter fit on training partition",
        "features_before_selection": int(len(feature_names)),
        "features_after_selection": int(len(selected_names)),
        "selected_features": selected_names,
    }
    quality["stratified_split"] = {
        "train_rows": int(len(X_train)), "validation_rows": int(len(X_validation)), "test_rows": int(len(X_test)),
        "train_fraud_rate": float(y_train.mean()), "validation_fraud_rate": float(y_validation.mean()), "test_fraud_rate": float(y_test.mean()),
    }
    # Raw partitions retain fields required for fairness and business-cost analysis.
    indices = {"train": X_train.index, "validation": X_validation.index, "test": X_test.index}
    return DataBundle(
        raw_train=cleaned.loc[indices["train"]].copy(), raw_validation=cleaned.loc[indices["validation"]].copy(), raw_test=cleaned.loc[indices["test"]].copy(),
        y_train=y_train.copy(), y_validation=y_validation.copy(), y_test=y_test.copy(), preprocessor=preprocessor, selector=selector,
        X_train=np.asarray(selected_train), X_validation=np.asarray(selected_validation), X_test=np.asarray(selected_test),
        selected_feature_names=selected_names, quality_report=quality,
    )


def save_processed_partitions(bundle: DataBundle, config: dict[str, Any]) -> None:
    """Persist split raw records and a quality report for independent inspection.

    Args:
        bundle: Prepared data bundle.
        config: Loaded configuration with processed-data destination.
    """
    destination = Path(config["paths"]["processed_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    for name, partition in (("train", bundle.raw_train), ("validation", bundle.raw_validation), ("test", bundle.raw_test)):
        partition.to_csv(destination / f"{name}_claims.csv", index=False)
    write_json(destination / "data_quality_report.json", bundle.quality_report)


def build_resampler(strategy: str, seed: int, k_neighbors: int = 5) -> Any | None:
    """Return a train-only imbalanced-learn sampler or ``None`` for class weighting.

    Args:
        strategy: ``class_weight``, ``random_under``, ``smote``, ``tomek`` or ``smoteenn``.
        seed: Random seed for stochastic samplers.
        k_neighbors: SMOTE neighbourhood size.

    Returns:
        Configured sampler, or ``None`` if the estimator should use class weights.

    Raises:
        ValueError: If the named strategy is unsupported.
    """
    normalized = strategy.lower()
    if normalized == "class_weight":
        return None
    if normalized == "random_under":
        return RandomUnderSampler(random_state=seed)
    if normalized == "smote":
        return SMOTE(random_state=seed, k_neighbors=k_neighbors)
    if normalized == "tomek":
        return TomekLinks()
    if normalized == "smoteenn":
        return SMOTEENN(random_state=seed, smote=SMOTE(random_state=seed, k_neighbors=k_neighbors))
    raise ValueError(f"Unsupported imbalance strategy: {strategy}")


def export_data_dictionary(frame: pd.DataFrame, config: dict[str, Any]) -> None:
    """Create a complete machine-readable and Markdown data dictionary from study data.

    Args:
        frame: Synthetic raw study data.
        config: Loaded project configuration.
    """
    descriptions = {
        "claim_id": "Synthetic claim identifier; excluded from modelling.", "policyholder_id": "Synthetic policyholder identifier; excluded from modelling.",
        "provider_id": "Synthetic provider identifier; target encoded using training labels only.", "claim_date": "Claim event date.",
        "state": "Claimant residence state in India.", "city": "Claimant residence city in India.", "age": "Claimant age in years (0–90).",
        "gender": "Self-reported gender category, retained for fairness audit.", "income_bracket": "Synthetic monthly-income category, retained for fairness audit.",
        "monthly_income_inr": "Synthetic estimated monthly income in INR.", "occupation_type": "Synthetic occupation category.",
        "disability_accommodation": "Accommodation status; fairness/audit field, not a decision rule.", "policy_type": "Individual, family floater, employer group, Ayushman Bharat or ECHS.",
        "insurer": "Synthetic insurer/product issuer.", "sum_insured_inr": "Policy coverage limit in INR.", "annual_premium_inr": "Annual premium in INR.",
        "policy_start_date": "Policy inception date.", "policy_duration_days": "Days elapsed from inception to claim.", "waiting_period_days": "Applicable waiting period in days.",
        "waiting_period_completed": "Whether the waiting period is complete at claim date.", "copay_percent": "Policyholder co-payment percentage.",
        "claim_amount_inr": "Claimed amount in Indian Rupees.", "claim_type": "Hospitalization, day-care, outpatient or pre-authorization.",
        "treatment_type": "Synthetic clinical treatment category.", "medical_practice": "Allopathic or Ayurvedic treatment practice.",
        "diagnosis_code": "Synthetic ICD-style diagnosis code.", "procedure_code": "Procedure description/code.",
        "hospitalization_days": "Length of stay; zero for outpatient care.", "procedure_count": "Number of procedures on claim.",
        "doctor_credential": "Synthetic credential/documentation category.", "hospital_name": "Synthetic provider name.", "hospital_tier": "Government, nursing home, tier-2 private, corporate or AYUSH centre.",
        "hospital_state": "Provider state.", "network_hospital": "Network status for the synthetic policy.", "distance_to_hospital_km": "Approximate claimant-to-provider travel distance.",
        "time_since_last_claim_days": "Elapsed days since prior claim.", "claims_past_12_months": "Prior 12-month claim count.", "total_historical_claims": "Total prior claims.",
        "historical_claimed_amount_inr": "Total prior claimed amount in INR.", "historical_average_claim_inr": "Prior mean claim amount in INR.",
        "historical_claim_std_inr": "Prior claim amount standard deviation in INR.", "historical_max_claim_inr": "Prior maximum claim amount in INR.",
        "rejected_claim_count": "Prior rejected-claim count proxy.", "provider_rejection_rate": "Historical provider rejection-rate proxy, generated independent of current label.",
        "provider_average_claim_inr": "Provider-level average claim proxy.", "provider_unique_patient_count": "Provider patient volume proxy.",
        "regional_treatment_baseline_inr": "Expected regional/tier treatment cost baseline.", "season": "Claim season in India.",
        "claim_submission_method": "Cashless, reimbursement, portal or branch submission.", "gst_amount_inr": "Synthetic GST component in INR where applicable.",
        "document_completeness_score": "Synthetic document-completeness score (0–1).", "is_fraud": "Binary synthetic target: 1 fraudulent, 0 legitimate.",
        "claim_legitimacy": "Human-readable form of synthetic target.",
    }
    rows = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            valid = f"{series.min(skipna=True):.3g} to {series.max(skipna=True):.3g}"
        else:
            values = series.dropna().astype(str).unique()
            valid = ", ".join(sorted(values)[:8]) + (" …" if len(values) > 8 else "")
        rows.append({"feature": column, "type": str(series.dtype), "valid_range_or_values": valid, "description": descriptions.get(column, "Documented synthetic study feature."), "fraud_relevance": "Target" if column == TARGET else "Potential contextual signal; not proof of fraud."})
    destination = Path(config["paths"]["documentation_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / "data_dictionary.csv", index=False)
    markdown = ["# Data Dictionary", "", "> **Data status:** educational synthetic Indian-context claims. It must not be interpreted as records from an insurer or used to make a real coverage decision.", "", "| Feature | Type | Valid range / values | Description | Fraud relevance |", "|---|---|---|---|---|"]
    for row in rows:
        markdown.append("| {feature} | {type} | {valid_range_or_values} | {description} | {fraud_relevance} |".format(**{k: str(v).replace("|", "/") for k,v in row.items()}))
    (destination / "data_dictionary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
