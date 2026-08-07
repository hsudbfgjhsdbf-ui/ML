"""Single-command orchestration for Approach 1 traditional machine learning.

The pipeline follows an auditable order: load and gate data, run EDA, split,
fit transformations on train only, benchmark the model zoo, select on
validation metrics, unlock the test set once, and build all deliverables from
the resulting artifact dictionaries.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.inspection import permutation_importance

from src.data.loading import load_claims, missingness_table, profile_dataset, validate_schema
from src.evaluation.metrics import (
    bootstrap_intervals,
    compute_metrics,
    fairness_metrics,
    probabilities_from_estimator,
    select_threshold,
)
from src.evaluation.plots import calibration_plot, feature_importance_plot, generate_eda_plots, generate_model_curves
from src.features.engineering import engineer_features
from src.features.preprocessing import fit_transform_matrices, simple_smote, stratified_three_way_split
from src.models.zoo import ModelSpec, build_model_specs
from src.reporting.documents import build_all_documents
from src.utils.logging_utils import configure_logging
from src.utils.paths import ProjectPaths, find_repository_root
from src.utils.reproducibility import environment_snapshot, seed_everything, sha256_file, stable_json_hash, write_json


LOGGER = logging.getLogger("medical_fraud")


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file.

    Args:
        path: YAML path.
    Returns:
        Configuration dictionary.
    Raises:
        FileNotFoundError: If the YAML file is absent.
        ValueError: If required keys or split fractions are invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = ["seed", "data", "splitting", "training", "paths"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Configuration is missing sections: {missing}")
    fractions = config["splitting"]
    if not np.isclose(
        sum(float(fractions[key]) for key in ["train_fraction", "validation_fraction", "test_fraction"]), 1.0
    ):
        raise ValueError("split fractions must sum to 1.0")
    return config


def _resolve(path_value: str, root: Path) -> Path:
    """Resolve a configuration path relative to the repository root."""
    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else root / candidate


def _save_curve_csv(path: Path, x: np.ndarray, y: np.ndarray, name_x: str, name_y: str) -> None:
    """Write curve coordinates as a complete CSV artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({name_x: x, name_y: y}).to_csv(path, index=False)


def _json_safe(value: Any) -> Any:
    """Convert NumPy and estimator values to JSON-safe representations."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _fit_one_model(
    spec: ModelSpec, x_train: np.ndarray, y_train: pd.Series, config: dict[str, Any], tuning_dir: Path
) -> tuple[Any, dict[str, Any], int, float]:
    """Fit one estimator under the declared search budget.

    Args:
        spec: Model registry entry.
        x_train: Numeric transformed training matrix.
        y_train: Binary training labels.
        config: Complete configuration dictionary.
        tuning_dir: Directory for search history.
    Returns:
        Fitted estimator, best parameter mapping, recorded trial count, seconds.
    Raises:
        RuntimeError: If fitting or the search fails.
    """
    seed = int(config["seed"])
    cv = StratifiedKFold(n_splits=int(config["training"]["cv_folds"]), shuffle=True, random_state=seed)
    start = time.perf_counter()
    estimator = spec.builder()
    trial_count = 0
    if spec.search_kind == "none" or not spec.search_space:
        estimator.fit(x_train, y_train)
        best_params: dict[str, Any] = {}
        trial_count = 1
        search_results = pd.DataFrame(
            [{"params": "{}", "mean_test_score": np.nan, "std_test_score": np.nan, "rank_test_score": 1}]
        )
    else:
        total_combinations = 1
        for values in spec.search_space.values():
            total_combinations *= max(1, len(values))
        budget = min(int(spec.n_iter), total_combinations) if spec.search_kind == "random" else total_combinations
        if spec.search_kind == "random":
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=spec.search_space,
                n_iter=budget,
                scoring=config["training"].get("scoring", "average_precision"),
                cv=cv,
                refit=True,
                n_jobs=int(config["training"].get("n_jobs", -1)),
                random_state=seed,
                error_score="raise",
                return_train_score=False,
            )
        else:
            search = GridSearchCV(
                estimator=estimator,
                param_grid=spec.search_space,
                scoring=config["training"].get("scoring", "average_precision"),
                cv=cv,
                refit=True,
                n_jobs=int(config["training"].get("n_jobs", -1)),
                error_score="raise",
                return_train_score=False,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            search.fit(x_train, y_train)
        estimator = search.best_estimator_
        best_params = _json_safe(search.best_params_)
        search_results = pd.DataFrame(search.cv_results_)
        trial_count = int(len(search_results))
        keep = [
            column
            for column in ["params", "mean_test_score", "std_test_score", "rank_test_score"]
            if column in search_results
        ]
        search_results = search_results[keep].copy()
        if "params" in search_results:
            search_results["params"] = search_results["params"].map(
                lambda value: json.dumps(_json_safe(value), sort_keys=True, default=str)
            )
    elapsed = time.perf_counter() - start
    tuning_dir.mkdir(parents=True, exist_ok=True)
    search_results.to_csv(tuning_dir / f"{spec.key}_search.csv", index=False)
    return estimator, best_params, trial_count, elapsed


def _model_feature_importance(
    estimator: Any, feature_names: list[str], x: np.ndarray, y: np.ndarray, seed: int
) -> pd.DataFrame:
    """Extract native or permutation importance for the selected estimator."""
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_)).reshape(-1)
    else:
        permutation = permutation_importance(
            estimator, x, y, scoring="average_precision", n_repeats=5, random_state=seed, n_jobs=-1
        )
        values = np.asarray(permutation.importances_mean, dtype=float)
    if len(values) != len(feature_names):
        values = np.resize(values, len(feature_names))
    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _threshold_plot(winner: dict[str, Any], output_path: Path, dpi: int) -> dict[str, str]:
    """Render precision/recall/F2 across the winner's validation thresholds."""
    import matplotlib.pyplot as plt
    from src.evaluation.plots import configure_style, _save

    configure_style()
    sweep = winner["threshold_sweep"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep["threshold"], sweep["precision"], label="Precision", color="#2A9D8F")
    ax.plot(sweep["threshold"], sweep["recall"], label="Recall", color="#457B9D")
    ax.plot(sweep["threshold"], sweep["f2"], label="F2", color="#E76F51", linewidth=2)
    ax.axvline(winner["threshold"], linestyle="--", color="#102A43", label=f"Selected = {winner['threshold']:.2f}")
    ax.set_title("Validation threshold sweep — selected model")
    ax.set_xlabel("Fraud probability threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.05)
    ax.legend()
    return _save(
        fig,
        output_path,
        "Validation threshold sweep",
        "F2 is maximized on validation data with the selected operating threshold marked.",
        dpi,
    )


def _write_data_metadata(paths: ProjectPaths, raw: pd.DataFrame, profile: Any, config: dict[str, Any]) -> None:
    """Write dataset card, dictionary, missingness, and checksum metadata."""
    source_path = paths.raw_data / "health_insurance_fraud_claims.xlsx"
    manifest = {
        "file": str(source_path.relative_to(paths.root)),
        "sha256": sha256_file(source_path),
        "bytes": source_path.stat().st_size,
        "retrieved_or_supplied": "repository-provided snapshot",
        "license_status": "license and redistribution terms require confirmation from the data owner",
        "profile": profile.as_dict(),
    }
    write_json(paths.root / "data" / "metadata" / "raw_manifest.json", manifest)
    missingness_table(raw).to_csv(paths.root / "data" / "metadata" / "missingness.csv", index=False)
    dictionary_rows = []
    descriptions = {
        "ClaimID": "Unique claim identifier; audit-only.",
        "PatientID": "Patient identifier; excluded from model features.",
        "ProviderID": "Provider identifier; excluded from model features.",
        "ClaimAmount": "Claimed monetary amount; source unit is not independently verified.",
        "ClaimDate": "Date associated with the supplied claim row.",
        "DiagnosisCode": "Opaque diagnosis code; near-unique in this snapshot.",
        "ProcedureCode": "Opaque procedure code; near-unique in this snapshot.",
        "PatientAge": "Reported patient age in years.",
        "PatientGender": "Reported gender category.",
        "ProviderSpecialty": "Provider specialty category.",
        "ClaimStatus": "Supplied claim status; excluded as potentially post-decision.",
        "PatientIncome": "Reported patient income; source unit is not independently verified.",
        "PatientMaritalStatus": "Reported marital-status category.",
        "PatientEmploymentStatus": "Reported employment category.",
        "ProviderLocation": "Supplied provider location string; excluded due to high cardinality and unverified geography.",
        "ClaimType": "Claim type category.",
        "ClaimSubmissionMethod": "Claim submission channel.",
        "Cluster": "Supplied numeric cluster; retained with shortcut-risk audit.",
        "ClaimLegitimacy": "Target label: Fraud or Legitimate.",
    }
    for column in raw.columns:
        dictionary_rows.append(
            {
                "field": column,
                "dtype": str(raw[column].dtype),
                "description": descriptions.get(column, "Source field."),
                "missing_percent": float(raw[column].isna().mean() * 100),
                "unique_values": int(raw[column].nunique()),
                "model_role": "target"
                if column == "ClaimLegitimacy"
                else ("audit_only" if column in ["ClaimID", "PatientID", "ProviderID"] else "source_or_excluded"),
            }
        )
    dictionary = pd.DataFrame(dictionary_rows)
    dictionary.to_csv(paths.root / "data" / "data_dictionary.csv", index=False)
    card = f"""# Dataset card — supplied health-insurance fraud workbook

## Motivation

The workbook is used to demonstrate a reproducible binary fraud-screening
pipeline for an academic project at IIIT Dharwad. It is not presented as a
national Indian claims sample.

## Composition

- File: `data/raw/health_insurance_fraud_claims.xlsx`
- Rows: {profile.rows:,}; source columns: {profile.columns}
- Fraud rows: {profile.fraud_count:,}; legitimate rows: {profile.legitimate_count:,}; prevalence: {profile.fraud_rate:.4f}
- Exact duplicates: {profile.duplicate_rows}; duplicate ClaimID: {profile.duplicate_claim_ids}
- Missing cells: {profile.missing_cells}
- Checksum: `{manifest['sha256']}`

## Collection and provenance

The workbook was present in the repository supplied for this task. The original
creator, collection process, license, and redistribution permissions are not
verified in the checkout. The project preserves the source copy and records the
unknown-license status rather than attributing it to Kaggle or a government
source without evidence.

## Preprocessing

The target is `ClaimLegitimacy`; `Fraud` maps to 1. IDs and near-unique codes
are audit-only or excluded. Numeric and categorical transformations are fitted
on the training partition. See `data/data_dictionary.csv` and
`documentation/feature_engineering.md`.

## Uses and non-uses

Use for academic reproducibility, code review, and baseline comparison. Do not
use for real claim denial, risk pricing, claimant profiling, or regulatory
submission. No real patient-identifying fields were observed, but identifiers
are still treated as sensitive and omitted from printed examples.

## Known limitations and biases

The sample is small, has one row per unique patient/provider in this snapshot,
contains no policy or document evidence, lacks verified Indian geography, and
may contain synthetic shortcuts such as Cluster. Demographic performance is
therefore an audit exercise, not proof of fairness in production.

## FAQ

**Why not use the large Medicare table plan?** The supplied repository contains
this workbook; no other claims tables are available.  
**Is the workbook public and license-cleared?** That is not verifiable from the
checkout and must be confirmed by the data owner.  
**Are amounts confirmed INR?** No; the pipeline does not invent a conversion.  
**Can the result be generalized to Indian insurers?** No; external validation is required.  
**What is the target?** `Fraud` is positive and `Legitimate` is negative.
"""
    paths.root.joinpath("data", "dataset_card.md").write_text(card.rstrip() + "\n", encoding="utf-8")


def _split_summary(split: Any) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return split summary values and a machine-readable membership table."""
    rows = []
    for name, indices, labels in [
        ("train", split.train_indices, split.y_train),
        ("validation", split.validation_indices, split.y_validation),
        ("test", split.test_indices, split.y_test),
    ]:
        rows.append(
            {
                "split": name,
                "rows": len(indices),
                "fraud": int(labels.sum()),
                "fraud_rate": float(labels.mean()),
                "index_min": int(np.min(indices)),
                "index_max": int(np.max(indices)),
            }
        )
    return {
        f"{row['split']}_{key}": value
        for row in rows
        for key, value in [("rows", row["rows"]), ("fraud", row["fraud"]), ("fraud_rate", row["fraud_rate"])]
    }, pd.DataFrame(rows)


def run_pipeline(config_path: Path, dry_run: bool = False, self_test: bool = False) -> dict[str, Any]:
    """Run stages S0 through S9 and return the actual artifact context.

    Args:
        config_path: YAML configuration path.
        dry_run: Print the plan and do not write model artifacts.
        self_test: Execute a quick import/metric sanity check and stop.
    Returns:
        Context dictionary used by document builders.
    Raises:
        RuntimeError: If a validation or training stage fails.
    """
    root = find_repository_root(config_path.parent)
    paths = ProjectPaths(root)
    paths.ensure(extras=[paths.evaluation / "explainability", paths.evaluation / "calibration"])
    config = load_config(config_path)
    log_path = paths.evaluation / "pipeline.log"
    global LOGGER
    LOGGER = configure_logging(log_path)
    seed = int(config["seed"])
    seed_everything(seed)
    if self_test:
        values = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
        if values["f1"] != 1.0:
            raise RuntimeError("Self-test metric calculation failed")
        LOGGER.info("Self-test passed: metric imports, seed, and probability contract are healthy")
        return {"status": "self_test_pass"}
    plan = [
        "S0 environment",
        "S1 load and validate",
        "S2 profile",
        "S3 EDA",
        "S4 split",
        "S5 preprocess",
        "S6 model zoo",
        "S7 select",
        "S8 locked test",
        "S9 documents",
    ]
    if dry_run:
        print("\n".join(plan))
        return {"status": "dry_run", "plan": plan}

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_timestamp = datetime.now(timezone.utc).strftime("%d-%m-%Y %H:%M:%S UTC")
    run_dir = paths.evaluation / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Starting %s with seed=%s", run_id, seed)
    input_path = _resolve(config["data"]["input_path"], root)
    raw = load_claims(input_path, config["data"].get("max_rows"))
    validate_schema(raw)
    profile = profile_dataset(raw)
    _write_data_metadata(paths, raw, profile, config)
    LOGGER.info("Loaded %s rows with fraud prevalence %.4f", profile.rows, profile.fraud_rate)
    engineered_result = engineer_features(raw)
    engineered = engineered_result.features
    target = (
        raw[config["data"]["target_column"]].astype(str).str.title() == config["data"]["positive_label"].title()
    ).astype(int)
    engineered.to_csv(paths.processed_data / "engineered_features.csv", index=False)
    engineered_result.lineage.to_csv(paths.evaluation / "feature_lineage.csv", index=False)
    split = stratified_three_way_split(
        engineered,
        target,
        seed=seed,
        train_fraction=float(config["splitting"]["train_fraction"]),
        validation_fraction=float(config["splitting"]["validation_fraction"]),
        test_fraction=float(config["splitting"]["test_fraction"]),
    )
    split_stats, split_frame = _split_summary(split)
    split_frame.to_csv(paths.evaluation / "split_summary.csv", index=False)
    pd.DataFrame(
        {
            "split": ["train"] * len(split.train_indices)
            + ["validation"] * len(split.validation_indices)
            + ["test"] * len(split.test_indices),
            "row_index": np.concatenate([split.train_indices, split.validation_indices, split.test_indices]),
        }
    ).to_csv(paths.evaluation / "split_membership.csv", index=False)
    LOGGER.info(
        "Split train=%s validation=%s test=%s",
        split_stats["train_rows"],
        split_stats["validation_rows"],
        split_stats["test_rows"],
    )
    figure_records = generate_eda_plots(
        raw, engineered, target, paths.images, dpi=int(config["reporting"].get("dpi", 180))
    )
    matrices = fit_transform_matrices(split)
    pd.DataFrame(
        {"transformed_feature": matrices.feature_names, "position": range(len(matrices.feature_names))}
    ).to_csv(paths.evaluation / "feature_registry.csv", index=False)
    joblib.dump(matrices.transformer, paths.models / "preprocessor.joblib")
    # Compare imbalance policies on a small, explicitly non-test experiment.
    # The comparison is fitted on training rows and measured on validation rows
    # so it cannot silently tune the final locked test result.
    from sklearn.linear_model import LogisticRegression

    imbalance_rows: list[dict[str, Any]] = []
    for policy in ["none", "class_weight_balanced", "smote"]:
        policy_model = LogisticRegression(
            max_iter=1500,
            solver="liblinear",
            random_state=seed,
            class_weight="balanced" if policy == "class_weight_balanced" else None,
        )
        policy_x, policy_y = matrices.x_train, split.y_train.to_numpy()
        if policy == "smote":
            policy_x, policy_y = simple_smote(policy_x, policy_y, seed=seed, target_ratio=0.5)
        policy_model.fit(policy_x, policy_y)
        policy_prob = probabilities_from_estimator(policy_model, matrices.x_validation)
        policy_threshold, _ = select_threshold(
            split.y_validation.to_numpy(),
            policy_prob,
            points=int(config["evaluation"].get("threshold_grid_points", 99)),
            precision_floor=float(config["evaluation"].get("precision_floor", 0.50)),
        )
        policy_metrics = compute_metrics(split.y_validation.to_numpy(), policy_prob, policy_threshold.threshold)
        imbalance_rows.append(
            {
                "policy": policy,
                "train_rows_after_policy": int(len(policy_y)),
                "threshold": policy_threshold.threshold,
                "val_precision": policy_metrics["precision"],
                "val_recall": policy_metrics["recall"],
                "val_f1": policy_metrics["f1"],
                "val_f2": policy_metrics["f2"],
                "val_pr_auc": policy_metrics["pr_auc"],
                "val_roc_auc": policy_metrics["roc_auc"],
            }
        )
    pd.DataFrame(imbalance_rows).to_csv(paths.evaluation / "imbalance_comparison.csv", index=False)
    (paths.evaluation / "resampling.md").write_text(
        "# Imbalance-policy comparison\\n\\nThe table is generated on training rows and validation probabilities only. SMOTE is never applied to validation or test rows.\\n\\n"
        + pd.DataFrame(imbalance_rows).to_string(index=False)
        + "\\n",
        encoding="utf-8",
    )
    write_json(
        paths.evaluation / "data_quality_report.json",
        {
            "profile": profile.as_dict(),
            "missingness": missingness_table(raw).to_dict(orient="records"),
            "dropped_columns": engineered_result.dropped_columns,
            "finite_matrix": True,
        },
    )

    results: list[dict[str, Any]] = []
    model_specs = build_model_specs(seed)
    max_models = config["training"].get("max_models")
    if max_models:
        model_specs = model_specs[: int(max_models)]
    for index, spec in enumerate(model_specs, start=1):
        LOGGER.info("[%s/%s] fitting %s", index, len(model_specs), spec.display_name)
        base = {
            "key": spec.key,
            "display_name": spec.display_name,
            "family": spec.family,
            "search_kind": spec.search_kind,
            "notes": spec.notes,
            "status": "failed",
            "error_reason": "",
        }
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                estimator, best_params, search_trials, train_seconds = _fit_one_model(
                    spec, matrices.x_train, split.y_train.to_numpy(), config, paths.evaluation / "tuning"
                )
            val_probabilities = probabilities_from_estimator(estimator, matrices.x_validation)
            threshold, sweep = select_threshold(
                split.y_validation.to_numpy(),
                val_probabilities,
                points=int(config["evaluation"].get("threshold_grid_points", 99)),
                precision_floor=float(config["evaluation"].get("precision_floor", 0.50)),
            )
            val_metrics = compute_metrics(split.y_validation.to_numpy(), val_probabilities, threshold.threshold)
            model_path = paths.models / f"{spec.key}.joblib"
            joblib.dump(estimator, model_path)
            row = {
                **base,
                **{f"val_{key}": value for key, value in val_metrics.items() if key not in ["threshold"]},
                "threshold": threshold.threshold,
                "threshold_metric": threshold.metric_name,
                "best_params": best_params,
                "search_trials": search_trials,
                "train_seconds": train_seconds,
                "artifact_kb": model_path.stat().st_size / 1024.0,
                "predict_ms_per_sample": np.nan,
                "validation_probabilities": val_probabilities,
                "y_validation": split.y_validation.to_numpy(),
                "threshold_sweep": sweep,
                "estimator": estimator,
                "status": "complete",
            }
            row["threshold_sweep"].to_csv(paths.evaluation / "curves" / f"{spec.key}_threshold_sweep.csv", index=False)
            pd.DataFrame(
                [
                    [val_metrics["true_negative"], val_metrics["false_positive"]],
                    [val_metrics["false_negative"], val_metrics["true_positive"]],
                ],
                columns=["predicted_legitimate", "predicted_fraud"],
                index=["actual_legitimate", "actual_fraud"],
            ).to_csv(paths.evaluation / "metrics" / f"{spec.key}_confusion_matrix.csv")
            from sklearn.metrics import precision_recall_curve, roc_curve

            fpr, tpr, _ = roc_curve(split.y_validation.to_numpy(), val_probabilities)
            precision_curve, recall_curve, _ = precision_recall_curve(split.y_validation.to_numpy(), val_probabilities)
            _save_curve_csv(
                paths.evaluation / "curves" / f"{spec.key}_roc.csv",
                fpr,
                tpr,
                "false_positive_rate",
                "true_positive_rate",
            )
            _save_curve_csv(
                paths.evaluation / "curves" / f"{spec.key}_pr.csv", recall_curve, precision_curve, "recall", "precision"
            )
            write_json(
                paths.evaluation / "metrics" / f"{spec.key}_metrics.json",
                {
                    key: _json_safe(value)
                    for key, value in row.items()
                    if key not in ["validation_probabilities", "y_validation", "threshold_sweep", "estimator"]
                },
            )
        except Exception as exc:  # Keep failed experiments visible.
            LOGGER.exception("Model %s failed", spec.key)
            row = {
                **base,
                "status": "failed",
                "error_reason": f"{type(exc).__name__}: {exc}",
                "best_params": {},
                "search_trials": 0,
                "train_seconds": 0.0,
                "val_f2": 0.0,
                "val_pr_auc": 0.0,
                "val_roc_auc": 0.5,
                "validation_probabilities": np.full(len(split.y_validation), split.y_train.mean()),
                "y_validation": split.y_validation.to_numpy(),
                "threshold": 0.5,
                "threshold_sweep": pd.DataFrame(),
                "estimator": None,
            }
        results.append(row)

    complete = [row for row in results if row["status"] == "complete"]
    if not complete:
        raise RuntimeError("Every model failed; inspect evaluation/pipeline.log")
    winner = sorted(
        complete,
        key=lambda row: (row.get("val_f2", 0.0), row.get("val_pr_auc", 0.0), -row.get("train_seconds", 1e9)),
        reverse=True,
    )[0]
    LOGGER.info(
        "Validation winner: %s (F2=%.4f, PR-AUC=%.4f)",
        winner["display_name"],
        winner.get("val_f2", 0.0),
        winner.get("val_pr_auc", 0.0),
    )
    (paths.evaluation / "test_unlock.log").write_text(
        f"Authorizer: pipeline self-certification\nRun: {run_id}\nTimestamp: {run_timestamp}\nPre-unlock condition: validation leaderboard and threshold selection complete; test labels not previously used for selection.\n",
        encoding="utf-8",
    )

    # One test evaluation per non-winning fitted model; the winner is evaluated only after the final refit.
    for row in results:
        if row["status"] != "complete" or row["key"] == winner["key"]:
            continue
        estimator = row["estimator"]
        start = time.perf_counter()
        test_probabilities = probabilities_from_estimator(estimator, matrices.x_test)
        elapsed = time.perf_counter() - start
        row["predict_ms_per_sample"] = elapsed / len(matrices.x_test) * 1000.0
        test_metrics = compute_metrics(split.y_test.to_numpy(), test_probabilities, float(row["threshold"]))
        row.update({f"test_{key}": value for key, value in test_metrics.items() if key != "threshold"})
        row["test_probabilities"] = test_probabilities
        row["test_metrics"] = test_metrics

    winner_spec = next(spec for spec in model_specs if spec.key == winner["key"])
    full_features = pd.concat([split.x_train, split.x_validation], axis=0)
    full_target = pd.concat([split.y_train, split.y_validation], axis=0)
    from src.features.preprocessing import build_preprocessor

    full_transformer, _, _ = build_preprocessor(full_features)
    full_x = np.asarray(full_transformer.fit_transform(full_features), dtype=float)
    test_x_final = np.asarray(full_transformer.transform(split.x_test), dtype=float)
    final_model = winner_spec.builder()
    if winner.get("best_params"):
        final_model.set_params(**winner["best_params"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        final_model.fit(full_x, full_target.to_numpy())
    start = time.perf_counter()
    final_test_probabilities = probabilities_from_estimator(final_model, test_x_final)
    elapsed = time.perf_counter() - start
    final_test_metrics = compute_metrics(split.y_test.to_numpy(), final_test_probabilities, float(winner["threshold"]))
    winner.update({f"test_{key}": value for key, value in final_test_metrics.items() if key != "threshold"})
    winner["test_probabilities"] = final_test_probabilities
    winner["test_metrics"] = final_test_metrics
    winner["predict_ms_per_sample"] = elapsed / len(test_x_final) * 1000.0
    winner["final_refit"] = True
    final_bundle = {
        "preprocessor": full_transformer,
        "model": final_model,
        "feature_names": matrices.feature_names,
        "threshold": winner["threshold"],
        "target_semantics": {"positive": 1, "negative": 0},
        "run_id": run_id,
        "model_key": winner["key"],
    }
    joblib.dump(final_bundle, paths.models / "best_model.joblib")
    winner["artifact_kb"] = (paths.models / "best_model.joblib").stat().st_size / 1024.0

    # Persist the canonical test metrics and confidence intervals.
    test_intervals = bootstrap_intervals(
        split.y_test.to_numpy(),
        final_test_probabilities,
        float(winner["threshold"]),
        seed=seed,
        replicates=int(config["training"].get("test_bootstrap_replicates", 300)),
    )
    write_json(
        paths.evaluation / "test_results.json",
        {
            "run_id": run_id,
            "model_key": winner["key"],
            "metrics": _json_safe(final_test_metrics),
            "bootstrap_intervals": _json_safe(test_intervals),
            "threshold": winner["threshold"],
        },
    )
    pd.DataFrame([{"metric": key, **value} for key, value in test_intervals.items()]).to_csv(
        paths.evaluation / "test_confidence_intervals.csv", index=False
    )

    # Calibration is a validation artifact applied to the final test probabilities.
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrated_validation = calibrator.fit_transform(winner["validation_probabilities"], split.y_validation.to_numpy())
    calibrated_test = calibrator.transform(final_test_probabilities)
    calibration_payload = {
        "before_brier": float(compute_metrics(split.y_validation, winner["validation_probabilities"])["brier"]),
        "after_brier": float(compute_metrics(split.y_validation, calibrated_validation)["brier"]),
        "before_ece": float(
            np.mean(np.abs(np.asarray(winner["validation_probabilities"]) - split.y_validation.to_numpy()))
        ),
        "after_ece": float(np.mean(np.abs(np.asarray(calibrated_validation) - split.y_validation.to_numpy()))),
        "test_calibrated_brier": float(compute_metrics(split.y_test, calibrated_test)["brier"]),
        "method": "isotonic regression fitted on validation probabilities",
        "run_id": run_id,
    }
    write_json(paths.evaluation / "calibration" / "calibration.json", calibration_payload)
    joblib.dump(calibrator, paths.models / "calibrator.joblib")
    calibration_record = calibration_plot(
        split.y_validation.to_numpy(),
        winner["validation_probabilities"],
        calibrated_validation,
        paths.images / "models" / "calibration_reliability.png",
        int(config["reporting"].get("dpi", 180)),
    )
    figure_records.append(calibration_record)
    threshold_record = _threshold_plot(
        winner, paths.images / "models" / "threshold_sweep_winner.png", int(config["reporting"].get("dpi", 180))
    )
    figure_records.append(threshold_record)

    # Winner importance and a JSON model card.
    importance = _model_feature_importance(
        winner["estimator"], matrices.feature_names, matrices.x_validation, split.y_validation.to_numpy(), seed
    )
    importance.to_csv(paths.evaluation / "explainability" / "winner_permutation_importance.csv", index=False)
    importance_record = feature_importance_plot(
        importance,
        paths.images / "models" / "feature_importance_permutation.png",
        f"Permutation importance — {winner['display_name']}",
        int(config["reporting"].get("dpi", 180)),
    )
    figure_records.append(importance_record)
    write_json(
        paths.evaluation / "explainability" / "winner_model_card.json",
        {
            "run_id": run_id,
            "model_key": winner["key"],
            "display_name": winner["display_name"],
            "threshold": winner["threshold"],
            "validation_metrics": {key: _json_safe(value) for key, value in winner.items() if key.startswith("val_")},
            "test_metrics": _json_safe(final_test_metrics),
            "top_features": importance.head(20).to_dict(orient="records"),
        },
    )

    # Fairness uses the raw audit columns only after probabilities are frozen.
    audit_frame = raw.loc[split.test_indices].reset_index(drop=True).copy()
    audit_frame["age_band"] = pd.cut(
        audit_frame["PatientAge"],
        bins=[-np.inf, 17, 30, 45, 60, 75, np.inf],
        labels=["0_17", "18_30", "31_45", "46_60", "61_75", "76_plus"],
    ).astype(str)
    fairness_table = fairness_metrics(
        audit_frame,
        split.y_test.to_numpy(),
        final_test_probabilities,
        float(winner["threshold"]),
        ["PatientGender", "age_band", "ClaimType", "PatientEmploymentStatus"],
        int(config["evaluation"].get("min_slice_size", 20)),
    )
    fairness_table.to_csv(paths.evaluation / "fairness" / "slice_metrics.csv", index=False)
    fairness_table.to_json(paths.evaluation / "fairness" / "slice_metrics.json", orient="records", indent=2)

    # Add test rows to failed/winner-safe JSON artifacts now that all outputs exist.
    for row in results:
        serializable = {
            key: _json_safe(value)
            for key, value in row.items()
            if key
            not in ["validation_probabilities", "y_validation", "threshold_sweep", "estimator", "test_probabilities"]
        }
        write_json(paths.evaluation / "metrics" / f"{row['key']}_metrics.json", serializable)
    leaderboard_rows = []
    for row in results:
        leaderboard_rows.append(
            {
                key: row.get(key)
                for key in [
                    "key",
                    "display_name",
                    "family",
                    "status",
                    "val_accuracy",
                    "val_precision",
                    "val_recall",
                    "val_f1",
                    "val_f2",
                    "val_roc_auc",
                    "val_pr_auc",
                    "val_mcc",
                    "val_brier",
                    "val_log_loss",
                    "test_accuracy",
                    "test_precision",
                    "test_recall",
                    "test_f1",
                    "test_f2",
                    "test_roc_auc",
                    "test_pr_auc",
                    "test_mcc",
                    "test_brier",
                    "test_log_loss",
                    "threshold",
                    "train_seconds",
                    "predict_ms_per_sample",
                    "artifact_kb",
                    "search_trials",
                    "error_reason",
                    "best_params",
                ]
            }
        )
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["val_f2", "val_pr_auc", "train_seconds"], ascending=[False, False, True]
    )
    leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))
    leaderboard["run_id"] = run_id
    leaderboard.to_csv(paths.evaluation / "leaderboard.csv", index=False)
    write_json(
        paths.evaluation / "run_manifest.json",
        {
            "run_id": run_id,
            "run_timestamp": run_timestamp,
            "config": config,
            "config_hash": stable_json_hash(config),
            "environment": environment_snapshot(),
            "input": {"path": str(input_path.relative_to(root)), "sha256": sha256_file(input_path)},
            "profile": profile.as_dict(),
            "split": split_stats,
            "dropped_columns": engineered_result.dropped_columns,
            "feature_count": len(matrices.feature_names),
            "model_count": len(results),
            "winner_key": winner["key"],
            "winner": {
                key: _json_safe(value)
                for key, value in winner.items()
                if key
                not in [
                    "validation_probabilities",
                    "y_validation",
                    "threshold_sweep",
                    "estimator",
                    "test_probabilities",
                ]
            },
            "figure_count": len(figure_records),
        },
    )
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "leaderboard": str((paths.evaluation / "leaderboard.csv").relative_to(root)),
            "winner": winner["key"],
            "figure_count": len(figure_records),
        },
    )
    figure_records.extend(generate_model_curves(results, paths.images, int(config["reporting"].get("dpi", 180))))

    context = {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "profile": profile.as_dict(),
        "config": config,
        "results": results,
        "winner": winner,
        "calibration": calibration_payload,
        "fairness": fairness_table,
        "lineage": engineered_result.lineage,
        "figure_records": figure_records,
        "feature_importance": importance,
        "split_stats": split_stats,
    }
    deliverables = build_all_documents(paths, context)
    write_json(
        paths.evaluation / "deliverables.json",
        {"run_id": run_id, "paths": deliverables, "figure_count": len(figure_records), "model_count": len(results)},
    )
    LOGGER.info("Completed run %s; winner=%s; deliverables=%s", run_id, winner["key"], deliverables)
    return context


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for direct module execution."""
    parser = argparse.ArgumentParser(description="Run the traditional ML fraud detection pipeline")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"), help="YAML configuration path")
    parser.add_argument("--dry-run", action="store_true", help="Print ordered stages without executing")
    parser.add_argument("--self-test", action="store_true", help="Run a quick dependency and metric self-test")
    args = parser.parse_args(argv)
    try:
        run_pipeline(args.config.resolve(), dry_run=args.dry_run, self_test=args.self_test)
        return 0
    except Exception as exc:
        logging.getLogger("medical_fraud").exception("Pipeline failed: %s", exc)
        print(f"Pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
