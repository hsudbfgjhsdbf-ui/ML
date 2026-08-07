"""One-command end-to-end runner for Approach 1 traditional ML.

Example:
    .venv/bin/python -m src.train --regenerate-data

The command audits the bundled file, creates the documented fallback when needed,
trains/tunes all registry models, evaluates the held-out test split once, writes visual
and academic deliverables, and records a verification manifest.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data_loading import audit_raw_workbook, load_or_create_study_data
from src.evaluate import (
    calibration_table, compare_imbalance_strategies, evaluate_model, fairness_table,
    save_evaluation_artifacts, significance_tests,
)
from src.models import fit_tuned_model, model_registry
from src.preprocessing import create_data_bundle, export_data_dictionary, save_processed_partitions
from src.reporting import generate_ieee_pdf, generate_markdown, generate_presentation
from src.utils import configure_logging, ensure_directories, load_config, set_global_seed, write_json
from src.visualize import generate_eda_plots, generate_fairness_and_calibration_plots, generate_model_plots


def _safe_stem(name: str) -> str:
    """Create a deterministic safe model artifact filename from its display name."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _artifact_manifest(config: dict[str, Any], expected: list[Path]) -> dict[str, Any]:
    """Check final artifact existence and create an honest verification manifest."""
    results = [{"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0} for path in expected]
    return {"all_expected_artifacts_exist": all(item["exists"] for item in results), "artifacts": results}


def run(config_path: str, regenerate_data: bool, fast: bool) -> int:
    """Execute the complete traditional ML study and academic deliverable generation.

    Args:
        config_path: YAML configuration path.
        regenerate_data: Rebuild deterministic synthetic data before running.
        fast: Development-only flag reducing CV/search breadth; never use it for final claims.

    Returns:
        Process exit code (zero on completed run).
    """
    config = load_config(config_path)
    if fast:
        # Explicitly recorded in metadata. Final academic results must use the default 5 folds.
        config["training"]["cv_folds"] = 3
        config["training"]["search_iterations"] = 2
        config["training"]["max_training_rows_for_svm"] = 2500
    ensure_directories(config)
    logger = configure_logging("traditional_ml")
    seed = int(config["project"]["random_seed"])
    set_global_seed(seed)
    logger.info("Starting Approach 1 pipeline (fast=%s, seed=%s).", fast, seed)

    source_audit = audit_raw_workbook(config["paths"]["raw_workbook"])
    write_json(Path(config["paths"]["raw_workbook"]).with_name("source_data_audit.json"), source_audit)
    raw = load_or_create_study_data(config, regenerate=regenerate_data)
    logger.info("Loaded synthetic study population: %s rows before duplicate cleaning.", len(raw))
    export_data_dictionary(raw, config)
    bundle = create_data_bundle(raw, config)
    save_processed_partitions(bundle, config)
    model_dir = Path(config["paths"]["models_dir"]); model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.preprocessor, model_dir / "approach_1_preprocessor.joblib")
    joblib.dump(bundle.selector, model_dir / "approach_1_feature_selector.joblib")
    write_json(model_dir / "approach_1_feature_manifest.json", {
        "selected_feature_names": bundle.selected_feature_names,
        "quality_report": bundle.quality_report,
        "config": config,
    })
    imbalance = compare_imbalance_strategies(bundle.X_train, bundle.y_train.to_numpy(), bundle.X_validation, bundle.y_validation.to_numpy(), config)
    logger.info("Completed train-only imbalance strategy comparison.")

    records: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}
    registry = model_registry(bundle.y_train.to_numpy(), seed, int(config["training"]["n_jobs"]))
    failures: list[dict[str, str]] = []
    for position, spec in enumerate(registry, 1):
        logger.info("[%s/%s] Training %s", position, len(registry), spec.name)
        try:
            trained = fit_tuned_model(spec, bundle.X_train, bundle.y_train.to_numpy(), bundle.X_validation, bundle.y_validation.to_numpy(), config, seed + position)
            artifact_path = model_dir / f"{_safe_stem(spec.name)}.joblib"
            payload = {
                "estimator": trained.estimator, "nonnegative_scaler": trained.nonnegative_scaler,
                "threshold": trained.threshold, "selected_feature_names": bundle.selected_feature_names,
                "best_hyperparameters": trained.best_params, "validation_f2": trained.validation_f2,
                "warning": "Synthetic-study screening model only. Do not use for automatic claim denial.",
            }
            joblib.dump(payload, artifact_path)
            evaluation, probability, prediction = evaluate_model(trained, bundle.X_test, bundle.y_test.to_numpy(), bundle.raw_test, config)
            evaluation["model_size_kb"] = artifact_path.stat().st_size / 1024.0
            evaluation["artifact_path"] = str(artifact_path)
            evaluation["_probability"] = probability
            evaluation["_prediction"] = prediction
            evaluation["_y_true"] = bundle.y_test.to_numpy()
            records.append(evaluation)
            trained_models[spec.name] = trained
            write_json(model_dir / f"{_safe_stem(spec.name)}.metadata.json", {
                key: value for key, value in evaluation.items() if not key.startswith("_")
            })
            logger.info("Completed %s: held-out F2=%.3f, recall=%.3f", spec.name, evaluation["f2"], evaluation["recall"])
        except Exception as exc:  # Keep an academic run informative if a platform-specific model fails.
            logger.exception("Model %s failed: %s", spec.name, exc)
            failures.append({"algorithm": spec.name, "error": f"{type(exc).__name__}: {exc}"})
    if len(records) < 12:
        raise RuntimeError(f"Only {len(records)} models completed; expected at least 12. Failures: {failures}")
    benchmark = pd.DataFrame(records).sort_values(["f2", "auc_roc"], ascending=False).reset_index(drop=True)
    best_name = benchmark.iloc[0]["algorithm"]
    best_prediction = benchmark.iloc[0]["_prediction"]
    fairness = fairness_table(bundle.raw_test, best_prediction, str(best_name))
    calibration = calibration_table(bundle.y_test.to_numpy(), benchmark.iloc[0]["_probability"])
    significance = significance_tests(benchmark)
    Path(config["paths"]["evaluation_dir"]).mkdir(parents=True, exist_ok=True)
    significance.to_csv(Path(config["paths"]["evaluation_dir"]) / "significance_tests.csv", index=False)
    calibration.to_csv(Path(config["paths"]["evaluation_dir"]) / "best_model_calibration.csv", index=False)
    # Claim-level probabilities are auditable test artifacts, synthetic and non-identifying.
    prediction_export = bundle.raw_test[["claim_id", "is_fraud", "claim_amount_inr"]].copy()
    for _, row in benchmark.iterrows():
        column = _safe_stem(str(row["algorithm"]))
        prediction_export[f"{column}_probability"] = row["_probability"]
        prediction_export[f"{column}_prediction"] = row["_prediction"]
    prediction_export.to_csv(Path(config["paths"]["evaluation_dir"]) / "held_out_predictions.csv", index=False)
    save_evaluation_artifacts(benchmark, fairness, imbalance, config)
    # Save a complete serving bundle for the F2-leading model.
    best_trained = trained_models[str(best_name)]
    joblib.dump({"preprocessor": bundle.preprocessor, "selector": bundle.selector, "model": best_trained.estimator, "nonnegative_scaler": best_trained.nonnegative_scaler, "threshold": best_trained.threshold, "feature_names": bundle.selected_feature_names, "model_name": best_name}, model_dir / "approach_1_best_serving_bundle.joblib")

    cleaned_for_eda = pd.concat([bundle.raw_train, bundle.raw_validation, bundle.raw_test], axis=0)
    visual_root = config["paths"]["visualization_dir"]
    assets = generate_eda_plots(cleaned_for_eda, visual_root)
    assets += generate_model_plots(benchmark, bundle.selected_feature_names, trained_models, bundle.X_test, bundle.y_test.to_numpy(), visual_root)
    assets += generate_fairness_and_calibration_plots(fairness, calibration, visual_root)
    logger.info("Generated %s visualization assets.", len(assets))
    documentation_path, evaluation_path = generate_markdown(config, source_audit, bundle.quality_report, benchmark, fairness, imbalance, significance)
    presentation_path = generate_presentation(config, benchmark, bundle.quality_report, source_audit)
    report_path = generate_ieee_pdf(config, benchmark, source_audit, assets)
    expected = [
        Path(config["paths"]["synthetic_data"]), Path(config["paths"]["processed_dir"]) / "train_claims.csv",
        Path(config["paths"]["evaluation_dir"]) / "benchmark_results.csv", Path(config["paths"]["evaluation_dir"]) / "approach_1_evaluation_report.md",
        documentation_path, presentation_path, report_path, model_dir / "approach_1_best_serving_bundle.joblib",
    ] + assets
    manifest = _artifact_manifest(config, expected)
    manifest.update({"fast_mode": fast, "model_failures": failures, "completed_models": list(benchmark["algorithm"]), "best_model": str(best_name)})
    write_json(Path(config["paths"]["evaluation_dir"]) / "verification_manifest.json", manifest)
    logger.info("Approach 1 complete. Best model: %s. All expected artifacts exist=%s", best_name, manifest["all_expected_artifacts_exist"])
    return 0


def main() -> int:
    """Parse command-line options and run the Approach-1 pipeline."""
    parser = argparse.ArgumentParser(description="Run reproducible traditional ML fraud-screening baseline.")
    parser.add_argument("--config", default="configs/traditional_ml.yaml", help="YAML configuration path.")
    parser.add_argument("--regenerate-data", action="store_true", help="Regenerate deterministic synthetic fallback data.")
    parser.add_argument("--fast", action="store_true", help="Development-only reduced CV/search. Not for final results.")
    args = parser.parse_args()
    try:
        return run(args.config, args.regenerate_data, args.fast)
    except Exception as exc:
        configure_logging("traditional_ml").exception("Pipeline aborted: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
