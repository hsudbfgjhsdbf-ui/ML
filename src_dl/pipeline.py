"""Single-command Approach 2 pipeline: deep tabular models plus XAI.

The implementation consumes Approach 1's frozen split and preprocessing
contract, trains five architectures across three seeds, evaluates validation
and locked-test metrics, produces common XAI artifacts, and builds dedicated
Approach 2 documentation, presentation, and PDF reports.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.isotonic import IsotonicRegression

from src.utils.paths import ProjectPaths, find_repository_root
from src.utils.reproducibility import environment_snapshot
from src_dl.data import DeepDataBundle, load_deep_data
from src_dl.evaluation import confidence_intervals, evaluate_fairness, evaluate_probabilities, predict_probabilities
from src_dl.models import MODEL_SPECS
from src_dl.plots import plot_calibration, plot_curves, plot_deep_leaderboard, plot_learning_curves, plot_xai
from src_dl.reporting import build_all_documents
from src_dl.training import TrainingConfig, train_model
from src_dl.utils import atomic_json, resolve_device, seed_everything
from src_dl.xai import faithfulness_score, local_dossier, occlusion_importance, stability_score

LOGGER = logging.getLogger("deep_fraud")


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate Approach 2 YAML configuration."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if len(config.get("seeds", [])) < 3:
        raise ValueError("Approach 2 requires at least three configured seeds")
    return config


def _safe(value: Any) -> Any:
    """Convert NumPy values to JSON-safe structures."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _metric_mean(values: list[dict[str, Any]], key: str) -> tuple[float, float]:
    """Return mean and population standard deviation across seeds."""
    array = np.asarray([float(item[key]) for item in values], dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def _write_evaluation(
    paths: ProjectPaths,
    run_id: str,
    data: DeepDataBundle,
    leaderboard: pd.DataFrame,
    winner: dict[str, Any],
    calibration: dict[str, Any],
    xai_rows: list[dict[str, Any]],
) -> None:
    """Write the dedicated Approach 2 evaluation hub and selection memo."""
    lines = [
        "# Evaluation hub — Approach 2 deep learning with XAI",
        "",
        f"**Run:** `{run_id}`  ",
        f"**Input split fingerprint:** `{data.split_fingerprint}`  ",
        f"**Feature fingerprint:** `{data.feature_fingerprint}`  ",
        f"**Device:** `{winner.get('device')}`  ",
        "**Positive class:** Fraud = 1.  ",
        "",
        "## 1. Comparability contract",
        "",
        "Approach 2 reuses Approach 1's supplied workbook, feature engineering, train-only transformer, and persisted split membership. Deep models differ in representation and optimization, not in the rows or metric definitions. The test set is evaluated only after validation selection.",
        "",
        "## 2. Deep leaderboard",
        "",
        leaderboard.to_string(index=False),
        "",
        f"**Selected deep model:** {winner['display_name']} (`{winner['key']}`), chosen by mean validation PR-AUC, then mean validation F2, then lower training time.",
        "",
        "## 3. Calibration",
        "",
        f"Calibration method: {calibration['method']}; validation Brier before/after: {calibration['before_brier']:.4f}/{calibration['after_brier']:.4f}; validation ECE proxy before/after: {calibration['before_ece']:.4f}/{calibration['after_ece']:.4f}.",
        "",
        "## 4. XAI",
        "",
        "All five architectures receive comparable occlusion importance, deletion-faithfulness, and jitter-stability artifacts. Native masks and attention tokens are auxiliary evidence and do not receive a scoring bonus. See `evaluation2/xai/`.",
        "",
        "## 5. Fairness",
        "",
        "The selected model is audited across gender, age band, claim type, and employment. Small slices are marked unstable; the synthetic-looking supplied workbook cannot support a claim of population fairness.",
        "",
        "## 6. Limitations",
        "",
        "The five deep models operate on the same transformed numeric/one-hot matrix. This is a transparent tabular bridge, not a document model or a full categorical entity-embedding system. Full Bayesian/Optuna search and later temporal/graph architectures are follow-on work.",
        "",
        "## 7. Artifacts",
        "",
        f"Run manifest: `evaluation2/runs/{run_id}/run_manifest.json`",
        "Training telemetry: `evaluation2/metrics/` and `images2/telemetry/`",
        "XAI: `evaluation2/xai/` and `images2/xai/`",
        "Presentation: `presentation2/approach_2_deep_learning_xai.pptx`",
        "Reports: `reports2/approach_2_project_report.pdf`, `reports2/approach_2_ieee_paper.pdf`",
    ]
    (paths.root / "evaluation2" / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    selection = f"""# Approach 2 selection memo

Run: `{run_id}`  
Selection policy: mean validation PR-AUC, then mean validation F2, then lower
training time. Test metrics are not used to select the model.

## Verdict

The selected deep model is **{winner['display_name']}** (`{winner['key']}`). Its
mean validation PR-AUC is **{winner['val_pr_auc']:.4f}**, mean validation F2 is
**{winner['val_f2']:.4f}**, and validation PR-AUC standard deviation across
three seeds is **{winner['val_pr_auc_std']:.4f}**.

The winner is a deep-learning comparison anchor, not a universal champion. The
same test split and target semantics are required for the later cross-approach
comparison.
"""
    (paths.root / "evaluation2" / "selection_memo.md").write_text(selection, encoding="utf-8")


def run_pipeline(config_path: Path, dry_run: bool = False, self_test: bool = False) -> dict[str, Any]:
    """Execute Approach 2 stages from contract validation through reports.

    Args:
        config_path: Path to `config_dl/default.yaml`.
        dry_run: Print stages without training.
        self_test: Validate torch model construction and exit.
    Returns:
        Actual run context used by reporting.
    """
    root = find_repository_root(config_path.parent)
    paths = ProjectPaths(root)
    paths.ensure()
    config = load_config(config_path)
    global LOGGER
    LOGGER = logging.getLogger("deep_fraud")
    LOGGER.setLevel(logging.INFO)
    if not LOGGER.handlers:
        LOGGER.addHandler(logging.StreamHandler())
    seed_everything(int(config["seed"]))
    device = resolve_device(config["training"].get("device", "auto"))
    stages = [
        "DL0 environment",
        "DL1 Approach 1 contract",
        "DL2 deep matrices",
        "DL3 five models × three seeds",
        "DL4 validation selection",
        "DL5 XAI",
        "DL6 calibration/fairness",
        "DL7 locked test",
        "DL8 docs/PPT/PDF",
    ]
    if dry_run:
        print("\n".join(stages))
        return {"status": "dry_run", "stages": stages}
    if self_test:
        from src_dl.models import build_model

        for spec in MODEL_SPECS:
            model = build_model(spec["key"], 16, config["models"])
            result = model(torch.zeros(4, 16))
            logits = result[0] if isinstance(result, tuple) else result
            assert logits.shape == (4,)
        print("Deep self-test passed: five architectures construct and emit one logit per row.")
        return {"status": "self_test_pass"}

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_timestamp = datetime.now(timezone.utc).strftime("%d-%m-%Y %H:%M:%S UTC")
    root_eval = root / "evaluation2"
    run_dir = root_eval / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for directory in [
        root_eval / "metrics",
        root_eval / "curves",
        root_eval / "tuning",
        root_eval / "xai",
        root_eval / "fairness",
        root_eval / "model_cards",
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    data = load_deep_data(root, config)
    LOGGER.info(
        "Approach 2 %s: rows=%s features=%s device=%s", run_id, data.profile["rows"], len(data.feature_names), device
    )
    pd.DataFrame({"feature": data.feature_names, "position": np.arange(len(data.feature_names))}).to_csv(
        root_eval / "feature_registry.csv", index=False
    )
    data.feature_lineage.to_csv(root_eval / "feature_lineage.csv", index=False)
    (root_eval / "dataset_contract.json").write_text(
        json.dumps(
            {
                "input_sha256": data.input_sha256,
                "split_fingerprint": data.split_fingerprint,
                "feature_fingerprint": data.feature_fingerprint,
                "profile": data.profile,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    training_config = TrainingConfig(
        epochs=int(config["training"]["epochs"]),
        patience=int(config["training"]["patience"]),
        batch_size=int(config["training"]["batch_size"]),
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
        reconstruction_weight=float(config["models"].get("autoencoder_reconstruction_weight", 0.10)),
        device=device,
    )
    seed_outputs: dict[str, list[dict[str, Any]]] = {}
    row_records: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        key = spec["key"]
        seed_outputs[key] = []
        for seed in config["seeds"]:
            LOGGER.info("Training %s seed %s", key, seed)
            output = train_model(
                key,
                len(data.feature_names),
                data.x_train,
                data.y_train,
                data.x_validation,
                data.y_validation,
                config["models"],
                training_config,
                int(seed),
                root / "checkpoints2",
                root_eval / "metrics",
            )
            validation_metrics, sweep = evaluate_probabilities(
                data.y_validation,
                output["validation_probabilities"],
                int(config["evaluation"]["threshold_grid_points"]),
                float(config["evaluation"]["precision_floor"]),
            )
            output["validation_metrics"] = validation_metrics
            output["threshold_sweep"] = sweep
            output["model_key"] = key
            output["display_name"] = spec["display_name"]
            output["family"] = spec["family"]
            seed_outputs[key].append(output)
        mean_probs = np.mean([item["validation_probabilities"] for item in seed_outputs[key]], axis=0)
        val_metrics, sweep = evaluate_probabilities(
            data.y_validation,
            mean_probs,
            int(config["evaluation"]["threshold_grid_points"]),
            float(config["evaluation"]["precision_floor"]),
        )
        seed_metrics = [item["validation_metrics"] for item in seed_outputs[key]]
        pr_mean, pr_std = _metric_mean(seed_metrics, "pr_auc")
        f2_mean, f2_std = _metric_mean(seed_metrics, "f2")
        best_seed_output = max(seed_outputs[key], key=lambda item: item["validation_metrics"]["pr_auc"])
        record = {
            "key": key,
            "display_name": spec["display_name"],
            "family": spec["family"],
            "status": "complete",
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_f2": f2_mean,
            "val_f2_std": f2_std,
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": pr_mean,
            "val_pr_auc_std": pr_std,
            "threshold": val_metrics["threshold"],
            "threshold_metric": val_metrics["threshold_metric"],
            "best_seed": best_seed_output["seed"],
            "training_seconds": float(sum(item["training_seconds"] for item in seed_outputs[key])),
            "parameter_count": int(best_seed_output["parameter_count"]),
            "epochs": int(best_seed_output["best_epoch"]),
            "device": str(device),
            "y_validation": data.y_validation.astype(int),
            "validation_probabilities": mean_probs,
            "threshold_sweep": sweep,
            "best_model": best_seed_output["model"],
        }
        record["val_metrics"] = val_metrics
        row_records.append(record)
        write_json = {
            k: _safe(v)
            for k, v in record.items()
            if k not in ["y_validation", "validation_probabilities", "threshold_sweep", "best_model", "val_metrics"]
        }
        atomic_json(root_eval / "metrics" / f"{key}_metrics.json", write_json)
        sweep.to_csv(root_eval / "curves" / f"{key}_threshold_sweep.csv", index=False)
        for item in seed_outputs[key]:
            item["telemetry"].to_csv(root_eval / "metrics" / f"{key}_s{item['seed']}_epoch_log.csv", index=False)
        plot_learning_curves(
            [item["telemetry"] for item in seed_outputs[key]],
            key,
            root / "images2" / "telemetry" / f"{key}_learning_curves.png",
        )

    leaderboard = (
        pd.DataFrame(
            [
                {
                    k: row.get(k)
                    for k in [
                        "key",
                        "display_name",
                        "family",
                        "status",
                        "val_accuracy",
                        "val_precision",
                        "val_recall",
                        "val_f1",
                        "val_f2",
                        "val_f2_std",
                        "val_roc_auc",
                        "val_pr_auc",
                        "val_pr_auc_std",
                        "threshold",
                        "best_seed",
                        "training_seconds",
                        "parameter_count",
                        "epochs",
                        "device",
                    ]
                }
                for row in row_records
            ]
        )
        .sort_values(["val_pr_auc", "val_f2", "training_seconds"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))
    leaderboard["run_id"] = run_id
    leaderboard.to_csv(root_eval / "leaderboard.csv", index=False)
    winner_key = str(leaderboard.iloc[0]["key"])
    winner = next(row for row in row_records if row["key"] == winner_key)
    LOGGER.info("Deep validation winner: %s", winner_key)
    (root_eval / "test_unlock.log").write_text(
        f"Run: {run_id}\nTimestamp: {run_timestamp}\nUnlock: validation leaderboard, XAI assets, threshold, calibration and fairness artifacts complete before test evaluation.\n",
        encoding="utf-8",
    )

    # XAI for every architecture uses its best validation seed and the same source feature order.
    xai_summary: list[dict[str, Any]] = []
    for row in row_records:
        importance = occlusion_importance(
            row["best_model"], data.x_validation, data.feature_names, device, int(config["training"]["xai_sample_size"])
        )
        importance.to_csv(root_eval / "xai" / f"{row['key']}_occlusion_importance.csv", index=False)
        faith = faithfulness_score(
            row["best_model"],
            data.x_validation,
            importance,
            device,
            sample_size=int(config["training"]["xai_sample_size"]),
        )
        stability = stability_score(
            row["best_model"],
            data.x_validation,
            importance,
            device,
            sample_size=int(config["training"]["xai_sample_size"]),
        )
        dossier = local_dossier(row["best_model"], data.x_validation, data.feature_names, importance, device, 0)
        atomic_json(root_eval / "xai" / f"{row['key']}_faithfulness.json", faith)
        atomic_json(root_eval / "xai" / f"{row['key']}_stability.json", stability)
        atomic_json(root_eval / "xai" / f"{row['key']}_local_dossier.json", dossier)
        row["faithfulness"] = faith["faithfulness_at_5"]
        row["stability"] = stability["jaccard_mean"]
        row["importance"] = importance
        xai_summary.append(
            {
                "key": row["key"],
                "faithfulness": row["faithfulness"],
                "stability": row["stability"],
                "top_feature": importance.iloc[0]["feature"],
            }
        )
    plot_deep_leaderboard(leaderboard, root / "images2" / "models" / "validation_comparison.png")
    plot_curves(row_records, root / "images2" / "models")
    plot_xai(winner["importance"], winner_key, root / "images2" / "xai" / f"{winner_key}_occlusion_importance.png")

    # Test evaluation is performed once per architecture using its best validation seed.
    for row in row_records:
        best_seed_output = max(seed_outputs[row["key"]], key=lambda item: item["validation_metrics"]["pr_auc"])
        test_prob = predict_probabilities(best_seed_output["model"], data.x_test, device)
        # The selected validation threshold, not a test-optimized threshold, is used in the final row.
        from src.evaluation.metrics import compute_metrics

        test_metrics = compute_metrics(data.y_test.astype(int), test_prob, float(row["threshold"]))
        row["test_probabilities"] = test_prob
        row["test_metrics"] = test_metrics
        for key, value in test_metrics.items():
            row[f"test_{key}"] = value
        atomic_json(
            root_eval / "metrics" / f"{row['key']}_test_metrics.json", {k: _safe(v) for k, v in test_metrics.items()}
        )
    winner = next(row for row in row_records if row["key"] == winner_key)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    before = winner["validation_probabilities"]
    after = calibrator.fit_transform(before, data.y_validation.astype(int))
    calibrated_test = calibrator.transform(winner["test_probabilities"])
    calibration = {
        "method": "isotonic regression on validation probabilities",
        "before_brier": float(compute_metrics(data.y_validation, before)["brier"]),
        "after_brier": float(compute_metrics(data.y_validation, after)["brier"]),
        "before_ece": float(np.mean(np.abs(before - data.y_validation))),
        "after_ece": float(np.mean(np.abs(after - data.y_validation))),
        "test_calibrated_brier": float(compute_metrics(data.y_test, calibrated_test)["brier"]),
        "run_id": run_id,
    }
    atomic_json(root_eval / "calibration.json", calibration)
    joblib.dump(calibrator, root / "checkpoints2" / "calibrator.joblib")
    plot_calibration(
        data.y_validation.astype(int), before, after, root / "images2" / "models" / "calibration_reliability.png"
    )
    fairness = evaluate_fairness(
        data.test_audit,
        data.y_test,
        winner["test_probabilities"],
        float(winner["threshold"]),
        int(config["evaluation"]["min_slice_size"]),
    )
    fairness.to_csv(root_eval / "fairness" / "winner_slice_metrics.csv", index=False)
    intervals = confidence_intervals(
        data.y_test,
        winner["test_probabilities"],
        float(winner["threshold"]),
        int(config["seed"]),
        int(config["training"]["bootstrap_replicates"]),
    )
    atomic_json(root_eval / "winner_test_intervals.json", intervals)
    leaderboard_final = (
        pd.DataFrame(
            [
                {
                    k: row.get(k)
                    for k in [
                        "key",
                        "display_name",
                        "family",
                        "status",
                        "val_f2",
                        "val_pr_auc",
                        "val_pr_auc_std",
                        "val_roc_auc",
                        "test_accuracy",
                        "test_precision",
                        "test_recall",
                        "test_f1",
                        "test_f2",
                        "test_roc_auc",
                        "test_pr_auc",
                        "threshold",
                        "training_seconds",
                        "parameter_count",
                        "faithfulness",
                        "stability",
                    ]
                }
                for row in row_records
            ]
        )
        .sort_values(["val_pr_auc", "val_f2", "training_seconds"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    leaderboard_final.insert(0, "rank", np.arange(1, len(leaderboard_final) + 1))
    leaderboard_final["run_id"] = run_id
    leaderboard_final.to_csv(root_eval / "leaderboard.csv", index=False)
    classical_path = root / "evaluation" / "leaderboard.csv"
    if classical_path.exists():
        classical = pd.read_csv(classical_path)
        comparison = pd.DataFrame(
            [
                {
                    "approach": "traditional_ml",
                    "model": str(row["key"]),
                    "pr_auc": row.get("val_pr_auc"),
                    "f2": row.get("val_f2"),
                    "test_f2": row.get("test_f2"),
                }
                for _, row in classical.head(3).iterrows()
            ]
            + [
                {
                    "approach": "deep_learning",
                    "model": str(row["key"]),
                    "pr_auc": row.get("val_pr_auc"),
                    "f2": row.get("val_f2"),
                    "test_f2": row.get("test_f2"),
                }
                for _, row in leaderboard_final.head(5).iterrows()
            ]
        )
        comparison.to_csv(root_eval / "classical_vs_deep_comparison.csv", index=False)
    winner_for_report = {
        k: _safe(v)
        for k, v in winner.items()
        if k
        not in [
            "y_validation",
            "validation_probabilities",
            "threshold_sweep",
            "best_model",
            "importance",
            "test_probabilities",
        ]
    }
    winner_for_report["key"] = winner_key
    winner_for_report["display_name"] = winner["display_name"]
    winner_for_report["val_pr_auc"] = float(winner["val_pr_auc"])
    winner_for_report["val_f2"] = float(winner["val_f2"])
    winner_for_report["test_f2"] = float(winner["test_f2"])
    winner_for_report["device"] = str(device)
    _write_evaluation(paths, run_id, data, leaderboard_final, winner_for_report, calibration, xai_summary)
    manifest = {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "config": config,
        "environment": environment_snapshot(),
        "device": str(device),
        "input_sha256": data.input_sha256,
        "split_fingerprint": data.split_fingerprint,
        "feature_fingerprint": data.feature_fingerprint,
        "feature_count": len(data.feature_names),
        "model_count": len(row_records),
        "winner_key": winner_key,
        "winner": winner_for_report,
        "xai": xai_summary,
    }
    atomic_json(root_eval / "run_manifest.json", manifest)
    atomic_json(
        run_dir / "run_manifest.json",
        {"run_id": run_id, "winner_key": winner_key, "split_fingerprint": data.split_fingerprint},
    )
    context = {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "config": config,
        "data": data,
        "rows": row_records,
        "leaderboard": leaderboard_final,
        "winner": winner,
        "winner_for_report": winner_for_report,
        "calibration": calibration,
        "fairness": fairness,
        "xai_summary": xai_summary,
        "device": str(device),
    }
    deliverables = build_all_documents(root, context)
    atomic_json(
        root_eval / "deliverables.json",
        {"run_id": run_id, "paths": deliverables, "model_count": len(row_records), "xai_count": len(xai_summary)},
    )
    return context


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Approach 2."""
    parser = argparse.ArgumentParser(description="Run deep learning with XAI for the medical insurance fraud project")
    parser.add_argument("--config", type=Path, default=Path("config_dl/default.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_pipeline(args.config.resolve(), dry_run=args.dry_run, self_test=args.self_test)
        return 0
    except Exception as exc:
        LOGGER.exception("Approach 2 failed: %s", exc)
        print(f"Approach 2 failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
