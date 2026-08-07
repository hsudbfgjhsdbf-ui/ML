"""Held-out metrics, fairness, INR-cost and significance analyses for fraud models."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, fbeta_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

from src.models import TrainedModel, optimal_f2_threshold
from src.preprocessing import build_resampler
from src.utils import write_json


def _safe_auc(metric: Any, y_true: np.ndarray, probability: np.ndarray) -> float:
    """Calculate an AUC-like metric while gracefully handling a one-class edge case."""
    try:
        return float(metric(y_true, probability))
    except ValueError:
        return float("nan")


def metric_row(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Calculate the complete specified binary-classification metric suite.

    Args:
        y_true: Ground-truth binary labels.
        probability: Fraud probabilities in [0, 1].
        threshold: Operational probability threshold chosen without test labels.

    Returns:
        Test metrics and confusion-matrix counts.
    """
    predicted = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "f2": float(fbeta_score(y_true, predicted, beta=2, zero_division=0)),
        "auc_roc": _safe_auc(roc_auc_score, y_true, probability),
        "auc_pr": _safe_auc(average_precision_score, y_true, probability),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp),
    }


def calculate_inr_cost(raw_test: pd.DataFrame, predicted: np.ndarray, config: dict[str, Any]) -> dict[str, float | int]:
    """Translate false negatives and false positives into an illustrative INR cost matrix.

    The fraud loss is the actual synthetic claimed amount on an approved fraud. The
    false-positive cost is an explicit review/customer-friction proxy, not a regulatory fine.

    Args:
        raw_test: Held-out raw records aligned to predictions.
        predicted: Binary flag predictions.
        config: Configuration containing INR cost assumptions.

    Returns:
        Count and cost breakdown in INR.
    """
    y_true = raw_test["is_fraud"].to_numpy(dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    false_negatives = (y_true == 1) & (predicted == 0)
    false_positives = (y_true == 0) & (predicted == 1)
    fraud_loss = float(raw_test.loc[false_negatives, "claim_amount_inr"].sum())
    fp_cost = float(false_positives.sum() * config["business_costs_inr"]["false_positive_review_cost"])
    return {
        "false_negative_count": int(false_negatives.sum()), "false_positive_count": int(false_positives.sum()),
        "false_negative_paid_fraud_loss_inr": round(fraud_loss, 2), "false_positive_review_and_friction_cost_inr": round(fp_cost, 2),
        "total_illustrative_cost_inr": round(fraud_loss + fp_cost, 2),
        "assumption": "FN cost equals synthetic claim amount; FP cost is configured review/friction proxy, not a real financial liability.",
    }


def evaluate_model(trained: TrainedModel, X_test: np.ndarray, y_test: np.ndarray, raw_test: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Time predictions and evaluate one trained model exactly once on the held-out test set.

    Args:
        trained: Model selected using train/CV and validation data.
        X_test: Selected transformed test matrix.
        y_test: Held-out labels.
        raw_test: Held-out records for cost/fairness analyses.
        config: Project configuration.

    Returns:
        Benchmark record, fraud probabilities and thresholded predictions.
    """
    test_x = trained.nonnegative_scaler.transform(X_test) if trained.nonnegative_scaler else X_test
    start = time.perf_counter()
    probability = trained.estimator.predict_proba(test_x)[:, 1]
    total_seconds = time.perf_counter() - start
    prediction = (probability >= trained.threshold).astype(int)
    metrics = metric_row(np.asarray(y_test), probability, trained.threshold)
    metrics.update({
        "algorithm": trained.name, "threshold": trained.threshold, "validation_f2": trained.validation_f2,
        "cv_f2_mean": trained.cv_f2_mean, "cv_f2_std": trained.cv_f2_std, "training_time_seconds": trained.training_seconds,
        "prediction_time_per_sample_ms": total_seconds / max(len(X_test), 1) * 1000,
        "tuned_hyperparameters": trained.tuned_hyperparameters, "best_hyperparameters": trained.best_params,
        "cv_f2_scores": trained.cv_f2_scores, "cost_matrix_inr": calculate_inr_cost(raw_test, prediction, config),
    })
    return metrics, probability, prediction


def fairness_table(raw_test: pd.DataFrame, prediction: np.ndarray, model_name: str) -> pd.DataFrame:
    """Calculate group accuracy, FPR, FNR, precision, recall and selection rate.

    The function evaluates demographic group outcomes; it does not state that any group
    should have a particular fraud prevalence. Small groups are retained but marked by N.

    Args:
        raw_test: Test records with demographic columns and true target.
        prediction: Model binary flag predictions aligned to rows.
        model_name: Name stored in output rows.

    Returns:
        Long-form fairness audit DataFrame.
    """
    frame = raw_test.copy()
    frame["prediction"] = prediction
    frame["age_group"] = pd.cut(frame["age"], bins=[-1, 17, 34, 49, 64, 150], labels=["Child (0–17)", "Young adult (18–34)", "Adult (35–49)", "Older adult (50–64)", "Senior (65+)"])
    rows: list[dict[str, Any]] = []
    for dimension in ["gender", "age_group", "state", "income_bracket", "treatment_type"]:
        if dimension not in frame:
            continue
        for group, subset in frame.groupby(dimension, observed=False, dropna=False):
            y = subset["is_fraud"].to_numpy(dtype=int)
            p = subset["prediction"].to_numpy(dtype=int)
            tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
            rows.append({
                "model": model_name, "dimension": dimension, "group": str(group), "n": len(subset),
                "accuracy": (tp + tn) / max(len(subset), 1), "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1),
                "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "selection_rate": p.mean(),
                "small_group_warning": bool(len(subset) < 30),
            })
    return pd.DataFrame(rows)


def calibration_table(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Calculate reliability-diagram points for a probabilistic classifier.

    Args:
        y_true: Binary labels.
        probability: Predicted fraud probabilities.
        bins: Equal-width calibration bins.

    Returns:
        Frame with mean predicted probability and observed fraud frequency.
    """
    observed, predicted = calibration_curve(y_true, probability, n_bins=bins, strategy="uniform")
    return pd.DataFrame({"mean_predicted_probability": predicted, "observed_fraud_rate": observed})


def compare_imbalance_strategies(X_train: np.ndarray, y_train: np.ndarray, X_validation: np.ndarray, y_validation: np.ndarray, config: dict[str, Any]) -> pd.DataFrame:
    """Compare requested imbalance strategies with one interpretable logistic baseline.

    This isolates sampling choice before expensive all-model training. Each sampler is fit
    only to the training partition; validation chooses an F2 threshold independently.

    Args:
        X_train: Selected train matrix.
        y_train: Train target.
        X_validation: Selected validation matrix.
        y_validation: Validation target.
        config: Project configuration.

    Returns:
        Strategy comparison table ranked by validation F2.
    """
    rows = []
    for strategy in ["class_weight", "random_under", "tomek", "smote", "smoteenn"]:
        sampler = build_resampler(strategy, int(config["project"]["random_seed"]), int(config["preprocessing"]["smote_k_neighbors"]))
        train_x, train_y = (sampler.fit_resample(X_train, y_train) if sampler else (X_train, y_train))
        estimator = LogisticRegression(max_iter=1800, solver="liblinear", class_weight="balanced" if strategy == "class_weight" else None, random_state=int(config["project"]["random_seed"]))
        estimator.fit(train_x, train_y)
        probabilities = estimator.predict_proba(X_validation)[:, 1]
        threshold, f2 = optimal_f2_threshold(y_validation, probabilities)
        result = metric_row(y_validation, probabilities, threshold)
        rows.append({"strategy": strategy, "training_rows_after_sampling": len(train_x), "threshold": threshold, "validation_f2": f2, "validation_recall": result["recall"], "validation_precision": result["precision"]})
    return pd.DataFrame(rows).sort_values(["validation_f2", "validation_recall"], ascending=False).reset_index(drop=True)


def significance_tests(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Run McNemar/Wilcoxon-style pairwise comparisons using saved prediction and CV data.

    Args:
        benchmark: Benchmark rows containing predictions in private object columns.

    Returns:
        Pairwise p-values against the F2-leading model. Values are descriptive because
        synthetic claims are not independent real-world insurer observations.
    """
    leader = benchmark.sort_values(["f2", "auc_roc"], ascending=False).iloc[0]
    y_true = np.asarray(leader["_y_true"], dtype=int)
    leader_correct = np.asarray(leader["_prediction"], dtype=int) == y_true
    rows = []
    for _, candidate in benchmark.iterrows():
        if candidate["algorithm"] == leader["algorithm"]:
            continue
        candidate_correct = np.asarray(candidate["_prediction"], dtype=int) == y_true
        b = int(np.sum(leader_correct & ~candidate_correct))
        c = int(np.sum(~leader_correct & candidate_correct))
        mcnemar_p = float(binomtest(min(b, c), n=b+c, p=.5).pvalue) if (b+c) else 1.0
        leader_cv, candidate_cv = np.asarray(leader["cv_f2_scores"], dtype=float), np.asarray(candidate["cv_f2_scores"], dtype=float)
        try:
            wilcoxon_p = float(wilcoxon(leader_cv, candidate_cv, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            wilcoxon_p = float("nan")
        rows.append({"reference_model": leader["algorithm"], "compared_model": candidate["algorithm"], "mcnemar_discordant_leader_only": b, "mcnemar_discordant_candidate_only": c, "mcnemar_exact_p": mcnemar_p, "wilcoxon_cv_f2_p": wilcoxon_p, "significant_at_0_05": bool(mcnemar_p < .05)})
    return pd.DataFrame(rows)


def save_evaluation_artifacts(benchmark: pd.DataFrame, fairness: pd.DataFrame, imbalance: pd.DataFrame, config: dict[str, Any]) -> None:
    """Persist tables while excluding in-memory prediction arrays from CSV/JSON output.

    Args:
        benchmark: Full benchmark frame, including internal arrays.
        fairness: Long-form fairness table.
        imbalance: Sampling-strategy comparison.
        config: Loaded configuration.
    """
    destination = Path(config["paths"]["evaluation_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    public = benchmark.drop(columns=[column for column in benchmark if column.startswith("_")], errors="ignore").copy()
    for col in ["best_hyperparameters", "cv_f2_scores", "cost_matrix_inr"]:
        if col in public:
            public[col] = public[col].map(lambda value: str(value))
    public.sort_values(["f2", "auc_roc"], ascending=False).to_csv(destination / "benchmark_results.csv", index=False)
    fairness.to_csv(destination / "fairness_by_group.csv", index=False)
    imbalance.to_csv(destination / "imbalance_strategy_comparison.csv", index=False)
    leader = public.sort_values(["f2", "auc_roc"], ascending=False).iloc[0].to_dict()
    write_json(destination / "best_model_summary.json", leader)
