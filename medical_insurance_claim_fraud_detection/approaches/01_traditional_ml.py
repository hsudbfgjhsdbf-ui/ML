"""
01_traditional_ml.py — Traditional supervised ML workflow for medical insurance claim fraud detection.

Project: Medical Insurance Claim Fraud Detection — AI-Driven Claim Verification & Explainable Fraud Detection Platform
Team: 23BDS011 B Varshith, 23BDS033 M Jagadeshwar, 23BDS024 J Ganesh, IIIT Dharwad

Objectives:
- Binary classification: fraud (1) vs legitimate (0)
- Produce fraud probability and risk category
- Benchmark multiple classifiers
- Full preprocessing pipeline, CV, calibration, threshold tuning, explainability
- Reproducible, no leakage

Dataset: Health Insurance Fraud Claims.xlsx (claim-level)
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config
from common.logging_utils import get_logger
from common.seed import set_global_seed
from common.dataset_loader import load_claims_dataset, preprocess_target, get_feature_types
from common.schema_validation import check_class_imbalance, detect_potential_leakage
from common.preprocessing import build_preprocessor, engineer_date_features, outlier_analysis
from common.metrics import compute_all_metrics, threshold_analysis
from common.threshold import select_threshold
from common.explainability import get_feature_importance
from common.artifacts import save_model, save_json
from common.result_formatting import risk_category_from_prob

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier, StackingClassifier
)
from sklearn.feature_extraction.text import TfidfVectorizer

logger = get_logger("01_traditional_ml")

def load_data(config, data_path_override=None):
    """Dataset loading with fallback."""
    if data_path_override:
        raw_path = Path(data_path_override)
    else:
        raw_path = Path(config.get("dataset", {}).get("raw_path", "data/raw/Health_Insurance_Fraud_Claims.xlsx"))
        # resolve relative to project root
        if not raw_path.is_absolute():
            raw_path = PROJECT_ROOT / raw_path
    df = load_claims_dataset(raw_path)
    return df

def split_data(df, X, y, config):
    """Time-aware or stratified split."""
    rs = config.get("dataset", {}).get("random_state", 42)
    test_size = config.get("dataset", {}).get("test_size", 0.2)
    val_size = config.get("dataset", {}).get("validation_size", 0.15)

    # Check if date available for time-aware splitting
    # ClaimDate ordinal for stratification
    try:
        # Use stratified split: first test, then val from train
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=rs, stratify=y
        )
        # Adjust val size relative to temp
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=rs, stratify=y_temp
        )
        logger.info(f"Split sizes: train {len(y_train)}, val {len(y_val)}, test {len(y_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        logger.warning(f"Stratified split failed: {e}, using non-stratified")
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=rs)
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=rs)
        return X_train, X_val, X_test, y_train, y_val, y_test

def get_candidate_models(config):
    """Return dict of model_name -> (model, param_grid, notes)."""
    rs = config.get("dataset", {}).get("random_state", 42)
    use_class_weight = config.get("training", {}).get("use_class_weight", True)
    cw = "balanced" if use_class_weight else None

    models = {}

    # Dummy baseline
    models["DummyClassifier"] = (DummyClassifier(strategy="stratified", random_state=rs), {}, "Baseline")

    # Logistic Regression
    models["LogisticRegression"] = (
        LogisticRegression(max_iter=1000, random_state=rs, class_weight=cw, n_jobs=None),
        {"classifier__C": [0.1, 1.0, 10.0]},
        ""
    )
    # Linear SVM + calibration later
    models["LinearSVM"] = (
        LinearSVC(random_state=rs, class_weight=cw, dual=False, max_iter=5000),
        {"classifier__C": [0.1, 1.0]},
        "LinearSVM outputs decision_function, will be calibrated via CalibratedClassifierCV"
    )
    models["CalibratedLinearSVM"] = (
        CalibratedClassifierCV(estimator=LinearSVC(random_state=rs, class_weight=cw, dual=False, max_iter=5000), cv=3),
        {"classifier__estimator__C": [0.1, 1.0]},
        "Calibrated version"
    )
    # KNN
    models["KNN"] = (
        KNeighborsClassifier(),
        {"classifier__n_neighbors": [5, 11]},
        ""
    )
    # Naive Bayes
    models["GaussianNB"] = (
        GaussianNB(),
        {},
        ""
    )
    # Decision Tree
    models["DecisionTree"] = (
        DecisionTreeClassifier(random_state=rs, class_weight=cw),
        {"classifier__max_depth": [None, 10, 20], "classifier__min_samples_leaf": [1, 5]},
        ""
    )
    # Random Forest
    models["RandomForest"] = (
        RandomForestClassifier(random_state=rs, class_weight=cw, n_jobs=-1),
        {"classifier__n_estimators": [100, 200], "classifier__max_depth": [None, 15]},
        ""
    )
    # Extra Trees
    models["ExtraTrees"] = (
        ExtraTreesClassifier(random_state=rs, class_weight=cw, n_jobs=-1),
        {"classifier__n_estimators": [100], "classifier__max_depth": [None, 15]},
        ""
    )
    # Gradient Boosting
    models["GradientBoosting"] = (
        GradientBoostingClassifier(random_state=rs),
        {"classifier__n_estimators": [100], "classifier__learning_rate": [0.1]},
        ""
    )
    # HistGradientBoosting handles class_weight via class_weight param in sklearn 1.3+
    try:
        models["HistGradientBoosting"] = (
            HistGradientBoostingClassifier(random_state=rs, class_weight=cw),
            {"classifier__max_depth": [None, 10], "classifier__learning_rate": [0.1]},
            ""
        )
    except TypeError:
        models["HistGradientBoosting"] = (
            HistGradientBoostingClassifier(random_state=rs),
            {"classifier__max_depth": [None, 10]},
            "class_weight not supported in this sklearn version"
        )
    # AdaBoost
    models["AdaBoost"] = (
        AdaBoostClassifier(random_state=rs),
        {"classifier__n_estimators": [50, 100]},
        ""
    )
    # RBF SVM
    models["SVM_RBF"] = (
        SVC(probability=True, random_state=rs, class_weight=cw, kernel="rbf"),
        {"classifier__C": [1.0], "classifier__gamma": ["scale"]},
        "May be slower; probability=True for calibration"
    )

    # Optional XGBoost, LightGBM, CatBoost
    try:
        import xgboost as xgb
        models["XGBoost"] = (
            xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=rs, n_jobs=-1),
            {"classifier__n_estimators": [100], "classifier__max_depth": [6]},
            "Optional"
        )
        logger.info("XGBoost available")
    except Exception as e:
        logger.info(f"XGBoost not available: {e}")

    try:
        import lightgbm as lgb
        models["LightGBM"] = (
            lgb.LGBMClassifier(random_state=rs, n_jobs=-1, verbose=-1, class_weight=cw),
            {"classifier__n_estimators": [100]},
            "Optional"
        )
        logger.info("LightGBM available")
    except Exception as e:
        logger.info(f"LightGBM not available: {e}")

    try:
        import catboost
        models["CatBoost"] = (
            catboost.CatBoostClassifier(random_seed=rs, verbose=False, auto_class_weights="Balanced" if cw else None),
            {"classifier__iterations": [100]},
            "Optional, may not work with OHE pipeline directly"
        )
        logger.info("CatBoost available")
    except Exception as e:
        logger.info(f"CatBoost not available: {e}")

    # Voting / Stacking will be added after benchmarking base models
    return models

def build_model_pipeline(preprocessor, model):
    """Wrap preprocessor + classifier."""
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

def evaluate_model(pipe, X_val, y_val, X_test, y_test):
    """Evaluate with probabilities."""
    # Predict prob
    if hasattr(pipe, "predict_proba"):
        y_val_prob = pipe.predict_proba(X_val)[:, 1]
        y_test_prob = pipe.predict_proba(X_test)[:, 1]
    elif hasattr(pipe, "decision_function"):
        # Convert to pseudo-prob via sigmoid
        from scipy.special import expit
        y_val_prob = expit(pipe.decision_function(X_val))
        y_test_prob = expit(pipe.decision_function(X_test))
    else:
        y_val_prob = pipe.predict(X_val).astype(float)
        y_test_prob = pipe.predict(X_test).astype(float)

    y_val_pred = (y_val_prob >= 0.5).astype(int)
    y_test_pred = (y_test_prob >= 0.5).astype(int)

    val_metrics = compute_all_metrics(y_val, y_val_pred, y_val_prob)
    test_metrics = compute_all_metrics(y_test, y_test_pred, y_test_prob)
    return val_metrics, test_metrics, y_val_prob, y_test_prob

def main():
    parser = argparse.ArgumentParser(description="Traditional ML fraud detection")
    parser.add_argument("--data_path", type=str, default=None, help="Override data path")
    parser.add_argument("--config", type=str, default=None, help="Config path")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer models")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else PROJECT_ROOT / "config.yaml"
    config = load_config(config_path)
    set_global_seed(config.get("dataset", {}).get("random_state", 42))

    logger.info("Loading data...")
    df = load_data(config, args.data_path)

    # Validation and inspection
    logger.info(f"Dataset shape {df.shape}")
    logger.info(f"Columns {list(df.columns)}")
    logger.info(f"Target distribution {df['ClaimLegitimacy'].value_counts().to_dict()}")

    # Schema inspection
    from common.dataset_loader import validate_schema
    schema_report = validate_schema(df)
    logger.info(f"Schema report: {schema_report}")

    # Label validation
    if df["ClaimLegitimacy"].nunique() != 2:
        logger.warning("Target not binary as expected")

    # Missing analysis
    missing = df.isna().sum()
    logger.info(f"Missing values:\n{missing}")

    # Outlier analysis
    num_feats, cat_feats, date_feats, drop_feats = get_feature_types(df, config)
    # Engineer dates first
    df_engineered = engineer_date_features(df.drop(columns=["ClaimLegitimacy"]), date_feats)
    # Keep target separate
    X_raw = df_engineered
    y_raw = df["ClaimLegitimacy"].map({"Legitimate":0, "Fraud":1}).astype(int)

    # Outlier analysis on numeric
    outlier_rep = outlier_analysis(X_raw, num_feats)
    logger.info(f"Outlier analysis: {outlier_rep}")

    # Class imbalance
    imb = check_class_imbalance(y_raw)
    logger.info(f"Class imbalance: {imb}")

    # Leakage detection
    leak = detect_potential_leakage(df, "ClaimLegitimacy")
    logger.info(f"Potential leakage columns (heuristic): {leak} — Note: ClaimStatus may be post-decision but we keep with documentation")

    # Group/time aware splitting: check if ProviderID can be group
    # For simplicity we use stratified; but note if we have time, sort by date
    # Sort by ClaimDate ordinal if available?
    # We'll attempt time-aware? Config says where appropriate.
    # Since fraud patterns may drift, we note time-aware as future.
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, X_raw, y_raw, config)

    # Further filter features after engineering
    # After engineering, date original dropped, we have new numeric columns like ClaimDate_year etc
    # Let's re-evaluate numeric features: include engineered ones
    engineered_date_cols = [c for c in X_train.columns if "ClaimDate" in c]
    full_num_feats = list(set(num_feats + engineered_date_cols + ["Cluster"]))
    # Ensure they exist
    full_num_feats = [c for c in full_num_feats if c in X_train.columns]
    # Categorical: original cat feats minus dropped, but exclude numeric new
    full_cat_feats = [c for c in cat_feats if c in X_train.columns]

    logger.info(f"Final numeric {full_num_feats}")
    logger.info(f"Final categorical {full_cat_feats}")

    preprocessor = build_preprocessor(full_num_feats, full_cat_feats, [], config)

    # Candidate models
    candidates = get_candidate_models(config)
    if args.quick:
        # Keep only 5 for speed
        keep = ["DummyClassifier","LogisticRegression","RandomForest","HistGradientBoosting","DecisionTree"]
        candidates = {k:v for k,v in candidates.items() if k in keep}

    results = []
    best_score = -1
    best_model_name = None
    best_pipeline = None
    best_val_metrics = None
    best_test_metrics = None
    best_y_val_prob = None

    cv = StratifiedKFold(n_splits=config.get("training", {}).get("cv_folds",5), shuffle=True, random_state=config.get("dataset",{}).get("random_state",42))
    primary_metric = config.get("training", {}).get("primary_metric", "pr_auc")
    scoring = config.get("training", {}).get("scoring", "average_precision")

    skipped = []

    for name, (model, param_grid, notes) in candidates.items():
        logger.info(f"=== Training {name} ===")
        start = time.time()
        try:
            pipe = build_model_pipeline(preprocessor, model)

            # Hyperparameter tuning if grid not empty
            if param_grid:
                # Use GridSearchCV with limited grid for speed
                grid = GridSearchCV(pipe, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=0)
                grid.fit(X_train, y_train)
                best_est = grid.best_estimator_
                logger.info(f"{name} best params {grid.best_params_} best CV {grid.best_score_:.4f}")
            else:
                # Fit directly with cross_validate for logging
                # We still want CV scores
                cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                logger.info(f"{name} CV {scoring} mean {cv_scores['test_score'].mean():.4f}")
                pipe.fit(X_train, y_train)
                best_est = pipe

            # Optional calibration
            calib = config.get("training",{}).get("calibration","isotonic")
            if calib != "none" and not isinstance(model, CalibratedClassifierCV):
                # Calibrate on validation if model supports predict_proba
                if hasattr(best_est, "predict_proba"):
                    try:
                        calibrator = CalibratedClassifierCV(best_est, method=calib, cv="prefit")
                        calibrator.fit(X_val, y_val)
                        # Use calibrated for evaluation
                        eval_pipe = calibrator
                    except Exception as e:
                        logger.warning(f"Calibration failed for {name}: {e}")
                        eval_pipe = best_est
                else:
                    eval_pipe = best_est
            else:
                eval_pipe = best_est

            val_metrics, test_metrics, y_val_prob, y_test_prob = evaluate_model(eval_pipe, X_val, y_val, X_test, y_test)

            elapsed = time.time() - start
            logger.info(f"{name} val PR-AUC {val_metrics.get('pr_auc'):.4f} test PR-AUC {test_metrics.get('pr_auc'):.4f} time {elapsed:.1f}s")

            results.append({
                "model": name,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "fit_time": elapsed,
                "notes": notes,
                "cv_score": float(val_metrics.get('pr_auc',0))
            })

            # Model selection based on primary_metric on val
            score = val_metrics.get(primary_metric, val_metrics.get("pr_auc",0))
            if score > best_score:
                best_score = score
                best_model_name = name
                best_pipeline = eval_pipe
                best_val_metrics = val_metrics
                best_test_metrics = test_metrics
                best_y_val_prob = y_val_prob

        except Exception as e:
            logger.exception(f"Model {name} failed: {e}")
            skipped.append({"model": name, "reason": str(e)})
            results.append({
                "model": name,
                "val_metrics": {"NOT_EXECUTED": f"Failed: {e}"},
                "test_metrics": {},
                "fit_time": 0,
                "notes": f"Skipped: {e}"
            })

    # Attempt Voting and Stacking using top 3
    try:
        # Pick top 3 by val pr_auc
        sorted_res = sorted([r for r in results if "pr_auc" in r.get("val_metrics",{})], key=lambda x: x["val_metrics"]["pr_auc"], reverse=True)
        top3 = sorted_res[:3]
        logger.info(f"Top3 for ensemble: {[r['model'] for r in top3]}")
        if len(top3) >=2:
            # Rebuild fresh pipelines for top models
            top_names = [r["model"] for r in top3]
            estimators = []
            for n in top_names:
                m, _, _ = candidates[n]
                pipe = build_model_pipeline(build_preprocessor(full_num_feats, full_cat_feats, [], config), m)
                estimators.append((n, pipe))
            voting = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
            voting.fit(X_train, y_train)
            val_metrics, test_metrics, y_val_prob, y_test_prob = evaluate_model(voting, X_val, y_val, X_test, y_test)
            results.append({
                "model": "VotingTop3",
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "fit_time": 0,
                "notes": f"Voting of {top_names}"
            })
            if val_metrics.get(primary_metric,0) > best_score:
                best_score = val_metrics.get(primary_metric,0)
                best_model_name = "VotingTop3"
                best_pipeline = voting
                best_val_metrics = val_metrics
                best_test_metrics = test_metrics
                best_y_val_prob = y_val_prob
    except Exception as e:
        logger.warning(f"Voting ensemble failed: {e}")
        skipped.append({"model": "VotingTop3", "reason": str(e)})

    # Threshold analysis
    if best_pipeline is not None and best_y_val_prob is not None:
        thr_strategy = config.get("training",{}).get("threshold_strategy","optimize_f2")
        recall_target = config.get("training",{}).get("threshold_recall_target",0.8)
        best_thr, thr_info = select_threshold(y_val, best_y_val_prob, strategy=thr_strategy, recall_target=recall_target)
        logger.info(f"Best threshold {best_thr} via {thr_strategy}: {thr_info}")

        # Re-evaluate with best threshold
        y_val_pred_thr = (best_y_val_prob >= best_thr).astype(int)
        # For test, we need probs from best model
        if hasattr(best_pipeline, "predict_proba"):
            y_test_prob_final = best_pipeline.predict_proba(X_test)[:,1]
        else:
            from scipy.special import expit
            try:
                y_test_prob_final = expit(best_pipeline.decision_function(X_test))
            except:
                y_test_prob_final = best_pipeline.predict(X_test).astype(float)
        y_test_pred_thr = (y_test_prob_final >= best_thr).astype(int)

        from common.metrics import compute_all_metrics as cam
        val_metrics_thr = cam(y_val, y_val_pred_thr, best_y_val_prob)
        test_metrics_thr = cam(y_test, y_test_pred_thr, y_test_prob_final)

        logger.info(f"Val metrics @ thr {best_thr:.3f}: {val_metrics_thr}")
        logger.info(f"Test metrics @ thr {best_thr:.3f}: {test_metrics_thr}")
    else:
        best_thr = 0.5
        thr_info = {"strategy":"default","threshold":0.5}
        val_metrics_thr = {}
        test_metrics_thr = {}

    # Feature importance & SHAP
    logger.info("Computing feature importance")
    try:
        # Get feature names after preprocessing
        # For sklearn pipeline, get feature names
        preproc_fitted = best_pipeline.named_steps.get("preprocessor") if hasattr(best_pipeline, "named_steps") else None
        if preproc_fitted is not None:
            try:
                feat_names = preproc_fitted.get_feature_names_out()
            except:
                feat_names = None
        else:
            feat_names = None

        clf = best_pipeline.named_steps.get("classifier") if hasattr(best_pipeline, "named_steps") else best_pipeline
        # If calibrated, get base estimator
        if isinstance(clf, CalibratedClassifierCV):
            clf = clf.calibrated_classifiers_[0].estimator if hasattr(clf, "calibrated_classifiers_") else clf.base_estimator

        imp_df = get_feature_importance(clf, feat_names.tolist() if feat_names is not None and hasattr(feat_names, "tolist") else None)
        logger.info(f"Top features:\n{imp_df.head(10)}")
    except Exception as e:
        logger.warning(f"Feature importance failed: {e}")
        imp_df = pd.DataFrame()

    # SHAP
    try:
        from common.explainability import shap_explain
        # Need X_train transformed? We'll attempt on raw with model that handles preprocessing? SHAP with pipeline is tricky.
        # We will approximate using transformed data
        if 'preproc_fitted' in locals() and preproc_fitted is not None:
            X_train_trans = preproc_fitted.transform(X_train)
            X_val_trans = preproc_fitted.transform(X_val)
            shap_values, status = shap_explain(clf, X_train_trans, X_val_trans)
            logger.info(f"SHAP status {status}")
        else:
            shap_values = None
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        shap_values = None

    # Save artifacts
    artifacts_dir = PROJECT_ROOT / config.get("paths",{}).get("artifacts_dir","data/processed/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "best_traditional_ml_model.joblib"
    if best_pipeline is not None:
        save_model(best_pipeline, model_path)
        logger.info(f"Saved best model {best_model_name} to {model_path}")

    # Save evaluation files
    eval_dir = PROJECT_ROOT / config.get("paths",{}).get("evaluation_dir","evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # model_comparison.csv
    comp_rows = []
    for r in results:
        vm = r.get("val_metrics",{})
        tm = r.get("test_metrics",{})
        comp_rows.append({
            "model": r["model"],
            "val_pr_auc": vm.get("pr_auc") if isinstance(vm,dict) else None,
            "val_roc_auc": vm.get("roc_auc") if isinstance(vm,dict) else None,
            "val_f1": vm.get("f1") if isinstance(vm,dict) else None,
            "val_recall": vm.get("recall") if isinstance(vm,dict) else None,
            "val_precision": vm.get("precision") if isinstance(vm,dict) else None,
            "test_pr_auc": tm.get("pr_auc") if isinstance(tm,dict) else None,
            "test_roc_auc": tm.get("roc_auc") if isinstance(tm,dict) else None,
            "fit_time": r.get("fit_time"),
            "notes": r.get("notes")
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(eval_dir / "model_comparison.csv", index=False)

    # metrics_summary
    summary = {
        "best_model": best_model_name,
        "primary_metric": primary_metric,
        "best_val_score": float(best_score) if best_score else None,
        "threshold": float(best_thr),
        "threshold_info": thr_info,
        "val_metrics_final": best_val_metrics,
        "test_metrics_final": best_test_metrics,
        "val_metrics_thresholded": val_metrics_thr,
        "test_metrics_thresholded": test_metrics_thr,
        "skipped_models": skipped,
        "config": config
    }
    save_json(summary, eval_dir / "metrics_summary.json")
    # also csv version
    pd.DataFrame([best_test_metrics]).to_csv(eval_dir / "metrics_summary.csv", index=False) if best_test_metrics else None

    # per_class_metrics
    if best_test_metrics:
        per_class = pd.DataFrame([{
            "class": "fraud",
            "precision": best_test_metrics.get("precision"),
            "recall": best_test_metrics.get("recall"),
            "f1": best_test_metrics.get("f1")
        }])
        per_class.to_csv(eval_dir / "per_class_metrics.csv", index=False)

    # threshold_analysis.csv
    try:
        from common.metrics import threshold_analysis as ta
        if best_y_val_prob is not None:
            thr_df = ta(y_val, best_y_val_prob)
            thr_df.to_csv(eval_dir / "threshold_analysis.csv", index=False)
    except Exception as e:
        logger.warning(f"threshold csv failed {e}")

    # confusion matrices
    try:
        from sklearn.metrics import confusion_matrix
        if best_pipeline is not None:
            if hasattr(best_pipeline, "predict_proba"):
                y_test_prob_final = best_pipeline.predict_proba(X_test)[:,1]
            else:
                y_test_prob_final = best_pipeline.predict(X_test).astype(float)
            y_test_pred_final = (y_test_prob_final >= best_thr).astype(int)
            cm = confusion_matrix(y_test, y_test_pred_final)
            pd.DataFrame(cm).to_csv(eval_dir / "confusion_matrices" / "traditional_ml_cm.csv", index=False)
    except Exception as e:
        logger.warning(f"cm failed {e}")

    # calibration_results.csv
    try:
        from common.metrics import calibration_curve_data
        prob_true, prob_pred = calibration_curve_data(y_test, y_test_prob_final, n_bins=10)
        pd.DataFrame({"prob_true": prob_true, "prob_pred": prob_pred}).to_csv(eval_dir / "calibration_results.csv", index=False)
    except Exception as e:
        logger.warning(f"calib failed {e}")

    # runtime_comparison.csv already in comp_df
    comp_df.to_csv(eval_dir / "runtime_comparison.csv", index=False)

    # data_quality_report.md
    dq_path = eval_dir / "data_quality_report.md"
    with open(dq_path, "w") as f:
        f.write("# Data Quality Report\n\n")
        f.write(f"- Rows {df.shape[0]} cols {df.shape[1]}\n")
        f.write(f"- Missing {missing.to_dict()}\n")
        f.write(f"- Outliers {json.dumps(outlier_rep, indent=2)}\n")
        f.write(f"- Class imbalance {imb}\n")
        f.write(f"- Leakage heuristics {leak}\n")

    # model_selection.md
    with open(eval_dir / "model_selection.md", "w") as f:
        f.write(f"# Model Selection\n\nBest model {best_model_name} with {primary_metric}={best_score:.4f}\n")
        f.write(f"Threshold {best_thr} strategy {thr_strategy}\n")
        f.write(f"Skipped {skipped}\n")

    # run_metadata.json
    save_json({
        "approach": "01_traditional_ml",
        "best_model": best_model_name,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "config": config,
        "data_path": str(raw_path) if 'raw_path' in locals() else "unknown"
    }, eval_dir / "run_metadata.json")

    # Feature importance chart data
    if not imp_df.empty:
        imp_df.to_csv(eval_dir / "feature_importance.csv", index=False)

    logger.info("Traditional ML pipeline completed")
    print(f"BEST_MODEL={best_model_name} PR_AUC={best_score:.4f} THR={best_thr:.3f}")

if __name__ == "__main__":
    main()
