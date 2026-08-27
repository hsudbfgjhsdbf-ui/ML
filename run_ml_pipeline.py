#!/usr/bin/env python
"""Approach 1 - Traditional Machine Learning end-to-end pipeline.

Runs the complete pipeline: data load -> feature engineering -> preprocessing
-> class-imbalance handling -> hyperparameter tuning for all 12 algorithms ->
evaluation -> benchmarking -> statistical tests -> fairness analysis -> model
& report serialisation -> visualisation generation.

Usage:
    python run_ml_pipeline.py
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import src.models as M
from src.data_loader import load_raw_data, save_summary
from src.evaluate import business_impact, bootstrap_ci, mcnemar_test
from src.feature_engineering import build_features
from src.preprocessing import (
    build_pipeline, remove_duplicates, report_missing, split_types,
)
from src.train import handle_imbalance, save_model, train_and_evaluate
from src.utils import ensure_dir, load_config, resolve, setup_logging
from src.visualize import generate_all_ml_plots

logger = setup_logging()


def prepare_data(cfg: dict):
    """Load, engineer and preprocess the data into train/val/test splits.

    Returns:
        dict: Contains splits, preprocessor and feature metadata.
    """
    raw = load_raw_data()
    save_summary(raw, resolve(cfg["data"]["processed_dir"]))

    # --- missing values -----------------------------------------------------
    miss = report_missing(raw)
    if not miss.empty:
        logger.info("Missing values detected:\n%s", miss)

    # --- duplicates ---------------------------------------------------------
    raw = remove_duplicates(raw)

    # --- feature engineering -----------------------------------------------
    feats = build_features(raw)

    target = cfg["data"]["target"]
    pos = cfg["data"]["positive_class"]
    y_raw = (feats[target] == pos).astype(int).values

    # Drop identifiers and target.
    # NOTE: the near-unique high-cardinality codes (DiagnosisCode, ProcedureCode,
    # ProviderLocation) are dropped to avoid target-encoding leakage - they have
    # ~4500 unique values across 4500 rows, so mean-encoding them on the target
    # would encode the answer itself. Frequency/other encoding adds no signal.
    id_cols = cfg["data"]["id_columns"]
    leak_cols = cfg["data"].get("high_cardinality", [])
    drop_cols = cfg["data"].get("drop_columns", [])
    X = feats.drop(columns=id_cols + [target] + leak_cols + drop_cols, errors="ignore")

    # identify numeric vs categorical
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    high_card: list = []   # no target-encoding of near-identifiers
    low_card = [c for c in cat_cols if X[c].nunique() <= 20]

    # --- split (stratified) -------------------------------------------------
    s = cfg["data"]["split"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y_raw, train_size=s["train"], stratify=y_raw,
        random_state=s["random_state"],
    )
    val_frac = s["val"] / (s["val"] + s["test"])
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, train_size=val_frac, stratify=y_tmp,
        random_state=s["random_state"],
    )

    # --- preprocess (fit on train only) ------------------------------------
    preproc = build_pipeline(num_cols=num_cols, low_cat_cols=low_card,
                             high_cat_cols=high_card)
    X_tr_p = preproc.fit_transform(X_tr, y_tr)
    X_val_p = preproc.transform(X_val)
    X_test_p = preproc.transform(X_test)

    # --- class imbalance handling (train only) ------------------------------
    strategy = cfg["imbalance"]["strategy"]
    X_tr_b, y_tr_b = handle_imbalance(
        X_tr_p, y_tr, strategy=strategy,
        seed=cfg["imbalance"]["smote_random_state"],
    )

    logger.info("Train shape: %s (after %s), Val: %s, Test: %s",
                X_tr_b.shape, strategy, X_val_p.shape, X_test_p.shape)
    logger.info("Train class balance: %s", dict(pd.Series(y_tr_b).value_counts()))

    # --- feature selection (mutual information) -----------------------------
    mi = mutual_info_classif(X_tr_p, y_tr, random_state=42)
    mi_series = pd.Series(mi, index=X_tr_p.columns).sort_values(ascending=False)

    return {
        "X_train": X_tr_b, "y_train": y_tr_b,
        "X_val": X_val_p, "y_val": y_val,
        "X_test": X_test_p, "y_test": y_test,
        "preproc": preproc, "mi_importance": mi_series,
        "num_cols": num_cols, "low_card": low_card, "high_card": high_card,
        "y_test_raw": y_test,
    }


def run(cfg_path: str = "config/config.yaml") -> None:
    """Run the full Approach-1 pipeline."""
    cfg = load_config(cfg_path)
    out = cfg["outputs"]
    model_dir = resolve(out["models_dir"])
    eval_dir = resolve(out["eval_dir"])
    viz_dir = resolve(out["viz_dir"])
    ensure_dir(model_dir); ensure_dir(eval_dir); ensure_dir(viz_dir)

    data = prepare_data(cfg)
    X_test, y_test = data["X_test"], data["y_test"]

    # persist preprocessor & feature importances
    with open(model_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(data["preproc"], f)
    data["mi_importance"].to_csv(model_dir / "feature_importance_mutual_info.csv")

    # --- train & evaluate all models (with resume-from-cache) ---------------
    import json
    from src.evaluate import EvalResult
    from sklearn.metrics import confusion_matrix as _cm

    results_cache = model_dir / "results_cache.json"
    entries = json.loads(results_cache.read_text()) if results_cache.exists() else []

    results = []
    for name in M.MODEL_FACTORIES:
        cached = next((e for e in entries if e["name"] == name), None)
        if cached is not None:
            logger.info("Loading cached result for %s", name)
            res = EvalResult(
                name=name, metrics=cached["metrics"],
                optimal_threshold=cached["threshold"],
                probabilities=np.asarray(cached["prob"]),
                y_true=np.asarray(cached["y_true"]),
                train_time=cached["train_time"],
                pred_time_ms=cached["pred_time_ms"],
                model_size_kb=cached["size_kb"],
                n_hyperparams=cached["n_hyper"],
            )
            yp = (res.probabilities >= res.optimal_threshold).astype(int)
            res.confusion_matrix = _cm(res.y_true, yp)
            results.append(res)
            continue

        res = train_and_evaluate(
            name, data["X_train"], data["y_train"], X_test, y_test, cfg,
        )
        # record model size
        import os
        fname = model_dir / (name.replace(" ", "_").lower() + ".pkl")
        res.model_size_kb = round(fname.stat().st_size / 1024, 2) if fname.exists() else 0.0
        save_model(name, res.model, model_dir)

        cache_entry = {
            "name": res.name, "metrics": res.metrics,
            "threshold": res.optimal_threshold,
            "train_time": res.train_time, "pred_time_ms": res.pred_time_ms,
            "size_kb": res.model_size_kb, "n_hyper": res.n_hyperparams,
            "prob": res.probabilities.tolist(),
            "y_true": np.asarray(res.y_true).tolist(),
        }
        entries.append(cache_entry)
        results_cache.write_text(json.dumps(entries))
        results.append(res)
        logger.info("Cached %s result", name)

    # --- ranking ------------------------------------------------------------
    ranked = sorted(results, key=lambda r: r.metrics["f2"], reverse=True)

    # --- business impact -----------------------------------------------------
    avg_claim = float(load_raw_data()["ClaimAmount"].mean())
    impact_rows = {}
    for r in ranked:
        cm = r.confusion_matrix
        impact_rows[r.name] = business_impact(cm, avg_claim)

    # --- statistical significance (McNemar vs best) --------------------------
    best = ranked[0]
    stat_rows = []
    for r in ranked[1:]:
        chi2, p = mcnemar_test(y_test, (r.probabilities >= r.optimal_threshold).astype(int),
                               (best.probabilities >= best.optimal_threshold).astype(int))
        stat_rows.append({
            "model": r.name, "vs_best": best.name, "chi2": chi2, "p_value": p,
            "significant_05": p < 0.05,
        })

    # --- write benchmark markdown -------------------------------------------
    write_benchmark_md(ranked, impact_rows, stat_rows, data, cfg, eval_dir)

    # --- serialise full results ---------------------------------------------
    with open(model_dir / "results.pkl", "wb") as f:
        pickle.dump({"results": results, "cfg": cfg}, f)

    # --- visualisations ------------------------------------------------------
    generate_all_ml_plots(data, ranked, impact_rows, viz_dir)

    logger.info("Approach 1 pipeline complete. Best model: %s (F2=%.3f)",
                best.name, best.metrics["f2"])


def write_benchmark_md(ranked, impact_rows, stat_rows, data, cfg, eval_dir) -> None:
    """Write the main evaluation markdown file with the benchmarking table."""
    cols = ["accuracy", "precision", "recall", "f1", "f2", "roc_auc", "pr_auc",
            "mcc", "train_time", "pred_time_ms", "model_size_kb", "n_hyperparams"]
    lines = ["# Approach 1 - Traditional ML Benchmarking Report\n",
             "## Overview", "",
             f"- Dataset rows: {data['X_test'].shape[0] + data['X_train'].shape[0] + data['X_val'].shape[0]}",
             f"- Imbalance strategy: {cfg['imbalance']['strategy']}",
             f"- CV scoring metric: F2, folds: {cfg['training']['cv_folds']}",
             "- Best model ranked by F2 score (secondary: ROC-AUC).\n",
             "## Benchmarking Table", "",
             "| Algorithm | Acc | Prec | Recall | F1 | **F2** | AUC-ROC | AUC-PR | MCC | Train(s) | Pred(ms) | Size(KB) | #HP |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ranked:
        m = r.metrics
        lines.append(
            f"| {r.name} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} | **{m['f2']:.3f}** | {m.get('roc_auc', 0):.3f} | "
            f"{m.get('pr_auc', 0):.3f} | {m['mcc']:.3f} | {r.train_time:.2f} | "
            f"{r.pred_time_ms:.3f} | {r.model_size_kb:.1f} | {r.n_hyperparams} |"
        )

    lines += ["\n## Ranking (by F2)", ""]
    for i, r in enumerate(ranked, 1):
        lines.append(f"{i}. **{r.name}** — F2={r.metrics['f2']:.4f}, ROC-AUC={r.metrics.get('roc_auc',0):.4f}")

    lines += ["\n## Business Impact (INR)", "",
              "| Model | FN (fraud approved) | FP (valid rejected) | Estimated fraud loss (INR) |",
              "|---|---|---|---|"]
    for r in ranked:
        imp = impact_rows[r.name]
        lines.append(f"| {r.name} | {imp['false_negative']} | {imp['false_positive']} | {imp['fraud_loss_inr']:,.0f} |")

    lines += ["\n## Statistical Significance (McNemar vs best)", "",
              "| Model | chi2 | p-value | p<0.05? |", "|---|---|---|---|"]
    for s in stat_rows:
        lines.append(f"| {s['model']} | {s['chi2']:.3f} | {s['p_value']:.4f} | {s['significant_05']} |")

    lines += ["\n## Optimal Thresholds", "", "| Model | Optimal threshold |", "|---|---|"]
    for r in ranked:
        lines.append(f"| {r.name} | {r.optimal_threshold:.3f} |")

    path = eval_dir / "ml_evaluation.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote benchmark report to %s", path)


if __name__ == "__main__":
    run()
