"""SHAP feature-attribution analysis for the best ML model (Approach 1).

Loads the saved best-performing model, preprocessor and the held-out test data,
then produces a SHAP summary plot and a bar plot of mean |SHAP| values. These
are used for the interpretability sections of the report and presentation.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_engineering import build_features
from src.preprocessing import build_pipeline, split_types
from src.utils import ROOT, load_config, resolve, setup_logging

logger = setup_logging()


def prepare_test_data(cfg: dict) -> np.ndarray:
    """Recompute the preprocessed test set (same splits as the ML pipeline)."""
    raw = pd.read_excel(resolve(cfg["data"]["raw_path"]))
    feats = build_features(raw)
    target = cfg["data"]["target"]
    pos = cfg["data"]["positive_class"]
    y_raw = (feats[target] == pos).astype(int).values
    id_cols = cfg["data"]["id_columns"]
    leak_cols = cfg["data"].get("high_cardinality", [])
    drop_cols = cfg["data"].get("drop_columns", [])
    X = feats.drop(columns=id_cols + [target] + leak_cols + drop_cols, errors="ignore")
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    high_card = []
    low_card = [c for c in cat_cols if X[c].nunique() <= 20]
    s = cfg["data"]["split"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y_raw, train_size=s["train"], stratify=y_raw, random_state=s["random_state"])
    val_frac = s["val"] / (s["val"] + s["test"])
    _, X_test, _, _ = train_test_split(
        X_tmp, y_tmp, train_size=val_frac, stratify=y_tmp, random_state=s["random_state"])
    preproc = build_pipeline(num_cols=num_cols, low_cat_cols=low_card,
                             high_cat_cols=high_card)
    preproc.fit_transform(X_tr, y_tr)
    return preproc.transform(X_test).values


def run(out_dir: Path) -> None:
    """Generate SHAP plots for the best ML model."""
    import shap
    cfg = load_config()
    model_dir = resolve(cfg["outputs"]["models_dir"])
    X_test = prepare_test_data(cfg)

    # load results to find the best model
    rp = model_dir / "results.pkl"
    if not rp.exists():
        logger.warning("results.pkl not found; skipping SHAP.")
        return
    with open(rp, "rb") as f:
        obj = pickle.load(f)
    results = sorted(obj["results"], key=lambda r: r.metrics["f2"], reverse=True)
    best = results[0]

    # load the fitted best model
    fname = model_dir / (best.name.replace(" ", "_").lower() + ".pkl")
    if not fname.exists():
        logger.warning("Best model file %s not found.", fname)
        return
    with open(fname, "rb") as f:
        model = pickle.load(f)

    # sample for speed
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=min(300, len(X_test)), replace=False)
    Xs = X_test[idx]

    tree_like = any(k in best.name for k in
                    ("Forest", "XGBoost", "LightGBM", "Gradient", "Tree"))
    try:
        if hasattr(model, "predict_proba") and tree_like:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(Xs[:100])
        else:
            explainer = shap.KernelExplainer(model.predict_proba, Xs[:50])
            shap_values = explainer.shap_values(Xs[:100])
    except Exception as e:  # noqa: BLE001
        logger.warning("SHAP explainer failed (%s); skipping.", e)
        return

    try:
        if isinstance(shap_values, list):
            shv = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        else:
            shv = shap_values

        if isinstance(shap_values, list):
            shv = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        else:
            shv = shap_values

        # summary plot
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.figure()
        shap.summary_plot(shv, Xs[:100], feature_names=None, show=False,
                          max_display=20)
        plt.title(f"SHAP Summary — {best.name}")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary.png", bbox_inches="tight", dpi=150)
        plt.close()

        plt.figure()
        shap.summary_plot(shv, Xs[:100], plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP Feature Importance — {best.name}")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_importance.png", bbox_inches="tight", dpi=150)
        plt.close()
        logger.info("Saved SHAP plots for %s", best.name)
    except Exception as e:  # noqa: BLE001
        logger.warning("SHAP analysis failed (%s); continuing.", e)


if __name__ == "__main__":
    run(ROOT / "visualizations" / "ml")
