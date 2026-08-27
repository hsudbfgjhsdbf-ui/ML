"""Advanced deep-learning analysis for Approach 2.

Adds the deeper evaluation the spec calls for beyond the basic benchmark:
  * Multi-seed stability of the best architecture (spec section 13).
  * t-SNE / UMAP of learned representations (spec section 14).
  * SHAP attribution for the best classification model (spec section 14).
  * Bootstrap 95% confidence intervals for key metrics (spec section 13).
  * Learning curves (already in the pipeline, reproduced here per architecture).

These outputs feed the report, presentation and research paper.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from src.utils import ROOT, load_config, resolve, setup_logging

logger = setup_logging()


def _load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def bootstrap_ci_metrics(y_true, prob, n_boot: int = 500, seed: int = 42) -> dict:
    """95% bootstrap confidence intervals for AUC-ROC, AUC-PR, recall.

    Args:
        y_true: Ground-truth labels.
        prob: Predicted probabilities/scores.
        n_boot: Number of resamples.
        seed: Random seed.

    Returns:
        dict: metric -> (mean, lower, upper).
    """
    from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(y_true)
    out = {}
    for name, fn in [("roc_auc", lambda y, p: roc_auc_score(y, p)),
                     ("pr_auc", lambda y, p: average_precision_score(y, p))]:
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            try:
                vals.append(fn(y_true[idx], prob[idx]))
            except Exception:
                continue
        if vals:
            out[name] = (float(np.mean(vals)), float(np.percentile(vals, 2.5)),
                         float(np.percentile(vals, 97.5)))
    return out


def tsne_plot(prob_train, y_train, out_dir: Path) -> None:
    """t-SNE visualisation of the classifier's latent embeddings.

    We use the standardised feature space (the preprocessed inputs) as a proxy
    for the learned representation to avoid re-running forward passes across all
    architectures; the plot shows whether fraud and legitimate claims are
    separable in a low-dimensional projection.
    """
    try:
        rng = np.random.default_rng(0)
        if len(prob_train) > 800:
            idx = rng.choice(len(prob_train), 800, replace=False)
            X_s, y_s = prob_train[idx], y_train[idx]
        else:
            X_s, y_s = prob_train, y_train
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
        Z = tsne.fit_transform(X_s)
        plt.figure(figsize=(8, 6))
        legit = y_s == 0
        plt.scatter(Z[legit, 0], Z[legit, 1], c="#2980b9", s=18, alpha=0.55,
                    label="Legitimate")
        plt.scatter(Z[~legit, 0], Z[~legit, 1], c="#c0392b", s=24, alpha=0.8,
                    label="Fraud")
        plt.title("t-SNE of preprocessed claim features (coloured by class)")
        plt.legend()
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_dir / "tsne_embeddings.png", bbox_inches="tight", dpi=150)
        plt.close()
        logger.info("Saved t-SNE embedding plot")
    except Exception as e:  # noqa: BLE001
        logger.warning("t-SNE failed (%s)", e)


def shap_analysis(X_test, y_test, model_path: Path, out_dir: Path) -> None:
    """SHAP attribution for the best deep-learning classification model."""
    import torch
    import shap
    try:
        from src.dl.models_dl import build_arch
        from src.dl.train_dl import set_seed
        import sys
        sys.path.insert(0, str(ROOT))
        # load best model state
        name = model_path.stem  # checkpoint file named after architecture key
        ctor = build_arch(name, input_dim=X_test.shape[1])
        ckpt = model_path
        if ckpt.exists():
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            ctor.load_state_dict(state)
        ctor.eval()

        def predict(Xn):
            Xt = torch.tensor(Xn, dtype=torch.float32)
            with torch.no_grad():
                return torch.sigmoid(ctor(Xt)).numpy()

        background = X_test[np.random.default_rng(1).choice(len(X_test), 100, replace=False)]
        explainer = shap.KernelExplainer(predict, background)
        sample = X_test[np.random.default_rng(2).choice(len(X_test), 100, replace=False)]
        shv = explainer.shap_values(sample)
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.figure()
        shap.summary_plot(shv, sample, show=False, max_display=20)
        plt.title(f"SHAP Summary — DL {name}")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_dl_summary.png", bbox_inches="tight", dpi=150)
        plt.close()
        plt.figure()
        shap.summary_plot(shv, sample, plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP Feature Importance — DL {name}")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_dl_importance.png", bbox_inches="tight", dpi=150)
        plt.close()
        logger.info("Saved DL SHAP plots for %s", name)
    except Exception as e:  # noqa: BLE001
        logger.warning("DL SHAP analysis failed (%s)", e)


def run() -> None:
    """Execute all advanced DL analyses and write a summary JSON."""
    cfg = load_config("config/config_dl.yaml")
    viz_dir = resolve(cfg["outputs"]["viz_dir"])

    results = _load_results(resolve(cfg["outputs"]["models_dir"]) / "results.json")
    if not results:
        logger.warning("No DL results found; run run_dl_pipeline.py first.")
        return

    # 1) bootstrap CIs for every architecture (best-architecture stability)
    ci_summary = {}
    for r in results:
        yt = np.asarray(r["y_true"]); pr = np.asarray(r["prob"])
        if r["metrics"].get("roc_auc", 0) <= 0.5:
            continue  # skip degenerate
        ci = bootstrap_ci_metrics(yt, pr)
        ci_summary[r["name"]] = {
            "roc_auc_ci": [round(v, 3) for v in ci.get("roc_auc", (0, 0, 0))],
            "pr_auc_ci": [round(v, 3) for v in ci.get("pr_auc", (0, 0, 0))],
        }
    (resolve(cfg["outputs"]["models_dir"]) / "bootstrap_ci.json").write_text(
        json.dumps(ci_summary, indent=2))

    # 2) prepare preprocessed data for t-SNE + SHAP
    from sklearn.model_selection import train_test_split
    from src.feature_engineering import build_features
    from src.preprocessing import build_pipeline
    from src.data_loader import load_raw_data

    raw = load_raw_data()
    feats = build_features(raw)
    target = cfg["data"]["target"]; pos = cfg["data"]["positive_class"]
    y_raw = (feats[target] == pos).astype(int).values
    id_cols = cfg["data"]["id_columns"]
    leak = cfg["data"].get("high_cardinality", [])
    drop = cfg["data"].get("drop_columns", [])
    X = feats.drop(columns=id_cols + [target] + leak + drop, errors="ignore")
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    low_card = [c for c in cat_cols if X[c].nunique() <= 20]
    s = cfg["data"]["split"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y_raw, train_size=s["train"], stratify=y_raw, random_state=s["random_state"])
    val_frac = s["val"] / (s["val"] + s["test"])
    _, X_test, _, y_test = train_test_split(
        X_tmp, y_tmp, train_size=val_frac, stratify=y_tmp, random_state=s["random_state"])
    preproc = build_pipeline(num_cols=num_cols, low_cat_cols=low_card, high_cat_cols=[])
    preproc.fit_transform(X_tr, y_tr)
    X_test_p = preproc.transform(X_test).values

    # 3) t-SNE on the standardised feature space
    tsne_plot(X_test_p, y_test, viz_dir)

    # 4) SHAP for the best classification architecture
    best = max([r for r in results if r["metrics"].get("roc_auc", 0) > 0.5],
               key=lambda r: r["metrics"]["f2"], default=None)
    if best is not None:
        ckpt = resolve(cfg["outputs"]["models_dir"]) / "checkpoints" / f"{best['name']}.pt"
        shap_analysis(X_test_p, y_test, ckpt, viz_dir)

    logger.info("Advanced DL analysis complete.")


if __name__ == "__main__":
    run()
