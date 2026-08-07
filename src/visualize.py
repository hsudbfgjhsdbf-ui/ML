"""Publication-ready, consistently coloured visualizations for Approach 1.

Fraud is always red and legitimate claims use blue/teal. Each plotting function writes
local PNG assets for direct Markdown, presentation and PDF reuse; it never opens an
interactive UI, allowing headless reproducible runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

FRAUD = "#c63d4f"
LEGITIMATE = "#1b7f8c"
NAVY = "#17324d"
GOLD = "#d69e2e"
PALETTE = {0: LEGITIMATE, 1: FRAUD, "Legitimate": LEGITIMATE, "Fraudulent": FRAUD}


def _prepare(root: str | Path, category: str) -> Path:
    """Create and return a visualization-category directory."""
    destination = Path(root) / category
    destination.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=.95)
    return destination


def _save(fig: plt.Figure, destination: Path, name: str) -> Path:
    """Write a high-resolution figure and release its memory."""
    path = destination / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def generate_eda_plots(frame: pd.DataFrame, root: str | Path) -> list[Path]:
    """Generate required distributions, relationships, correlations and temporal EDA charts.

    Args:
        frame: Cleaned raw synthetic claims.
        root: Root visualization directory.

    Returns:
        Paths to created image files.
    """
    dest, assets = _prepare(root, "eda"), []
    display = frame.copy()
    display["Fraud status"] = display["is_fraud"].map({0: "Legitimate", 1: "Fraudulent"})
    # Right-skewed amount distribution uses log x scaling to retain the tail.
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    sns.histplot(data=display, x="claim_amount_inr", hue="Fraud status", bins=55, stat="density", common_norm=False, palette={"Legitimate": LEGITIMATE, "Fraudulent": FRAUD}, alpha=.45, ax=ax)
    ax.set_xscale("log"); ax.set_title("Claim Amount Distribution by Synthetic Fraud Label"); ax.set_xlabel("Claim amount (INR, log scale)"); ax.set_ylabel("Density")
    assets.append(_save(fig, dest, "claim_amount_distribution.png"))
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    sns.boxplot(data=display, x="Fraud status", y="claim_amount_inr", palette={"Legitimate": LEGITIMATE, "Fraudulent": FRAUD}, showfliers=False, ax=ax)
    ax.set_title("Claim Amount Spread by Class (Outliers Hidden Only for Readability)"); ax.set_xlabel(""); ax.set_ylabel("Claim amount (INR)")
    assets.append(_save(fig, dest, "claim_amount_boxplot.png"))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    display["Fraud status"].value_counts().plot.pie(autopct="%.1f%%", colors=[LEGITIMATE, FRAUD], startangle=90, ax=axes[0], ylabel="")
    axes[0].set_title("Class Distribution")
    sns.countplot(data=display, y="claim_type", hue="Fraud status", palette={"Legitimate": LEGITIMATE, "Fraudulent": FRAUD}, ax=axes[1])
    axes[1].set_title("Claim-Type Mix by Class"); axes[1].set_xlabel("Claims"); axes[1].set_ylabel("")
    assets.append(_save(fig, dest, "class_and_claim_type_mix.png"))
    numeric = display.select_dtypes(include=[np.number]).drop(columns=["is_fraud"], errors="ignore")
    correlations = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(correlations, cmap="vlag", center=0, vmin=-1, vmax=1, square=False, ax=ax, cbar_kws={"label": "Pearson correlation"})
    ax.set_title("Numeric Feature Correlation Heatmap")
    assets.append(_save(fig, dest, "numeric_correlation_heatmap.png"))
    # Minimal correlation network avoids an additional runtime dependency.
    strongest = [(a, b, correlations.loc[a, b]) for i, a in enumerate(correlations.columns) for b in correlations.columns[i+1:] if abs(correlations.loc[a, b]) >= .35]
    nodes = sorted({x for a,b,_ in strongest for x in (a,b)})[:18]
    if nodes:
        theta = np.linspace(0, 2*np.pi, len(nodes), endpoint=False); coords = {name:(np.cos(t), np.sin(t)) for name,t in zip(nodes,theta)}
        fig, ax = plt.subplots(figsize=(10, 8))
        for a,b,r in strongest:
            if a in coords and b in coords:
                xa,ya=coords[a]; xb,yb=coords[b]; ax.plot([xa,xb],[ya,yb], color=FRAUD if r>0 else LEGITIMATE, alpha=min(.75,abs(r)), lw=1+3*abs(r))
        for name,(x,y) in coords.items():
            ax.scatter(x,y,s=700,color=NAVY,edgecolor="white",zorder=3); ax.text(x,y,name.replace("_","\n"),ha="center",va="center",color="white",fontsize=7,zorder=4)
        ax.axis("off"); ax.set_title("Correlation Network (|r| ≥ 0.35; red positive, teal negative)")
        assets.append(_save(fig, dest, "correlation_network.png"))
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    sample = display.sample(min(3500, len(display)), random_state=7)
    sns.scatterplot(data=sample, x="hospitalization_days", y="claim_amount_inr", hue="Fraud status", palette={"Legitimate": LEGITIMATE, "Fraudulent": FRAUD}, alpha=.62, s=27, ax=ax)
    ax.set_yscale("log"); ax.set_title("Claim Amount vs Hospitalization Duration"); ax.set_xlabel("Hospitalization days"); ax.set_ylabel("Claim amount (INR, log scale)")
    assets.append(_save(fig, dest, "amount_vs_duration.png"))
    rate = display.pivot_table(index="state", columns="treatment_type", values="is_fraud", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 6.4)); sns.heatmap(rate, cmap="Reds", annot=True, fmt=".1%", ax=ax, cbar_kws={"label":"Fraud rate"})
    ax.set_title("Synthetic Fraud Rate by Claimant State and Treatment Type"); ax.set_xlabel("Treatment type"); ax.set_ylabel("Claimant state")
    assets.append(_save(fig, dest, "fraud_rate_state_treatment_heatmap.png"))
    timeline = display.assign(month=pd.to_datetime(display["claim_date"]).dt.to_period("M").astype(str)).groupby(["month", "Fraud status"]).size().reset_index(name="claims")
    fig, ax = plt.subplots(figsize=(10.5, 5.2)); sns.lineplot(data=timeline, x="month", y="claims", hue="Fraud status", marker="o", palette={"Legitimate": LEGITIMATE, "Fraudulent": FRAUD}, ax=ax)
    ax.set_title("Monthly Claim Pattern by Synthetic Class"); ax.set_xlabel("Claim month"); ax.set_ylabel("Claims"); ax.tick_params(axis="x", rotation=45)
    assets.append(_save(fig, dest, "monthly_claim_patterns.png"))
    return assets


def generate_model_plots(benchmark: pd.DataFrame, selected_features: list[str], models: dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, root: str | Path) -> list[Path]:
    """Plot all-model ROC/PR/comparison figures and selected interpretability artifacts.

    Args:
        benchmark: Full evaluation frame with private probability arrays.
        selected_features: Matrix-column names after selection.
        models: Mapping from model name to TrainedModel.
        X_test: Selected test matrix.
        y_test: Held-out labels.
        root: Root visualization directory.

    Returns:
        Generated asset paths.
    """
    dest, assets = _prepare(root, "model_comparison"), []
    ordered = benchmark.sort_values("f2", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    for _, row in ordered.iterrows():
        RocCurveDisplay.from_predictions(y_test, row["_probability"], name=f"{row['algorithm']} ({row['auc_roc']:.3f})", ax=ax)
    ax.plot([0,1],[0,1],"--",color="gray",lw=1); ax.set_title("Held-out ROC Curves — All Algorithms")
    ax.legend(fontsize=7, loc="lower right")
    assets.append(_save(fig, dest, "roc_curves_all_models.png"))
    fig, ax = plt.subplots(figsize=(10, 7))
    for _, row in ordered.iterrows():
        PrecisionRecallDisplay.from_predictions(y_test, row["_probability"], name=f"{row['algorithm']} ({row['auc_pr']:.3f})", ax=ax)
    ax.set_title("Held-out Precision–Recall Curves — All Algorithms"); ax.legend(fontsize=7, loc="lower left")
    assets.append(_save(fig, dest, "precision_recall_curves_all_models.png"))
    measures = ["accuracy", "precision", "recall", "f1", "f2", "auc_roc", "auc_pr", "mcc"]
    long = ordered.melt(id_vars="algorithm", value_vars=measures, var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(15, 7)); sns.barplot(data=long, x="algorithm", y="score", hue="metric", palette="tab10", ax=ax)
    ax.set_ylim(0,1.05); ax.set_title("Held-out Metric Comparison"); ax.set_xlabel(""); ax.set_ylabel("Score"); ax.tick_params(axis="x", rotation=58); ax.legend(ncol=4, fontsize=8)
    assets.append(_save(fig, dest, "grouped_metric_comparison.png"))
    top = ordered.head(min(5, len(ordered)))
    radar_metrics = ["accuracy", "precision", "recall", "f1", "f2", "auc_roc", "auc_pr", "mcc"]
    angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist(); angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw={"polar":True})
    for _, row in top.iterrows():
        values = [row[m] for m in radar_metrics]; values += values[:1]
        ax.plot(angles, values, label=row["algorithm"], lw=2); ax.fill(angles, values, alpha=.06)
    ax.set_xticks(angles[:-1], [m.upper().replace("_", "-") for m in radar_metrics]); ax.set_ylim(0,1); ax.set_title("Top Five Models: Multi-metric Radar", pad=28); ax.legend(loc="upper right", bbox_to_anchor=(1.34,1.13), fontsize=8)
    assets.append(_save(fig, dest, "top_five_radar.png"))
    fig, ax = plt.subplots(figsize=(9.5, 6)); sns.scatterplot(data=ordered, x="training_time_seconds", y="accuracy", size="model_size_kb", hue="f2", palette="Reds", sizes=(50,500), ax=ax)
    for _, row in ordered.head(7).iterrows(): ax.annotate(row["algorithm"], (row["training_time_seconds"], row["accuracy"]), fontsize=7, xytext=(4,3), textcoords="offset points")
    ax.set_title("Computational Efficiency: Training Time vs Accuracy"); ax.set_xlabel("Training+tuning time (seconds)"); ax.set_ylabel("Held-out accuracy")
    assets.append(_save(fig, dest, "training_time_vs_accuracy.png"))
    # Confusions are read from benchmark (rows include counts) to avoid recomputing.
    cols, rows = 4, int(np.ceil(len(ordered)/4)); fig, axes = plt.subplots(rows, cols, figsize=(13, 3.0*rows)); axes=np.asarray(axes).ravel()
    for ax, (_, row) in zip(axes, ordered.iterrows()):
        matrix=np.array([[row["true_negative"],row["false_positive"]],[row["false_negative"],row["true_positive"]]])
        sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=ax, xticklabels=["Approve","Flag"], yticklabels=["Legitimate","Fraud"])
        ax.set_title(row["algorithm"],fontsize=9); ax.set_xlabel("Prediction",fontsize=8); ax.set_ylabel("Actual",fontsize=8)
    for ax in axes[len(ordered):]: ax.axis("off")
    fig.suptitle("Held-out Confusion Matrices at Validation-Selected Thresholds", y=1.01)
    assets.append(_save(fig, dest, "confusion_matrices_all_models.png"))
    # Tree/linear feature importance uses top F2 model with available native attribution.
    interp_dest = _prepare(root, "interpretability")
    for name, trained in models.items():
        estimator = trained.estimator
        importance = getattr(estimator, "feature_importances_", None)
        if importance is None and hasattr(estimator, "coef_"):
            importance = np.abs(np.ravel(estimator.coef_))
        if importance is not None and len(importance) == len(selected_features):
            importance_frame = pd.DataFrame({"feature":selected_features,"importance":importance}).sort_values("importance",ascending=False).head(20)
            fig, ax=plt.subplots(figsize=(9,6)); sns.barplot(data=importance_frame, y="feature", x="importance", color=FRAUD, ax=ax)
            ax.set_title(f"Top 20 Feature Importances — {name}"); ax.set_xlabel("Native importance / |coefficient|"); ax.set_ylabel("")
            assets.append(_save(fig, interp_dest, f"feature_importance_{name.lower().replace(' ','_').replace('(','').replace(')','')}.png"))
    leader = ordered.iloc[0]
    # Reliable diagram from actual selected features, independent of optional SHAP version behavior.
    fig, ax = plt.subplots(figsize=(10.5,4.3)); ax.axis("off")
    boxes=[("Raw Indian\nclaim",.05,LEGITIMATE),("Train-only\npreprocessing",.27,NAVY),("Selected\nfeatures",.49,GOLD),("F2-tuned\nmodel",.70,FRAUD),("Human review\nrecommendation",.88,NAVY)]
    for label,x,color in boxes:
        ax.text(x,.55,label,ha="center",va="center",fontsize=11,color="white",bbox={"boxstyle":"round,pad=.7","fc":color,"ec":"white"})
    for _,x,_ in boxes[:-1]: ax.annotate("",xy=(x+.10,.55),xytext=(x+.06,.55),arrowprops={"arrowstyle":"->","lw":2,"color":"#555"})
    ax.set_title(f"Approach 1 Technical Pipeline (Best held-out F2: {leader['algorithm']})",fontsize=14,pad=18)
    assets.append(_save(fig, _prepare(root,"technical"), "traditional_ml_pipeline_diagram.png"))
    return assets


def generate_fairness_and_calibration_plots(fairness: pd.DataFrame, calibration: pd.DataFrame, root: str | Path) -> list[Path]:
    """Plot audit-friendly fairness and probability-calibration summaries.

    Args:
        fairness: Group metrics for selected best model.
        calibration: Reliability points for selected best model.
        root: Root visualization directory.

    Returns:
        Asset file paths.
    """
    dest, assets = _prepare(root,"fairness"), []
    plot = fairness[fairness["dimension"].isin(["gender","age_group","income_bracket"])].copy()
    if not plot.empty:
        fig, ax=plt.subplots(figsize=(12,6)); sns.barplot(data=plot, x="group", y="fnr", hue="dimension", palette="Set2", ax=ax)
        ax.set_ylim(0,1); ax.set_title("False Negative Rate by Protected/Audit Group — Best Model"); ax.set_xlabel("Group"); ax.set_ylabel("False negative rate"); ax.tick_params(axis="x",rotation=35)
        assets.append(_save(fig,dest,"fairness_fnr_groups.png"))
    fig, ax=plt.subplots(figsize=(6.6,5.4)); ax.plot([0,1],[0,1],"--",color="gray",label="Perfect calibration"); ax.plot(calibration["mean_predicted_probability"],calibration["observed_fraud_rate"],marker="o",color=FRAUD,label="Best model")
    ax.set_title("Reliability Diagram — Best Model"); ax.set_xlabel("Mean predicted fraud probability"); ax.set_ylabel("Observed fraud rate"); ax.legend()
    assets.append(_save(fig,dest,"calibration_reliability_diagram.png"))
    return assets
