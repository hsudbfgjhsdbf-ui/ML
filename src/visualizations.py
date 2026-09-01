"""
High-Resolution Publication Visualizer and Chart Generator.
Creates comprehensive distribution plots, correlation heatmaps, ROC/PR curves,
radar charts, SHAP summary visuals, and Indian healthcare fraud heatmaps.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.config import config, VISUALIZATIONS_DIR
from src.utils import logger

# Professional Styling Configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300
})

FRAUD_COLOR = "#D9381E"   # Crisp Crimson for Fraud
LEGIT_COLOR = "#1F77B4"   # Classic Sapphire for Legitimate
ACCENT_GREEN = "#2CA02C"  # Emerald Accent

def generate_all_visualizations(
    df_raw: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    ml_results: Dict[str, Any],
    dl_results: Dict[str, Any],
    y_test: np.ndarray,
    ml_probs: Dict[str, np.ndarray],
    dl_probs: Dict[str, np.ndarray],
    feature_names: List[str],
    save_dir: Path = VISUALIZATIONS_DIR
) -> List[Path]:
    """
    Generates and saves the entire visual asset suite for presentations and research papers.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    logger.info(f"Generating full visualization suite into {save_dir}")
    
    # -------------------------------------------------------------
    # 1. Dataset Class Distribution & Claim Types (Pie / Bar Chart)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Pie chart of fraud vs legitimate
    fraud_counts = df_synthetic["Is_Fraud"].value_counts()
    axes[0].pie(
        fraud_counts,
        labels=["Legitimate Claims (Approve)", "Fraudulent Claims (Reject)"],
        autopct="%1.1f%%",
        startangle=140,
        colors=[LEGIT_COLOR, FRAUD_COLOR],
        explode=(0, 0.1),
        shadow=True
    )
    axes[0].set_title("Ground-Truth Class Distribution\n(Indian Insurance Dataset)")
    
    # Bar chart of Claim Types
    sns.countplot(
        data=df_synthetic,
        x="Claim_Type",
        hue="Is_Fraud",
        palette=[LEGIT_COLOR, FRAUD_COLOR],
        ax=axes[1]
    )
    axes[1].set_title("Claim Volume & Fraud Breakdown by Claim Type")
    axes[1].set_xlabel("Claim Type")
    axes[1].set_ylabel("Number of Claims")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(["Legitimate", "Fraud"])
    
    plt.tight_layout()
    p1 = save_dir / "01_class_distribution_and_types.png"
    plt.savefig(p1, bbox_inches="tight")
    plt.close()
    generated_files.append(p1)
    
    # -------------------------------------------------------------
    # 2. Claim Amount Distribution (Histogram & KDE by Class)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        data=df_synthetic,
        x="Claim_Amount_INR",
        hue="Is_Fraud",
        palette=[LEGIT_COLOR, FRAUD_COLOR],
        bins=40,
        kde=True,
        ax=ax,
        log_scale=(True, False)
    )
    ax.set_title("Distribution of Claim Amounts (INR ₹ Log-Scale) by Legitimacy")
    ax.set_xlabel("Claimed Amount (INR ₹ Log Scale)")
    ax.set_ylabel("Claim Count")
    ax.legend(["Fraudulent", "Legitimate"])
    
    p2 = save_dir / "02_claim_amount_distribution.png"
    plt.savefig(p2, bbox_inches="tight")
    plt.close()
    generated_files.append(p2)
    
    # -------------------------------------------------------------
    # 3. Correlation Heatmap of Numeric Features
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 9))
    num_cols = [c for c in df_synthetic.columns if pd.api.types.is_numeric_dtype(df_synthetic[c]) and "ID" not in c][:12]
    corr_mat = df_synthetic[num_cols].corr()
    
    sns.heatmap(
        corr_mat,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Pairwise Pearson Correlation Matrix of Domain Features")
    plt.xticks(rotation=45, ha="right")
    
    p3 = save_dir / "03_correlation_heatmap.png"
    plt.savefig(p3, bbox_inches="tight")
    plt.close()
    generated_files.append(p3)
    
    # -------------------------------------------------------------
    # 4. Top 20 Feature Importance Bar Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    if "Random_Forest" in ml_results:
        rf_model = ml_results["Random_Forest"]["model"]
        if hasattr(rf_model, "feature_importances_"):
            imps = rf_model.feature_importances_
            names = feature_names[:len(imps)]
            feat_df = pd.DataFrame({"Feature": names, "Importance": imps}).sort_values("Importance", ascending=True).tail(18)
            
            ax.barh(feat_df["Feature"], feat_df["Importance"], color="#2B5B84")
            ax.set_title("Top Predictive Features Ranked by Random Forest Gini Importance")
            ax.set_xlabel("Relative Feature Importance")
            
    p4 = save_dir / "04_top_feature_importance.png"
    plt.savefig(p4, bbox_inches="tight")
    plt.close()
    generated_files.append(p4)
    
    # -------------------------------------------------------------
    # 5. Overlaid ROC Curves for All ML & DL Models
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    from sklearn.metrics import roc_curve, auc
    
    # Plot top ML
    for name in ["Random_Forest", "XGBoost", "LightGBM", "Logistic_Regression", "Voting_Ensemble_Soft"]:
        if name in ml_probs:
            fpr, tpr, _ = roc_curve(y_test, ml_probs[name])
            roc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"ML: {name} (AUC = {roc_val:.3f})")
            
    # Plot top DL
    for name in ["Tabular_FT_Transformer", "TabNet_Attention", "Tabular_ResNet", "Deep_and_Cross_DCN"]:
        if name in dl_probs:
            fpr, tpr, _ = roc_curve(y_test, dl_probs[name])
            roc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, linestyle="--", label=f"DL: {name} (AUC = {roc_val:.3f})")
            
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle=":")
    ax.set_title("Overlaid Receiver Operating Characteristic (ROC) Curves")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, alpha=0.3)
    
    p5 = save_dir / "05_overlaid_roc_curves.png"
    plt.savefig(p5, bbox_inches="tight")
    plt.close()
    generated_files.append(p5)
    
    # -------------------------------------------------------------
    # 6. Overlaid Precision-Recall Curves
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    for name in ["XGBoost", "LightGBM", "Random_Forest", "Tabular_FT_Transformer", "TabNet_Attention"]:
        probs = ml_probs.get(name, dl_probs.get(name))
        if probs is not None:
            prec, rec, _ = precision_recall_curve(y_test, probs)
            ap = average_precision_score(y_test, probs)
            ax.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {ap:.3f})")
            
    ax.set_title("Overlaid Precision-Recall (PR) Curves for Imbalanced Fraud Detection")
    ax.set_xlabel("Recall (Fraud Coverage)")
    ax.set_ylabel("Precision (Fraud Purity)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, alpha=0.3)
    
    p6 = save_dir / "06_overlaid_pr_curves.png"
    plt.savefig(p6, bbox_inches="tight")
    plt.close()
    generated_files.append(p6)
    
    # -------------------------------------------------------------
    # 7. Model Benchmarking Comparison (Grouped Bar Chart: F2, Recall, Precision, AUC)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6))
    
    benchmark_models = ["XGBoost", "LightGBM", "Random_Forest", "Tabular_FT_Transformer", "TabNet_Attention", "Tabular_ResNet", "Logistic_Regression"]
    recalls, precisions, f2s, aucs = [], [], [], []
    valid_names = []
    
    for m in benchmark_models:
        meta = ml_results.get(m, dl_results.get(m))
        if meta:
            mets = meta["metrics"]
            valid_names.append(m.replace("_", " "))
            recalls.append(mets["recall"])
            precisions.append(mets["precision"])
            f2s.append(mets["f2_score"])
            aucs.append(mets["roc_auc"])
            
    x = np.arange(len(valid_names))
    width = 0.20
    
    ax.bar(x - 1.5*width, f2s, width, label="F2-Score (Primary)", color="#D9381E")
    ax.bar(x - 0.5*width, recalls, width, label="Recall", color="#E67E22")
    ax.bar(x + 0.5*width, precisions, width, label="Precision", color="#3498DB")
    ax.bar(x + 1.5*width, aucs, width, label="AUC-ROC", color="#27AE60")
    
    ax.set_title("Multi-Metric Benchmark Comparison Across Top Algorithms")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_names, rotation=20, ha="right")
    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    
    p7 = save_dir / "07_benchmark_metrics_barchart.png"
    plt.savefig(p7, bbox_inches="tight")
    plt.close()
    generated_files.append(p7)
    
    # -------------------------------------------------------------
    # 8. Training Time vs F2-Score Computational Trade-off
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_names = list(ml_results.keys()) + list(dl_results.keys())
    times, scores, labels = [], [], []
    
    for name in all_names:
        meta = ml_results.get(name, dl_results.get(name))
        if meta:
            t = meta["metrics"].get("training_time_sec", 0.5)
            s = meta["metrics"].get("f2_score", 0.5)
            times.append(t)
            scores.append(s)
            labels.append(name.replace("_", " "))
            
    scatter = ax.scatter(times, scores, c=scores, cmap="plasma", s=140, edgecolors="black", alpha=0.85)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (times[i] + 0.05, scores[i]), fontsize=8.5)
        
    ax.set_title("Computational Cost vs Fraud Detection Efficacy (F2-Score)")
    ax.set_xlabel("Training Duration (Seconds)")
    ax.set_ylabel("Test F2-Score (Target Metric)")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, label="F2 Score")
    
    p8 = save_dir / "08_training_time_vs_f2_tradeoff.png"
    plt.savefig(p8, bbox_inches="tight")
    plt.close()
    generated_files.append(p8)
    
    # -------------------------------------------------------------
    # 9. Hospital Tier and Geographic Heatmap of Fraud Rates
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    if "Patient_State" in df_synthetic.columns and "Hospital_Tier" in df_synthetic.columns:
        pvt = df_synthetic.pivot_table(
            index="Patient_State",
            columns="Hospital_Tier",
            values="Is_Fraud",
            aggfunc="mean"
        ) * 100.0
        
        sns.heatmap(pvt, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Fraud Rate (%)"}, ax=ax)
        ax.set_title("Geographic and Hospital Tier Fraud Risk Heatmap (% Fraud Incidence)")
        ax.set_xlabel("Admitting Healthcare Facility Tier")
        ax.set_ylabel("Claimant Residence State")
        
    p9 = save_dir / "09_state_tier_fraud_heatmap.png"
    plt.savefig(p9, bbox_inches="tight")
    plt.close()
    generated_files.append(p9)
    
    # -------------------------------------------------------------
    # 10. Multi-Agent Verification Architecture Diagram (Visual Graph)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    
    # Draw Agent AI architecture diagram using matplotlib boxes and arrows
    boxes = [
        ("Claim Submission\n& Multi-File Upload\n(Bills, Rx, Discharge)", (0.08, 0.5), "#E8F4F8", "#2B5B84"),
        ("Coordinator Agent\n(LangGraph State Machine\n& Routing)", (0.32, 0.5), "#EBF5FB", "#2980B9"),
        ("Document Agent\n(OCR / VLM\nStructured JSON)", (0.58, 0.78), "#EAF2F8", "#34495E"),
        ("Policy Agent\n(RAG Clause\nVerification)", (0.58, 0.50), "#EAF2F8", "#34495E"),
        ("Anomaly & History\nAgent (Tariff\n& Ring Check)", (0.58, 0.22), "#EAF2F8", "#34495E"),
        ("Reasoning & Decision Agent\n(Bilingual Explanation\n& Evidence Synthesis)", (0.85, 0.5), "#FDEDEC", "#C0392B")
    ]
    
    for text, (x, y), face, edge in boxes:
        ax.text(
            x, y, text,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=face, edgecolor=edge, linewidth=2),
            fontsize=9.5, fontweight="bold"
        )
        
    # Draw Arrows
    arrows = [
        ((0.17, 0.5), (0.23, 0.5)),
        ((0.41, 0.55), (0.49, 0.75)),
        ((0.41, 0.50), (0.49, 0.50)),
        ((0.41, 0.45), (0.49, 0.25)),
        ((0.67, 0.75), (0.75, 0.55)),
        ((0.67, 0.50), (0.75, 0.50)),
        ((0.67, 0.25), (0.75, 0.45)),
    ]
    for start, end in arrows:
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(arrowstyle="->", lw=2, color="#2C3E50")
        )
        
    ax.set_title("Approach 3: Multi-Agent Cognitive Claim Verification Pipeline Architecture", fontsize=13, fontweight="bold", pad=20)
    p10 = save_dir / "10_multi_agent_workflow_architecture.png"
    plt.savefig(p10, bbox_inches="tight")
    plt.close()
    generated_files.append(p10)
    
    logger.info(f"Successfully rendered and saved {len(generated_files)} high-resolution visualization figures.")
    return generated_files
