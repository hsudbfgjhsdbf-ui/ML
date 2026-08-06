"""Generate visualizations from actual dataset and evaluation results."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config

config = load_config(PROJECT_ROOT/"config.yaml")
images_dir = PROJECT_ROOT / config.get("paths",{}).get("images_dir","images")
images_dir.mkdir(parents=True, exist_ok=True)
# also visualizations symlink?
vis_dir = PROJECT_ROOT / "visualizations"
vis_dir.mkdir(parents=True, exist_ok=True)

# Load data
raw_path = PROJECT_ROOT / config.get("dataset",{}).get("raw_path","data/raw/Health_Insurance_Fraud_Claims.xlsx")
if not raw_path.exists():
    raw_path = PROJECT_ROOT.parent / "Health Insurance Fraud Claims.xlsx"
    if not raw_path.exists():
        raw_path = Path("/home/user/ML/Health Insurance Fraud Claims.xlsx")

print(f"Loading {raw_path}")
if raw_path.suffix in [".xlsx",".xls"]:
    df = pd.read_excel(raw_path)
else:
    df = pd.read_csv(raw_path)

# 1 Class distribution
plt.figure(figsize=(6,4))
vc = df["ClaimLegitimacy"].value_counts()
sns.barplot(x=vc.index, y=vc.values)
plt.title("Class Distribution (ClaimLegitimacy)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(images_dir/"class_distribution.png", dpi=150)
plt.savefig(vis_dir/"class_distribution.png", dpi=150)
plt.close()

# 2 Missing values
plt.figure(figsize=(8,4))
miss = df.isna().sum()
sns.barplot(x=miss.index, y=miss.values)
plt.xticks(rotation=45, ha='right')
plt.title("Missing Value Count per Column")
plt.tight_layout()
plt.savefig(images_dir/"missing_values.png", dpi=150)
plt.savefig(vis_dir/"missing_values.png", dpi=150)
plt.close()

# 3 Numerical distributions
num_cols = ["ClaimAmount","PatientAge","PatientIncome"]
for col in num_cols:
    if col in df.columns:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        sns.histplot(df[col], kde=True, bins=50)
        plt.title(f"{col} distribution")
        plt.subplot(1,2,2)
        sns.boxplot(y=df[col])
        plt.title(f"{col} boxplot")
        plt.tight_layout()
        plt.savefig(images_dir/f"{col.lower()}_distribution.png", dpi=150)
        plt.savefig(vis_dir/f"{col.lower()}_distribution.png", dpi=150)
        plt.close()

        # Fraud vs non-fraud comparison
        plt.figure(figsize=(6,4))
        sns.boxplot(x="ClaimLegitimacy", y=col, data=df)
        plt.title(f"{col} by Fraud vs Legitimate")
        plt.tight_layout()
        plt.savefig(images_dir/f"{col.lower()}_fraud_comparison.png", dpi=150)
        plt.savefig(vis_dir/f"{col.lower()}_fraud_comparison.png", dpi=150)
        plt.close()

# 4 Correlation heatmap for numerical
plt.figure(figsize=(8,6))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap Numerical Features")
plt.tight_layout()
plt.savefig(images_dir/"correlation_heatmap.png", dpi=150)
plt.savefig(vis_dir/"correlation_heatmap.png", dpi=150)
plt.close()

# 5 Categorical fraud rates
cat_cols = ["ProviderSpecialty","ClaimType","ClaimStatus","PatientGender"]
for col in cat_cols:
    if col in df.columns:
        plt.figure(figsize=(8,4))
        # fraud rate per category
        fraud_rate = df.groupby(col)["ClaimLegitimacy"].apply(lambda x: (x=="Fraud").mean()).sort_values(ascending=False)
        fraud_rate.plot(kind='bar')
        plt.title(f"Fraud Rate by {col}")
        plt.ylabel("Fraud Rate")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(images_dir/f"fraud_rate_by_{col.lower()}.png", dpi=150)
        plt.savefig(vis_dir/f"fraud_rate_by_{col.lower()}.png", dpi=150)
        plt.close()

# 6 ROC and PR curves from evaluation if available
eval_dir = PROJECT_ROOT / config.get("paths",{}).get("evaluation_dir","evaluation")
try:
    thr_path = eval_dir / "threshold_analysis.csv"
    if thr_path.exists():
        thr_df = pd.read_csv(thr_path)
        if "threshold" in thr_df.columns and "precision" in thr_df.columns and "recall" in thr_df.columns:
            plt.figure(figsize=(6,4))
            plt.plot(thr_df["recall"], thr_df["precision"], marker='o')
            plt.title("Precision-Recall vs Threshold (Validation)")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(images_dir/"threshold_precision_recall.png", dpi=150)
            plt.savefig(vis_dir/"threshold_precision_recall.png", dpi=150)
            plt.close()

            plt.figure(figsize=(6,4))
            plt.plot(thr_df["threshold"], thr_df["f1"], label="F1")
            plt.plot(thr_df["threshold"], thr_df["f2"], label="F2")
            plt.plot(thr_df["threshold"], thr_df["precision"], label="Precision")
            plt.plot(thr_df["threshold"], thr_df["recall"], label="Recall")
            plt.title("Threshold Performance")
            plt.xlabel("Threshold")
            plt.legend()
            plt.tight_layout()
            plt.savefig(images_dir/"threshold_performance.png", dpi=150)
            plt.savefig(vis_dir/"threshold_performance.png", dpi=150)
            plt.close()
except Exception as e:
    print(f"Threshold plots failed {e}")

# 7 Model comparison
try:
    comp_path = eval_dir / "model_comparison.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        if "model" in comp_df.columns and "val_pr_auc" in comp_df.columns:
            plt.figure(figsize=(10,5))
            comp_sorted = comp_df.sort_values("val_pr_auc", ascending=False)
            sns.barplot(x="val_pr_auc", y="model", data=comp_sorted)
            plt.title("Model Comparison - Val PR-AUC")
            plt.tight_layout()
            plt.savefig(images_dir/"model_comparison_pr_auc.png", dpi=150)
            plt.savefig(vis_dir/"model_comparison_pr_auc.png", dpi=150)
            plt.close()

            plt.figure(figsize=(10,5))
            if "fit_time" in comp_sorted.columns:
                sns.barplot(x="fit_time", y="model", data=comp_sorted)
                plt.title("Model Runtime Comparison")
                plt.tight_layout()
                plt.savefig(images_dir/"runtime_comparison.png", dpi=150)
                plt.savefig(vis_dir/"runtime_comparison.png", dpi=150)
                plt.close()
except Exception as e:
    print(f"Model comp plot failed {e}")

# 8 Feature importance
try:
    imp_path = eval_dir / "feature_importance.csv"
    if imp_path.exists():
        imp_df = pd.read_csv(imp_path)
        top20 = imp_df.head(20)
        plt.figure(figsize=(10,6))
        sns.barplot(x="importance", y="feature", data=top20)
        plt.title("Top 20 Feature Importance")
        plt.tight_layout()
        plt.savefig(images_dir/"feature_importance.png", dpi=150)
        plt.savefig(vis_dir/"feature_importance.png", dpi=150)
        plt.close()
except Exception as e:
    print(f"Feature imp plot failed {e}")

# 9 Confusion matrix
try:
    cm_path = eval_dir / "confusion_matrices" / "traditional_ml_cm.csv"
    if cm_path.exists():
        cm = pd.read_csv(cm_path, header=None).values
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix - Traditional ML")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(images_dir/"confusion_matrix.png", dpi=150)
        plt.savefig(vis_dir/"confusion_matrix.png", dpi=150)
        plt.close()
except Exception as e:
    print(f"CM plot failed {e}")

# 10 Anomaly score distribution
try:
    ano_path = eval_dir / "anomaly_scores.json"
    if ano_path.exists():
        import json
        with open(ano_path) as f:
            data = json.load(f)
        # y_test + scores
        y = data.get("y_test",[])
        scores = data.get("scores",{})
        # Plot distribution for first model
        if scores:
            first_model = list(scores.keys())[0]
            sc = scores[first_model]
            plt.figure(figsize=(8,4))
            # Fraud vs legit scores
            sc_arr = np.array(sc)
            y_arr = np.array(y)
            sns.histplot(sc_arr[y_arr==0], label="Legitimate", kde=True, color="blue", alpha=0.5)
            sns.histplot(sc_arr[y_arr==1], label="Fraud", kde=True, color="red", alpha=0.5)
            plt.title(f"Anomaly Score Distribution - {first_model}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(images_dir/"anomaly_score_distribution.png", dpi=150)
            plt.savefig(vis_dir/"anomaly_score_distribution.png", dpi=150)
            plt.close()
except Exception as e:
    print(f"Anomaly plot failed {e}")

# 11 ROC and PR curves using sklearn if we have probs? Use deep learning threshold analysis maybe
try:
    import json
    # For deep learning
    deep_thr_path = eval_dir / "deep_learning_threshold_analysis.csv"
    if deep_thr_path.exists():
        ddf = pd.read_csv(deep_thr_path)
        plt.figure(figsize=(6,4))
        plt.plot(ddf["recall"], ddf["precision"])
        plt.title("DL Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tight_layout()
        plt.savefig(images_dir/"dl_pr_curve.png", dpi=150)
        plt.savefig(vis_dir/"dl_pr_curve.png", dpi=150)
        plt.close()
except Exception as e:
    print(f"DL PR failed {e}")

# 12 End-to-end architecture diagram via matplotlib (simple flow)
plt.figure(figsize=(12,4))
plt.axis('off')
text = """
[Claim Input] -> [Document Intelligence: OCR/VLM] -> [Preprocessing: Scaling, Encoding, Date Features]
-> [Traditional ML (RF, XGB, etc) | Deep Learning MLP | Anomaly Detection (IsolationForest, LOF)]
-> [Policy/RAG Retrieval: Policy Rules, Fraud Indicators]
-> [Agentic Reasoning: Doc Verify, Policy Match, Consistency, Historical]
-> [Hybrid Decision Synthesis: Prob + Anomaly + Doc + Policy]
-> [Explainability: SHAP, Feature Importance, Evidence Citations]
-> [Final Output: APPROVE / FLAG_FOR_MANUAL_REVIEW / REJECT_OR_ESCALATE + Human Review]
"""
plt.text(0.5, 0.5, text, ha='center', va='center', fontsize=9, wrap=True, family='monospace')
plt.title("End-to-End Architecture Flow")
plt.tight_layout()
plt.savefig(images_dir/"architecture_diagram.png", dpi=150)
plt.savefig(vis_dir/"architecture_diagram.png", dpi=150)
plt.close()

# 13 Document validation flow
plt.figure(figsize=(10,3))
plt.axis('off')
txt2="""
[Document Upload] -> [OCR: Tesseract/EasyOCR/PaddleOCR/Fallback]
-> [Type Identification: Bill, Prescription, Discharge]
-> [Field Extraction: Dates, Amounts, Codes, Provider]
-> [Validation: Bill Total, Date Consistency, Duplicate, Missing Docs]
-> [Risk: LOW/MEDIUM/HIGH] -> [Hybrid Pipeline]
"""
plt.text(0.5,0.5,txt2, ha='center', va='center', fontsize=9, family='monospace')
plt.title("Document Validation Flow")
plt.tight_layout()
plt.savefig(images_dir/"document_validation_flow.png", dpi=150)
plt.savefig(vis_dir/"document_validation_flow.png", dpi=150)
plt.close()

print(f"Visualizations saved to {images_dir}")
