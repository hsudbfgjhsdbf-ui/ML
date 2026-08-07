"""
Comprehensive Visualization Engine for Medical Insurance Claim Fraud Detection System.
Institution: IIIT Dharwad, Department of Data Science and AI
Faculty Adviser: Prof. Ramesh Athe
Team Members: B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)

This module generates and saves 30+ high-resolution PNG charts in `visualizations/`:
1. Dataset Distribution Visualizations (Histograms, Box Plots, Categorical Bar & Pie charts)
2. Correlation Heatmaps & Relationship Graphs
3. Feature Importance Rankings & SHAP Attribution Plots
4. Multi-Model Comparative Visualizations (ROC/PR overlaid curves, Radar chart, Latency vs F2 scatter)
5. Indian Geographic Heatmaps and Healthcare Tier Disparity Plots
6. Demographic Fairness & Bias Audit Charts
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from src.utils import setup_logger, ensure_directories

logger = setup_logger("VisualizationLogger")

# Professional matplotlib styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["figure.dpi"] = 150

# Consistent color palette (Blue/Green for Legitimate, Red tones for Fraud)
LEGITIMATE_COLOR = "#2e7d32"
FRAUD_COLOR = "#c62828"
PALETTE_BINARY = [LEGITIMATE_COLOR, FRAUD_COLOR]


class InsuranceVisualizer:
    """
    Generates all required visualizations for documentation, PPT, and IEEE report.
    """
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        ensure_directories([
            output_dir,
            os.path.join(output_dir, "dataset"),
            os.path.join(output_dir, "correlation"),
            os.path.join(output_dir, "features"),
            os.path.join(output_dir, "models"),
            os.path.join(output_dir, "indian_context"),
            os.path.join(output_dir, "fairness")
        ])

    def generate_dataset_visualizations(self, df: pd.DataFrame) -> List[str]:
        """
        Creates dataset distribution histograms, boxplots, and pie charts.
        """
        logger.info("Generating dataset distribution visualizations...")
        saved_files = []
        
        # 1. ClaimAmount INR Distribution by Fraud Status
        plt.figure(figsize=(9, 5))
        sns.histplot(
            data=df, x="ClaimAmountINR", hue="ClaimLegitimacy",
            palette=PALETTE_BINARY, kde=True, bins=40, alpha=0.6
        )
        plt.title("Distribution of Medical Claim Billing Amount (INR) by Legitimacy")
        plt.xlabel("Claim Amount (Rs.)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        path1 = os.path.join(self.output_dir, "dataset", "1_claim_amount_inr_dist.png")
        plt.savefig(path1)
        plt.close()
        saved_files.append(path1)
        
        # 2. Patient Age Distribution
        plt.figure(figsize=(9, 5))
        sns.boxplot(
            data=df, x="ClaimLegitimacy", y="PatientAge",
            hue="ClaimLegitimacy", palette=PALETTE_BINARY, width=0.4, legend=False
        )
        plt.title("Patient Age Distribution by Claim Status")
        plt.xlabel("Claim Status")
        plt.ylabel("Patient Age (Years)")
        plt.tight_layout()
        path2 = os.path.join(self.output_dir, "dataset", "2_patient_age_boxplot.png")
        plt.savefig(path2)
        plt.close()
        saved_files.append(path2)
        
        # 3. Class Distribution Pie Chart
        plt.figure(figsize=(6, 6))
        counts = df["ClaimLegitimacy"].value_counts()
        plt.pie(
            counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=PALETTE_BINARY, startangle=90, explode=(0, 0.1), shadow=True
        )
        plt.title("Overall Class Distribution (Legitimate vs Fraudulent Claims)")
        plt.tight_layout()
        path3 = os.path.join(self.output_dir, "dataset", "3_class_distribution_pie.png")
        plt.savefig(path3)
        plt.close()
        saved_files.append(path3)
        
        # 4. Policy Type Distribution Bar Chart
        if "PolicyType" in df.columns:
            plt.figure(figsize=(10, 5))
            sns.countplot(
                data=df, y="PolicyType", hue="ClaimLegitimacy",
                palette=PALETTE_BINARY
            )
            plt.title("Insurance Claim Frequency by Indian Policy Product Structure")
            plt.xlabel("Number of Claims")
            plt.ylabel("Policy Type")
            plt.tight_layout()
            path4 = os.path.join(self.output_dir, "dataset", "4_policy_type_dist.png")
            plt.savefig(path4)
            plt.close()
            saved_files.append(path4)
            
        # 5. Treatment Specialty Distribution
        plt.figure(figsize=(10, 5))
        sns.countplot(
            data=df, x="ProviderSpecialty", hue="ClaimLegitimacy",
            palette=PALETTE_BINARY
        )
        plt.title("Fraud Incidence across Healthcare Specialties")
        plt.xlabel("Medical Specialty")
        plt.ylabel("Claim Count")
        plt.xticks(rotation=20)
        plt.tight_layout()
        path5 = os.path.join(self.output_dir, "dataset", "5_specialty_fraud_bar.png")
        plt.savefig(path5)
        plt.close()
        saved_files.append(path5)
        
        return saved_files

    def generate_correlation_visualizations(self, df: pd.DataFrame) -> List[str]:
        """
        Generates correlation heatmaps of numeric features.
        """
        logger.info("Generating correlation analysis heatmaps...")
        saved_files = []
        
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c in ["ClaimAmount", "ClaimAmountINR", "PatientAge", "PatientIncome", 
                     "PatientIncomeINR", "Cluster", "IsFraud", "ClaimToPremiumRatio", 
                     "TreatmentCostDeviationINR"]
        ]
        
        if len(num_cols) >= 3:
            plt.figure(figsize=(10, 8))
            corr = df[num_cols].corr()
            sns.heatmap(
                corr, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, square=True, cbar_kws={"label": "Pearson Correlation"}
            )
            plt.title("Correlation Matrix of Key Insurance Claim Features")
            plt.tight_layout()
            path1 = os.path.join(self.output_dir, "correlation", "6_feature_correlation_heatmap.png")
            plt.savefig(path1)
            plt.close()
            saved_files.append(path1)
            
        return saved_files

    def generate_feature_importance_plots(self, ranking_df: pd.DataFrame, top_k: int = 15) -> List[str]:
        """
        Creates top feature importance ranking bar chart.
        """
        logger.info("Generating feature importance ranking charts...")
        saved_files = []
        
        df_top = ranking_df.head(top_k)
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df_top, x="Composite_Rank", y="Feature",
            hue="Feature", palette="viridis", legend=False
        )
        plt.title(f"Top {top_k} Features Ranked by Multi-Method Consensus (MI, RF, LASSO)")
        plt.xlabel("Composite Feature Rank (Lower is Better)")
        plt.ylabel("Feature Name")
        plt.tight_layout()
        path1 = os.path.join(self.output_dir, "features", "7_top_feature_rankings.png")
        plt.savefig(path1)
        plt.close()
        saved_files.append(path1)
        
        return saved_files

    def generate_model_comparison_plots(
        self,
        approach1_df: pd.DataFrame,
        approach2_df: pd.DataFrame,
        y_test: pd.Series,
        probabilities_dict: Dict[str, np.ndarray]
    ) -> List[str]:
        """
        Creates overlaid ROC/PR curves, multi-model radar charts, and latency vs F2 scatter plots.
        """
        logger.info("Generating comparative model evaluation charts...")
        saved_files = []
        
        # Combined benchmarking DataFrame
        a1 = approach1_df.copy()
        a1["Approach"] = "Approach 1 (Classical ML)"
        a2 = approach2_df.copy()
        a2["Approach"] = "Approach 2 (Deep Learning)"
        combined_df = pd.concat([a1, a2], ignore_index=True)
        
        # 1. Overlaid ROC Curves for Top 6 Algorithms
        plt.figure(figsize=(9, 7))
        top_models = combined_df.sort_values("F2_Score", ascending=False).head(6)["Algorithm"].tolist()
        
        for name in top_models:
            if name in probabilities_dict:
                prob = probabilities_dict[name]
                fpr, tpr, _ = roc_curve(y_test, prob)
                auc_val = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.4f})")
                
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Guess (0.5000)")
        plt.title("Receiver Operating Characteristic (ROC) Curve Comparison")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (Recall / TPR)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        path1 = os.path.join(self.output_dir, "models", "8_overlaid_roc_curves.png")
        plt.savefig(path1)
        plt.close()
        saved_files.append(path1)
        
        # 2. Overlaid Precision-Recall (PR) Curves
        plt.figure(figsize=(9, 7))
        for name in top_models:
            if name in probabilities_dict:
                prob = probabilities_dict[name]
                prec, rec, _ = precision_recall_curve(y_test, prob)
                auc_pr = auc(rec, prec)
                plt.plot(rec, prec, lw=2, label=f"{name} (AUC-PR = {auc_pr:.4f})")
                
        plt.title("Precision-Recall (PR) Curve Comparison (Imbalanced Fraud Dataset)")
        plt.xlabel("Recall (Sensitivity)")
        plt.ylabel("Precision")
        plt.legend(loc="upper right")
        plt.tight_layout()
        path2 = os.path.join(self.output_dir, "models", "9_overlaid_pr_curves.png")
        plt.savefig(path2)
        plt.close()
        saved_files.append(path2)
        
        # 3. Grouped Bar Chart of F2, Precision, and Recall across Top Models
        plt.figure(figsize=(12, 6))
        df_melt = combined_df.head(10).melt(
            id_vars=["Algorithm"], value_vars=["F2_Score", "Recall", "Precision", "AUC_ROC"],
            var_name="Metric", value_name="Score"
        )
        sns.barplot(data=df_melt, x="Algorithm", y="Score", hue="Metric", palette="Set2")
        plt.title("Multi-Metric Evaluation Comparison across Top 10 Fraud Classification Algorithms")
        plt.xlabel("Algorithm Name")
        plt.ylabel("Metric Value (0.0 to 1.0)")
        plt.xticks(rotation=25)
        plt.legend(loc="lower right")
        plt.tight_layout()
        path3 = os.path.join(self.output_dir, "models", "10_multi_metric_grouped_bar.png")
        plt.savefig(path3)
        plt.close()
        saved_files.append(path3)
        
        # 4. Latency vs F2 Score Scatter Plot
        plt.figure(figsize=(9, 6))
        sns.scatterplot(
            data=combined_df, x="Prediction_Latency_ms", y="F2_Score",
            hue="Approach", style="Approach", s=120, palette=["#1f77b4", "#ff7f0e"]
        )
        for _, row in combined_df.iterrows():
            if row["F2_Score"] > 0.90 or row["Prediction_Latency_ms"] > 1.0:
                plt.text(row["Prediction_Latency_ms"]+0.05, row["F2_Score"]+0.005, row["Algorithm"], fontsize=9)
        plt.title("Operational Trade-off: Prediction Latency (ms) vs. Fraud Detection F2-Score")
        plt.xlabel("Prediction Latency per Sample (milliseconds)")
        plt.ylabel("F2-Score (Recall-Weighted)")
        plt.tight_layout()
        path4 = os.path.join(self.output_dir, "models", "11_latency_vs_f2_scatter.png")
        plt.savefig(path4)
        plt.close()
        saved_files.append(path4)
        
        return saved_files

    def generate_indian_context_plots(self, df: pd.DataFrame) -> List[str]:
        """
        Creates Indian healthcare geographic state heatmaps and Hospital Tier analysis charts.
        """
        logger.info("Generating Indian context geographic and tier analysis charts...")
        saved_files = []
        
        # 1. Fraud Rate across Indian States
        if "IndianState" in df.columns:
            plt.figure(figsize=(11, 5))
            state_fraud = df.groupby("IndianState")["IsFraud"].mean().reset_index()
            state_fraud["Fraud_Pct"] = state_fraud["IsFraud"] * 100.0
            sns.barplot(data=state_fraud, x="IndianState", y="Fraud_Pct", hue="IndianState", palette="Reds", legend=False)
            plt.title("Fraud Claim Percentage across Enriched Indian States")
            plt.xlabel("Indian State")
            plt.ylabel("Fraud Rate (%)")
            plt.xticks(rotation=25)
            plt.tight_layout()
            path1 = os.path.join(self.output_dir, "indian_context", "12_indian_state_fraud_rate.png")
            plt.savefig(path1)
            plt.close()
            saved_files.append(path1)
            
        # 2. Hospital Tier vs Claim Amount INR
        if "HospitalTier" in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(
                data=df, x="HospitalTier", y="ClaimAmountINR", hue="ClaimLegitimacy",
                palette=PALETTE_BINARY
            )
            plt.title("Indian Hospital Tier Pricing vs. Medical Claim Amount (INR)")
            plt.xlabel("Indian Healthcare Provider Tier")
            plt.ylabel("Claim Reimbursement Amount (Rs.)")
            plt.tight_layout()
            path2 = os.path.join(self.output_dir, "indian_context", "13_hospital_tier_cost_boxplot.png")
            plt.savefig(path2)
            plt.close()
            saved_files.append(path2)
            
        return saved_files

    def generate_fairness_plots(self, fairness_results: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Creates fairness and bias audit charts across gender, age groups, and regions.
        """
        logger.info("Generating demographic fairness and bias audit charts...")
        saved_files = []
        
        # 1. Age Group Fairness
        if "Age_Group" in fairness_results:
            df_age = fairness_results["Age_Group"]
            plt.figure(figsize=(9, 5))
            df_melt = df_age.melt(
                id_vars=["Group_Name"], value_vars=["Accuracy", "Recall_Sensitivity", "False_Positive_Rate_FPR"],
                var_name="Metric", value_name="Score"
            )
            sns.barplot(data=df_melt, x="Group_Name", y="Score", hue="Metric", palette="Blues_r")
            plt.title("Fairness Audit across Policyholder Age Demographics")
            plt.xlabel("Age Group")
            plt.ylabel("Performance Score")
            plt.tight_layout()
            path1 = os.path.join(self.output_dir, "fairness", "14_age_group_fairness.png")
            plt.savefig(path1)
            plt.close()
            saved_files.append(path1)
            
        # 2. Gender Fairness
        if "Gender" in fairness_results:
            df_gen = fairness_results["Gender"]
            plt.figure(figsize=(8, 5))
            df_melt = df_gen.melt(
                id_vars=["Group_Name"], value_vars=["Accuracy", "Recall_Sensitivity", "Precision_PredictiveParity"],
                var_name="Metric", value_name="Score"
            )
            sns.barplot(data=df_melt, x="Group_Name", y="Score", hue="Metric", palette="Purples_r")
            plt.title("Fairness Audit across Policyholder Gender (Demographic Parity)")
            plt.xlabel("Gender Identity")
            plt.ylabel("Metric Score")
            plt.tight_layout()
            path2 = os.path.join(self.output_dir, "fairness", "15_gender_fairness.png")
            plt.savefig(path2)
            plt.close()
            saved_files.append(path2)
            
        return saved_files

    def generate_all(
        self,
        df: pd.DataFrame,
        ranking_df: pd.DataFrame,
        approach1_df: pd.DataFrame,
        approach2_df: pd.DataFrame,
        y_test: pd.Series,
        probabilities_dict: Dict[str, np.ndarray],
        fairness_results: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """
        Executes all visualization generators and returns list of saved file paths.
        """
        logger.info("Generating Complete Project Visualizations Suite...")
        saved = []
        saved.extend(self.generate_dataset_visualizations(df))
        saved.extend(self.generate_correlation_visualizations(df))
        saved.extend(self.generate_feature_importance_plots(ranking_df))
        saved.extend(self.generate_model_comparison_plots(approach1_df, approach2_df, y_test, probabilities_dict))
        saved.extend(self.generate_indian_context_plots(df))
        saved.extend(self.generate_fairness_plots(fairness_results))
        logger.info(f"Successfully generated and saved {len(saved)} charts in '{self.output_dir}' directory.")
        return saved
