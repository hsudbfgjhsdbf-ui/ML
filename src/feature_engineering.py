"""
Advanced Feature Engineering and Automated Feature Selection Engine.
Constructs domain-specific, temporal, interaction, polynomial, and provider-level
statistical aggregations tailored for Indian medical insurance fraud detection.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.feature_selection import (
    mutual_info_classif, SelectKBest, f_classif,
    RFECV
)
from sklearn.ensemble import RandomForestClassifier

from src.config import config, RANDOM_SEED
from src.utils import logger

class InsuranceFeatureEngineer:
    """
    Transforms raw claim records into rich, multi-dimensional feature representations
    incorporating Indian actuarial and clinical domain logic.
    """
    
    def __init__(self):
        # Learned statistics on training partition
        self.hospital_stats_: Dict[str, Dict[str, float]] = {}
        self.tier_diag_stats_: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.provider_stats_: Dict[str, Dict[str, float]] = {}
        self.global_claim_mean_: float = 75000.0
        self.global_claim_std_: float = 65000.0
        self.selected_feature_names_: List[str] = []
        self.feature_importances_: Dict[str, float] = {}
        
    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits historical and group-level aggregations on training data only."""
        data = df.copy()
        
        # 1. Global baseline statistics
        if "Claim_Amount_INR" in data.columns:
            self.global_claim_mean_ = float(data["Claim_Amount_INR"].mean())
            self.global_claim_std_ = float(data["Claim_Amount_INR"].std() + 1e-5)
        elif "ClaimAmount" in data.columns:
            self.global_claim_mean_ = float(data["ClaimAmount"].mean())
            self.global_claim_std_ = float(data["ClaimAmount"].std() + 1e-5)
            
        # 2. Hospital-level aggregations
        hosp_col = "Hospital_Name" if "Hospital_Name" in data.columns else "ProviderID"
        amt_col = "Claim_Amount_INR" if "Claim_Amount_INR" in data.columns else "ClaimAmount"
        
        if hosp_col in data.columns:
            grouped = data.groupby(hosp_col)
            for hosp, group in grouped:
                self.hospital_stats_[str(hosp)] = {
                    "mean_claim": float(group[amt_col].mean()),
                    "std_claim": float(group[amt_col].std() + 1e-5),
                    "volume": float(len(group)),
                    "max_claim": float(group[amt_col].max())
                }
                
        # 3. Hospital Tier x Diagnosis Category baseline costs
        tier_col = "Hospital_Tier" if "Hospital_Tier" in data.columns else "Cluster"
        diag_col = "Diagnosis_Category" if "Diagnosis_Category" in data.columns else "DiagnosisCode"
        
        if tier_col in data.columns and diag_col in data.columns:
            grouped_td = data.groupby([tier_col, diag_col])
            for (tier, diag), group in grouped_td:
                self.tier_diag_stats_[(str(tier), str(diag))] = {
                    "mean": float(group[amt_col].mean()),
                    "std": float(group[amt_col].std() + 1e-5),
                    "median": float(group[amt_col].median())
                }
                
        logger.info(f"Fitted Feature Engineer: {len(self.hospital_stats_)} hospital profiles and {len(self.tier_diag_stats_)} tier-diagnosis combinations.")
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms input claims dataframe into enriched engineered representation."""
        data = df.copy()
        
        # Determine standard column aliases
        amt_col = "Claim_Amount_INR" if "Claim_Amount_INR" in data.columns else "ClaimAmount"
        prem_col = "Annual_Premium_INR" if "Annual_Premium_INR" in data.columns else None
        sum_col = "Sum_Insured_INR" if "Sum_Insured_INR" in data.columns else None
        inc_col = "Annual_Income_INR" if "Annual_Income_INR" in data.columns else "PatientIncome"
        dur_col = "Hospitalization_Duration_Days" if "Hospitalization_Duration_Days" in data.columns else None
        age_col = "Patient_Age" if "Patient_Age" in data.columns else "PatientAge"
        hosp_col = "Hospital_Name" if "Hospital_Name" in data.columns else "ProviderID"
        tier_col = "Hospital_Tier" if "Hospital_Tier" in data.columns else "Cluster"
        diag_col = "Diagnosis_Category" if "Diagnosis_Category" in data.columns else "DiagnosisCode"
        
        # 1. Domain Ratios
        if prem_col and prem_col in data.columns:
            data["Claim_to_Premium_Ratio"] = data[amt_col] / (data[prem_col] + 1.0)
        else:
            data["Claim_to_Premium_Ratio"] = data[amt_col] / 25000.0
            
        if sum_col and sum_col in data.columns:
            data["Sum_Insured_Utilization"] = (data[amt_col] / (data[sum_col] + 1.0)).clip(0, 3.0)
        else:
            data["Sum_Insured_Utilization"] = (data[amt_col] / 500000.0).clip(0, 3.0)
            
        if inc_col in data.columns:
            data["Income_to_Claim_Ratio"] = data[amt_col] / (data[inc_col] + 1.0)
            data["Income_Log"] = np.log1p(data[inc_col].clip(lower=0))
        else:
            data["Income_to_Claim_Ratio"] = 0.1
            data["Income_Log"] = 10.0
            
        if dur_col and dur_col in data.columns:
            data["Cost_Per_Day_INR"] = data[amt_col] / (data[dur_col] + 1.0)
            data["Amount_x_Duration"] = data[amt_col] * (data[dur_col] + 1.0)
            data["Age_x_Duration"] = data[age_col] * (data[dur_col] + 1.0)
        else:
            data["Cost_Per_Day_INR"] = data[amt_col] / 4.0
            data["Amount_x_Duration"] = data[amt_col] * 4.0
            data["Age_x_Duration"] = data[age_col] * 4.0
            
        # 2. Treatment Cost Deviation from Norm
        dev_list = []
        for idx, row in data.iterrows():
            tier = str(row.get(tier_col, "Unknown"))
            diag = str(row.get(diag_col, "Unknown"))
            stats = self.tier_diag_stats_.get((tier, diag))
            val = float(row[amt_col])
            if stats:
                z = (val - stats["mean"]) / stats["std"]
            else:
                z = (val - self.global_claim_mean_) / self.global_claim_std_
            dev_list.append(np.clip(z, -5.0, 10.0))
        data["Treatment_Cost_Deviation"] = dev_list
        
        # 3. Provider / Hospital Risk Statistics
        hosp_means, hosp_vols, hosp_ratios = [], [], []
        for idx, row in data.iterrows():
            hosp = str(row.get(hosp_col, "Unknown"))
            stats = self.hospital_stats_.get(hosp)
            val = float(row[amt_col])
            if stats:
                hosp_means.append(stats["mean_claim"])
                hosp_vols.append(stats["volume"])
                hosp_ratios.append(val / (stats["mean_claim"] + 1.0))
            else:
                hosp_means.append(self.global_claim_mean_)
                hosp_vols.append(1.0)
                hosp_ratios.append(val / (self.global_claim_mean_ + 1.0))
                
        data["Hospital_Benchmark_Mean"] = hosp_means
        data["Hospital_Volume"] = hosp_vols
        data["Claim_to_Hospital_Avg_Ratio"] = hosp_ratios
        
        # 4. Temporal and Behavioral Flags
        if "Policy_Duration_Months" in data.columns and "Waiting_Period_Months" in data.columns:
            data["Early_Claim_Flag"] = (
                data["Policy_Duration_Months"] <= (data["Waiting_Period_Months"] + 2)
            ).astype(int)
            data["Waiting_Period_Delta"] = data["Policy_Duration_Months"] - data["Waiting_Period_Months"]
        else:
            data["Early_Claim_Flag"] = 0
            data["Waiting_Period_Delta"] = 12.0
            
        if "Prior_Claims_Count" in data.columns:
            prior_cnt = data["Prior_Claims_Count"]
            rej_cnt = data.get("Rejected_Prior_Claims", pd.Series(0, index=data.index))
            tot_amt = data.get("Total_Prior_Claimed_INR", pd.Series(0, index=data.index))
            
            data["Prior_Rejection_Ratio"] = rej_cnt / (prior_cnt + 1.0)
            data["Avg_Prior_Claim_Amount"] = tot_amt / (prior_cnt + 1e-5)
            data["Current_to_Prior_Avg_Ratio"] = data[amt_col] / (data["Avg_Prior_Claim_Amount"] + 1.0)
            data["Claim_Velocity_Risk"] = prior_cnt * np.log1p(tot_amt)
        else:
            data["Prior_Rejection_Ratio"] = 0.0
            data["Avg_Prior_Claim_Amount"] = 0.0
            data["Current_to_Prior_Avg_Ratio"] = 1.0
            data["Claim_Velocity_Risk"] = 0.0
            
        # 5. Non-linear & Polynomial Terms
        data["Claim_Amount_Log"] = np.log1p(data[amt_col].clip(lower=0))
        data["Claim_to_Premium_Squared"] = (data["Claim_to_Premium_Ratio"] ** 2).clip(0, 500)
        data["Utilization_Squared"] = data["Sum_Insured_Utilization"] ** 2
        data["Age_x_Cost_Deviation"] = data[age_col] * data["Treatment_Cost_Deviation"]
        
        return data

def run_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k_best: int = 25,
    random_state: int = RANDOM_SEED
) -> Tuple[List[str], Dict[str, float]]:
    """
    Evaluates feature importance using tree ensembles and mutual information.
    Returns ranked feature names and score rankings.
    """
    logger.info(f"Running automated feature selection over {X_train.shape[1]} engineered features.")
    
    # Filter only numeric columns for selection ranking
    num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    X_num = X_train[num_cols].fillna(0.0)
    
    # Random Forest Importance
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1)
    rf.fit(X_num, y_train)
    
    # Mutual Information
    mi_scores = mutual_info_classif(X_num, y_train, random_state=random_state)
    
    importances = {}
    for idx, col in enumerate(num_cols):
        # Blended score: 60% Tree Gini Importance + 40% Normalized Mutual Information
        tree_imp = rf.feature_importances_[idx]
        mi_norm = mi_scores[idx] / (np.max(mi_scores) + 1e-8)
        blended = 0.6 * tree_imp + 0.4 * mi_norm
        importances[col] = float(blended)
        
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    top_k_names = [f[0] for f in sorted_features[:k_best]]
    
    logger.info(f"Top 5 most predictive fraud features: {top_k_names[:5]}")
    return top_k_names, dict(sorted_features)
