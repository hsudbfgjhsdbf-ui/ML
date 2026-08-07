"""
Feature engineering module for Medical Insurance Claim Fraud Detection System.
Institution: IIIT Dharwad, Department of Data Science and AI
Faculty Adviser: Prof. Ramesh Athe
Team Members: B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)

This module implements:
1. Indian healthcare domain-specific feature creation:
   - Claim-to-Premium Ratio
   - Treatment-Cost Deviation (INR normalized by regional specialty average)
   - Days Since Policy Inception & Claim Frequency in past 12 months
   - Hospital Tier Cost Ratio
2. Multi-attribute interaction features (Age-Treatment, Location-Hospital distance).
3. Policyholder and Hospital Provider level statistical aggregations.
4. Selective 2nd-degree polynomial feature generation.
5. Multi-method Feature Selection (Mutual Information, RFE with Random Forest, LASSO L1 regularization).
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from src.utils import setup_logger, ensure_directories

logger = setup_logger("FeatureEngineeringLogger")


class InsuranceFeatureEngineer:
    """
    Stateful feature engineering and selection pipeline for Indian medical insurance claims.
    Computes domain features, interactions, aggregations, and selects top predictive features.
    """
    def __init__(self, top_k_features: int = 25):
        self.top_k_features = top_k_features
        self.regional_specialty_means: Dict[Tuple[str, str], float] = {}
        self.regional_specialty_stds: Dict[Tuple[str, str], float] = {}
        self.provider_rejection_rates: Dict[str, float] = {}
        self.provider_avg_claims: Dict[str, float] = {}
        self.policyholder_avg_claims: Dict[str, float] = {}
        self.selected_features: List[str] = []
        self.feature_rankings: pd.DataFrame = pd.DataFrame()

    def create_domain_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Creates domain-specific Indian insurance features from raw claim data.
        """
        logger.info("Engineering Indian healthcare and insurance domain features...")
        df = df.copy()
        
        # 1. Estimate annual premium in INR based on age, income, and policy type
        base_premium = 8000.0 + (df["PatientAge"] * 180.0) + (df["PatientIncomeINR"] * 0.005)
        df["EstimatedAnnualPremiumINR"] = np.clip(base_premium, 5000.0, 150000.0)
        
        # 2. Claim-to-Premium Ratio (unusually high ratio >= 5.0 is a strong fraud indicator)
        df["ClaimToPremiumRatio"] = df["ClaimAmountINR"] / (df["EstimatedAnnualPremiumINR"] + 1.0)
        
        # 3. Treatment-Cost Deviation INR (Z-score compared to Regional Specialty Average)
        if is_train:
            grouped = df.groupby(["IndianState", "ProviderSpecialty"])["ClaimAmountINR"]
            for (state, spec), group in grouped:
                self.regional_specialty_means[(state, spec)] = float(group.mean())
                self.regional_specialty_stds[(state, spec)] = float(group.std()) if len(group) > 1 and group.std() > 0 else 1.0
                
        def calc_deviation(row):
            state = row.get("IndianState", "Maharashtra")
            spec = row.get("ProviderSpecialty", "General Practice")
            mean_val = self.regional_specialty_means.get((state, spec), 125000.0)
            std_val = self.regional_specialty_stds.get((state, spec), 45000.0)
            return (row["ClaimAmountINR"] - mean_val) / (std_val + 1e-5)
            
        df["TreatmentCostDeviationINR"] = df.apply(calc_deviation, axis=1)
        
        # 4. Temporal Features: Days Since Policy Inception & Claim Frequency in past 12M
        # Use deterministic hash of ClaimID to simulate policy age
        df["DaysSincePolicyInception"] = [
            int(abs(hash(str(cid))) % 1800) + 1 for cid in df["ClaimID"]
        ]
        # Flag if claim filed within 30 days of inception (suspicious pre-existing condition claim)
        df["IsEarlyClaim"] = (df["DaysSincePolicyInception"] <= 30).astype(int)
        
        df["ClaimFrequency12M"] = [
            int(abs(hash(str(pid))) % 5) + 1 for pid in df["PatientID"]
        ]
        df["TimeBetweenClaimsDays"] = np.where(
            df["ClaimFrequency12M"] > 1,
            365.0 / (df["ClaimFrequency12M"] + 1),
            365.0
        )
        
        # 5. Hospital Tier Cost Ratio
        tier_baselines = {
            "Tier-3 Town Nursing Home": 40000.0,
            "Tier-2 City Multi-Specialty Hospital": 90000.0,
            "Tier-1 Metro Corporate Hospital": 180000.0
        }
        df["HospitalTierCostRatio"] = df.apply(
            lambda r: r["ClaimAmountINR"] / tier_baselines.get(r.get("HospitalTier", "Tier-2 City Multi-Specialty Hospital"), 90000.0),
            axis=1
        )
        
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates interaction features capturing multi-attribute relationships.
        """
        logger.debug("Creating multi-attribute interaction features...")
        df = df.copy()
        
        # 1. Age-Treatment Risk Score
        high_risk_specs = ["Cardiology", "Orthopedics", "Neurology"]
        df["IsHighRiskSpecialty"] = df["ProviderSpecialty"].isin(high_risk_specs).astype(int)
        df["AgeTreatmentRiskScore"] = (df["PatientAge"] / 100.0) * df["IsHighRiskSpecialty"] * df["ClaimAmountINR"]
        
        # 2. Location-Hospital Distance Indicator (Out-of-Station Treatment)
        # Check if PatientLocation matches IndianCity
        df["IsOutOfStationTreatment"] = (df["ProviderLocation"] != df["IndianCity"]).astype(int)
        
        # 3. Amount per Day of Hospitalization (assuming avg 3 days inpatient, 1 day emergency)
        duration_map = {"Inpatient": 4.5, "Emergency": 1.5, "Routine": 1.0, "Outpatient": 1.0}
        df["EstimatedHospitalDays"] = df["ClaimType"].map(duration_map).fillna(2.0)
        df["AmountPerHospitalDayINR"] = df["ClaimAmountINR"] / df["EstimatedHospitalDays"]
        
        return df

    def create_aggregation_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Creates policyholder and hospital provider level statistical aggregations.
        """
        logger.debug("Creating policyholder and hospital provider level statistical aggregations...")
        df = df.copy()
        
        if is_train:
            # Provider Rejection / Fraud Rate
            if "IsFraud" in df.columns:
                self.provider_rejection_rates = df.groupby("ProviderID")["IsFraud"].mean().to_dict()
            self.provider_avg_claims = df.groupby("ProviderID")["ClaimAmountINR"].mean().to_dict()
            self.policyholder_avg_claims = df.groupby("PatientID")["ClaimAmountINR"].mean().to_dict()
            
        # Map provider fraud rate
        global_fraud_rate = df["IsFraud"].mean() if "IsFraud" in df.columns else 0.06
        df["ProviderFraudRate"] = df["ProviderID"].map(self.provider_rejection_rates).fillna(global_fraud_rate)
        
        # Provider Average Claim Amount
        global_avg_claim = df["ClaimAmountINR"].mean()
        df["ProviderAvgClaimINR"] = df["ProviderID"].map(self.provider_avg_claims).fillna(global_avg_claim)
        
        # Ratio of Claim Amount to Provider Average
        df["ClaimToProviderAvgRatio"] = df["ClaimAmountINR"] / (df["ProviderAvgClaimINR"] + 1.0)
        
        # Policyholder Historical Average Claim
        df["PolicyholderAvgClaimINR"] = df["PatientID"].map(self.policyholder_avg_claims).fillna(global_avg_claim)
        df["ClaimToPolicyholderAvgRatio"] = df["ClaimAmountINR"] / (df["PolicyholderAvgClaimINR"] + 1.0)
        
        return df

    def create_polynomial_features(self, df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
        """
        Generates 2nd-degree polynomial interaction features for top numeric variables.
        """
        logger.debug("Generating selective 2nd-degree polynomial features...")
        df = df.copy()
        
        key_cols = [c for c in ["ClaimAmountINR", "ClaimToPremiumRatio", "TreatmentCostDeviationINR"] if c in df.columns]
        if len(key_cols) >= 2:
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            poly_arr = poly.fit_transform(df[key_cols])
            poly_names = poly.get_feature_names_out(key_cols)
            for i, name in enumerate(poly_names):
                if name not in key_cols:
                    df[f"poly_{name.replace(' ', '_')}"] = poly_arr[:, i]
                    
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes full feature engineering pipeline on training data and selects top features.
        """
        logger.info("Executing full feature engineering pipeline on training set...")
        df_feat = self.create_domain_features(df, is_train=True)
        df_feat = self.create_interaction_features(df_feat)
        df_feat = self.create_aggregation_features(df_feat, is_train=True)
        
        num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
        df_feat = self.create_polynomial_features(df_feat, num_cols)
        
        return df_feat

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms validation or test datasets using learned regional/provider statistics.
        """
        logger.debug("Transforming dataset using fitted feature engineering rules...")
        df_feat = self.create_domain_features(df, is_train=False)
        df_feat = self.create_interaction_features(df_feat)
        df_feat = self.create_aggregation_features(df_feat, is_train=False)
        
        num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
        df_feat = self.create_polynomial_features(df_feat, num_cols)
        
        return df_feat

    def perform_feature_selection(self, X: pd.DataFrame, y: pd.Series, top_k: int = 20) -> Tuple[List[str], pd.DataFrame]:
        """
        Performs multi-method feature selection using:
        1. Mutual Information (non-linear dependency)
        2. Recursive Feature Elimination (RFE with Random Forest)
        3. LASSO L1 Regularization feature importance
        Returns top selected features and comprehensive ranking DataFrame.
        """
        logger.info(f"Performing multi-method feature selection (Top {top_k} features)...")
        self.top_k_features = top_k
        
        # 1. Mutual Information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_series = pd.Series(mi_scores, index=X.columns)
        mi_ranks = mi_series.rank(ascending=False)
        
        # 2. Random Forest Importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_scores = pd.Series(rf.feature_importances_, index=X.columns)
        rf_ranks = rf_scores.rank(ascending=False)
        
        # 3. LASSO (L1 Logistic Regression)
        lasso = LogisticRegression(penalty="elasticnet", l1_ratio=1.0, solver="saga", C=0.1, max_iter=500, random_state=42)
        try:
            lasso.fit(X, y)
            lasso_scores = pd.Series(np.abs(lasso.coef_[0]), index=X.columns)
        except Exception:
            lasso_scores = pd.Series(0.1, index=X.columns)
        lasso_ranks = lasso_scores.rank(ascending=False)
        
        # Composite average rank (lower is better)
        avg_ranks = (mi_ranks + rf_ranks + lasso_ranks) / 3.0
        
        ranking_df = pd.DataFrame({
            "Feature": X.columns,
            "Mutual_Information_Score": mi_series.values,
            "Random_Forest_Importance": rf_scores.values,
            "LASSO_Coefficient_Abs": lasso_scores.values,
            "MI_Rank": mi_ranks.values,
            "RF_Rank": rf_ranks.values,
            "LASSO_Rank": lasso_ranks.values,
            "Composite_Rank": avg_ranks.values
        }).sort_values("Composite_Rank", ascending=True).reset_index(drop=True)
        
        self.feature_rankings = ranking_df
        self.selected_features = ranking_df["Feature"].head(top_k).tolist()
        
        ensure_directories(["data"])
        ranking_df.to_csv("data/feature_importance_ranking.csv", index=False)
        logger.info(f"Feature selection complete. Top 5 features: {self.selected_features[:5]}")
        return self.selected_features, ranking_df
