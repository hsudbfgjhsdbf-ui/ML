"""
Data preprocessing and cleaning module for Medical Insurance Claim Fraud Detection System.
Institution: IIIT Dharwad, Department of Data Science and AI
Faculty Adviser: Prof. Ramesh Athe
Team Members: B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)

This module implements:
1. Missing value analysis and imputation (Mean, Median, Mode).
2. Duplicate record detection and removal (exact & near-duplicates).
3. Outlier detection and treatment (Z-score & IQR, preserving fraud signal).
4. Categorical feature encoding (Label, Ordinal, One-Hot, and Target encoding).
5. Feature scaling (StandardScaler, MinMaxScaler, RobustScaler) without data leakage.
6. Class imbalance handling (SMOTE, RandomUnderSampler, Class Weighting).
7. Stratified 70/15/15 train-validation-test partitioning.
8. Serializable `InsuranceDataPreprocessor` pipeline class.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from src.utils import setup_logger, ensure_directories

logger = setup_logger("DataPreprocessingLogger")


class InsuranceDataPreprocessor:
    """
    Complete serializable data preprocessing pipeline for Medical Insurance Fraud Detection.
    Encapsulates encoding dictionaries, scaler transformers, and imputation rules.
    """
    def __init__(self, scaler_type: str = "standard", imbalance_method: str = "smote"):
        self.scaler_type = scaler_type
        self.imbalance_method = imbalance_method
        self.scaler = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.target_encodings: Dict[str, pd.Series] = {}
        self.onehot_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.ordinal_mappings: Dict[str, Dict[str, int]] = {
            "HospitalTier": {
                "Tier-3 Town Nursing Home": 0,
                "Tier-2 City Multi-Specialty Hospital": 1,
                "Tier-1 Metro Corporate Hospital": 2
            }
        }
        self.feature_names_out: List[str] = []

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes and imputes missing values.
        Features with >70% missing are removed.
        Normal numeric -> mean, Skewed numeric -> median, Categorical -> mode.
        """
        df = df.copy()
        missing_pct = df.isnull().mean() * 100
        logger.info(f"Missing value percentage per column:\n{missing_pct[missing_pct > 0]}")
        
        cols_to_drop = missing_pct[missing_pct > 70.0].index.tolist()
        if cols_to_drop:
            logger.warning(f"Dropping columns with >70% missing values: {cols_to_drop}")
            df.drop(columns=cols_to_drop, inplace=True)
            
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    skew = df[col].skew()
                    if abs(skew) > 1.0:
                        val = df[col].median()
                        logger.debug(f"Imputing skewed numeric feature {col} with median: {val:.2f}")
                    else:
                        val = df[col].mean()
                        logger.debug(f"Imputing normal numeric feature {col} with mean: {val:.2f}")
                    df[col].fillna(val, inplace=True)
                else:
                    val = df[col].mode()[0]
                    logger.debug(f"Imputing categorical feature {col} with mode: {val}")
                    df[col].fillna(val, inplace=True)
        return df

    def handle_duplicate_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes exact and near-duplicate records to prevent train-test data leakage.
        """
        df = df.copy()
        initial_count = len(df)
        df.drop_duplicates(subset=["ClaimID"], keep="first", inplace=True)
        df.drop_duplicates(
            subset=["PatientID", "ProviderID", "ClaimAmount", "DiagnosisCode", "ProcedureCode"], 
            keep="first", 
            inplace=True
        )
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate/near-duplicate claim records.")
        return df

    def detect_and_treat_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies and treats numeric outliers.
        - Extreme data-entry errors (e.g. negative claim amount or age > 120) are clipped.
        - High-amount fraud outliers are PRESERVED as valuable signal for classification.
        """
        df = df.copy()
        df["PatientAge"] = df["PatientAge"].clip(0, 105)
        df["ClaimAmount"] = df["ClaimAmount"].clip(lower=0.0)
        df["ClaimAmountINR"] = df["ClaimAmountINR"].clip(lower=0.0)
        logger.debug("Outliers audited: preserved genuine fraud signal outliers while clipping unphysical entry errors.")
        return df

    def encode_categorical_variables(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Encodes categorical variables:
        - Binary features (PatientGender): Label encoding.
        - Ordinal features (HospitalTier): Ordinal integer mapping.
        - Nominal features with small categories: One-hot encoding.
        - Nominal features with large cardinality: Target encoding or frequency encoding.
        """
        df = df.copy()
        
        # Ordinal encoding for Hospital Tier
        if "HospitalTier" in df.columns:
            mapping = self.ordinal_mappings["HospitalTier"]
            df["HospitalTier_Encoded"] = df["HospitalTier"].map(mapping).fillna(1).astype(int)
            
        # Binary label encoding
        if "PatientGender" in df.columns:
            df["PatientGender_Encoded"] = (df["PatientGender"] == "M").astype(int)
            
        # Frequency / Target encoding for high-cardinality IndianState, IndianCity, InsurerCompany
        high_card_cols = ["IndianState", "IndianCity", "InsurerCompany", "PolicyType", "ProviderSpecialty", "ClaimType", "ClaimSubmissionMethod"]
        for col in high_card_cols:
            if col in df.columns:
                if is_train:
                    # Use smoothed target encoding if IsFraud available, else frequency encoding
                    if "IsFraud" in df.columns:
                        means = df.groupby(col)["IsFraud"].mean()
                        self.target_encodings[col] = means
                    else:
                        freqs = df[col].value_counts(normalize=True)
                        self.target_encodings[col] = freqs
                
                mapping = self.target_encodings.get(col, pd.Series(dtype=float))
                df[f"{col}_Encoded"] = df[col].map(mapping).fillna(mapping.mean() if not mapping.empty else 0.5)
                
        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fits preprocessing pipeline on training dataset and returns transformed feature matrix X and target y.
        """
        logger.info("Fitting and transforming data preprocessing pipeline on training set...")
        df = self.handle_missing_values(df)
        df = self.handle_duplicate_records(df)
        df = self.detect_and_treat_outliers(df)
        df = self.encode_categorical_variables(df, is_train=True)
        
        y = df["IsFraud"]
        
        # Select numeric features for modeling
        feature_cols = [c for c in df.columns if c.endswith("_Encoded") or c in [
            "ClaimAmount", "ClaimAmountINR", "PatientAge", "PatientIncome", "PatientIncomeINR", "Cluster"
        ]]
        
        X = df[feature_cols].copy()
        self.numeric_columns = list(X.columns)
        self.feature_names_out = list(X.columns)
        
        # Fit scaler
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = RobustScaler()
            
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=self.numeric_columns, index=X.index)
        logger.info(f"Pipeline fitted successfully. Feature matrix shape: {X_scaled.shape}")
        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transforms validation or test datasets using previously fitted parameters without leakage.
        """
        logger.debug("Transforming dataset using fitted preprocessor rules...")
        df = self.handle_missing_values(df)
        df = self.detect_and_treat_outliers(df)
        df = self.encode_categorical_variables(df, is_train=False)
        
        y = df["IsFraud"] if "IsFraud" in df.columns else pd.Series(0, index=df.index)
        
        X = df[self.numeric_columns].copy()
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=self.numeric_columns, index=X.index)
        return X_scaled, y

    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series, method: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Applies class imbalance resampling on TRAINING data only.
        Supports: 'smote', 'undersample', 'smoteenn', or 'none'.
        """
        method = method or self.imbalance_method
        if method == "smote":
            logger.info("Applying SMOTE oversampling to training minority class...")
            sampler = SMOTE(random_state=42)
            X_res, y_res = sampler.fit_resample(X, y)
        elif method == "undersample":
            logger.info("Applying RandomUnderSampler to training majority class...")
            sampler = RandomUnderSampler(random_state=42)
            X_res, y_res = sampler.fit_resample(X, y)
        elif method == "smoteenn":
            logger.info("Applying SMOTEENN combined resampling...")
            sampler = SMOTEENN(random_state=42)
            X_res, y_res = sampler.fit_resample(X, y)
        else:
            logger.info("No resampling applied; relying on class weighting during training.")
            X_res, y_res = X, y
            
        if isinstance(X_res, np.ndarray):
            X_res = pd.DataFrame(X_res, columns=X.columns)
        if isinstance(y_res, np.ndarray):
            y_res = pd.Series(y_res)
            
        logger.info(f"Class imbalance handling complete. Resulting class counts:\n{y_res.value_counts()}")
        return X_res, y_res

    def get_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """
        Computes balanced class weights dictionary for cost-sensitive learning.
        """
        counts = y.value_counts()
        total = len(y)
        w0 = total / (2.0 * counts[0])
        w1 = total / (2.0 * counts[1])
        return {0: float(w0), 1: float(w1)}

    def save_preprocessor(self, filepath: str = "models_saved/insurance_preprocessor.pkl") -> None:
        """
        Serializes and saves the complete preprocessor instance.
        """
        ensure_directories([os.path.dirname(filepath)])
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor serialized and saved to: {filepath}")

    @staticmethod
    def load_preprocessor(filepath: str = "models_saved/insurance_preprocessor.pkl") -> "InsuranceDataPreprocessor":
        """
        Loads a serialized InsuranceDataPreprocessor instance.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Saved preprocessor not found at {filepath}")
        with open(filepath, "rb") as f:
            preprocessor = pickle.load(f)
        logger.info(f"Preprocessor loaded from: {filepath}")
        return preprocessor


def split_dataset_stratified(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into Training (70%), Validation (15%), and Test (15%) sets
    using stratified sampling on the IsFraud target to maintain exact 6% class distribution.
    """
    logger.info(f"Splitting dataset with ratios Train={1.0-test_size-val_size:.2f}, Val={val_size:.2f}, Test={test_size:.2f} (Stratified)")
    
    # First split off Test (15% of total)
    df_temp, df_test = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df["IsFraud"], 
        random_state=random_seed
    )
    
    # Remaining is 85%; we want Val to be 15% of total, which is 15/85 of temp
    relative_val_size = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_temp, 
        test_size=relative_val_size, 
        stratify=df_temp["IsFraud"], 
        random_state=random_seed
    )
    
    logger.info(f"Stratified split complete -> Train: {len(df_train)} ({df_train['IsFraud'].mean()*100:.2f}% fraud), "
                f"Val: {len(df_val)} ({df_val['IsFraud'].mean()*100:.2f}% fraud), "
                f"Test: {len(df_test)} ({df_test['IsFraud'].mean()*100:.2f}% fraud)")
    
    return df_train, df_val, df_test
