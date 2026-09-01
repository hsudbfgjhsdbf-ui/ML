"""
Data Preprocessing, Cleaning, Imputation, Encoding, and Scaling Pipeline.
Provides modular, reproducible, serializable transformers for training, validation, and test splits.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, LabelEncoder
)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN

from src.config import config, RANDOM_SEED
from src.utils import logger

class MedicalClaimPreprocessor(BaseEstimator, TransformerMixin):
    """
    Production-grade scikit-learn compatible preprocessor for medical insurance claims.
    Encapsulates imputation, outlier treatment, categorical encoding, and scaling.
    """
    
    def __init__(
        self,
        scaling_strategy: str = "standard", # 'standard', 'minmax', 'robust'
        encoding_strategy: str = "onehot",  # 'onehot', 'ordinal', 'target'
        handle_outliers: bool = True,
        outlier_method: str = "iqr",        # 'iqr', 'zscore'
        iqr_multiplier: float = 3.0         # Mild capping to preserve legitimate fraud signals
    ):
        self.scaling_strategy = scaling_strategy
        self.encoding_strategy = encoding_strategy
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        self.iqr_multiplier = iqr_multiplier
        
        # State learned during fit
        self.numeric_features_: List[str] = []
        self.categorical_features_: List[str] = []
        self.feature_names_out_: List[str] = []
        
        # Imputers
        self.num_imputer_: Optional[SimpleImputer] = None
        self.cat_imputer_: Optional[SimpleImputer] = None
        
        # Encoders & Scalers
        self.scaler_: Any = None
        self.encoder_: Any = None
        self.ordinal_encoders_: Dict[str, OrdinalEncoder] = {}
        self.target_means_: Dict[str, Dict[Any, float]] = {}
        self.global_target_mean_: float = 0.10
        
        # Outlier bounds
        self.outlier_bounds_: Dict[str, Tuple[float, float]] = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits all statistical parameters on training data strictly to avoid leakage."""
        df = X.copy()
        
        # Filter metadata / ID columns
        cols_to_drop = [c for c in df.columns if "ID" in c or "Date" in c or c in ["Fraud_Pattern_Type", "Is_Fraud", "ClaimLegitimacy", "ClaimLegitimacy_Binary"]]
        usable_cols = [c for c in df.columns if c not in cols_to_drop]
        
        self.numeric_features_ = [c for c in usable_cols if pd.api.types.is_numeric_dtype(df[c])]
        self.categorical_features_ = [c for c in usable_cols if not pd.api.types.is_numeric_dtype(df[c])]
        
        logger.debug(f"Fitting preprocessor with {len(self.numeric_features_)} numeric and {len(self.categorical_features_)} categorical features.")
        
        # 1. Fit Imputers
        if self.numeric_features_:
            self.num_imputer_ = SimpleImputer(strategy="median")
            self.num_imputer_.fit(df[self.numeric_features_])
            
        if self.categorical_features_:
            self.cat_imputer_ = SimpleImputer(strategy="most_frequent")
            self.cat_imputer_.fit(df[self.categorical_features_])
            
        # 2. Fit Outlier Bounds (only on continuous currency/duration fields)
        if self.handle_outliers:
            for col in self.numeric_features_:
                vals = df[col].dropna()
                if self.outlier_method == "iqr":
                    q25 = vals.quantile(0.25)
                    q75 = vals.quantile(0.75)
                    iqr = q75 - q25
                    lower = max(0.0, q25 - self.iqr_multiplier * iqr)
                    upper = q75 + self.iqr_multiplier * iqr
                else: # zscore
                    mu = vals.mean()
                    std = vals.std() + 1e-8
                    lower = max(0.0, mu - 4.0 * std)
                    upper = mu + 4.0 * std
                self.outlier_bounds_[col] = (lower, upper)
                
        # 3. Fit Encoders
        if self.encoding_strategy == "onehot" and self.categorical_features_:
            self.encoder_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            cat_imputed = self.cat_imputer_.transform(df[self.categorical_features_])
            self.encoder_.fit(cat_imputed)
        elif self.encoding_strategy == "target" and self.categorical_features_ and y is not None:
            self.global_target_mean_ = float(y.mean())
            for col in self.categorical_features_:
                means = y.groupby(df[col]).mean().to_dict()
                self.target_means_[col] = means
                
        # 4. Fit Scaler
        if self.scaling_strategy == "standard":
            self.scaler_ = StandardScaler()
        elif self.scaling_strategy == "minmax":
            self.scaler_ = MinMaxScaler()
        elif self.scaling_strategy == "robust":
            self.scaler_ = RobustScaler()
        else:
            self.scaler_ = StandardScaler()
            
        if self.numeric_features_:
            num_imputed = self.num_imputer_.transform(df[self.numeric_features_])
            self.scaler_.fit(num_imputed)
            
        return self
        
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Applies learned transformations to input dataframe."""
        df = X.copy()
        
        # Drop identifiers
        cols_to_drop = [c for c in df.columns if "ID" in c or "Date" in c or c in ["Fraud_Pattern_Type", "Is_Fraud", "ClaimLegitimacy", "ClaimLegitimacy_Binary"]]
        
        # Transform numeric
        if self.numeric_features_:
            num_data = self.num_imputer_.transform(df[self.numeric_features_])
            num_df = pd.DataFrame(num_data, columns=self.numeric_features_, index=df.index)
            
            # Apply outlier treatment
            if self.handle_outliers:
                for col, (lower, upper) in self.outlier_bounds_.items():
                    if col in num_df.columns:
                        num_df[col] = num_df[col].clip(lower, upper)
                        
            num_scaled = self.scaler_.transform(num_df)
        else:
            num_scaled = np.empty((len(df), 0))
            
        # Transform categorical
        if self.categorical_features_:
            cat_data = self.cat_imputer_.transform(df[self.categorical_features_])
            if self.encoding_strategy == "onehot":
                cat_encoded = self.encoder_.transform(cat_data)
            elif self.encoding_strategy == "target":
                cat_encoded_list = []
                for i, col in enumerate(self.categorical_features_):
                    series = pd.Series(cat_data[:, i], index=df.index)
                    mapping = self.target_means_.get(col, {})
                    encoded = series.map(mapping).fillna(self.global_target_mean_).values.reshape(-1, 1)
                    cat_encoded_list.append(encoded)
                cat_encoded = np.hstack(cat_encoded_list)
            else: # default ordinal
                cat_encoded = cat_data.astype(str)
        else:
            cat_encoded = np.empty((len(df), 0))
            
        X_out = np.hstack([num_scaled, cat_encoded])
        return X_out

def apply_resampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str = "smote", # 'none', 'smote', 'borderline', 'adasyn', 'smoteenn', 'undersample'
    random_state: int = RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies sophisticated synthetic oversampling or hybrid resampling on training data.
    """
    if strategy == "none" or strategy is None:
        return X_train, y_train
        
    logger.info(f"Applying class imbalance mitigation strategy: {strategy.upper()} on training data (Initial class count: {np.bincount(y_train)})")
    
    if strategy == "smote":
        sampler = SMOTE(random_state=random_state, sampling_strategy="auto", k_neighbors=5)
    elif strategy == "borderline":
        sampler = BorderlineSMOTE(random_state=random_state, sampling_strategy="auto", k_neighbors=5)
    elif strategy == "adasyn":
        sampler = ADASYN(random_state=random_state, sampling_strategy="auto", n_neighbors=5)
    elif strategy == "smoteenn":
        sampler = SMOTEENN(random_state=random_state, smote=SMOTE(random_state=random_state, k_neighbors=5))
    elif strategy == "undersample":
        sampler = RandomUnderSampler(random_state=random_state, sampling_strategy=0.5)
    else:
        sampler = SMOTE(random_state=random_state)
        
    try:
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        logger.info(f"Resampling complete. New shape: {X_res.shape}, Class distribution: {np.bincount(y_res)}")
        return X_res, y_res
    except Exception as e:
        logger.warning(f"Resampling failed with error: {e}. Falling back to standard SMOTE.")
        fallback = SMOTE(random_state=random_state, k_neighbors=3)
        return fallback.fit_resample(X_train, y_train)
