"""Preprocessing pipeline builder."""
from typing import Tuple, List
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def get_scaler(name: str):
    name = (name or "standard").lower()
    if name == "standard":
        return StandardScaler()
    elif name == "robust":
        return RobustScaler()
    elif name == "minmax":
        return MinMaxScaler()
    else:
        return "passthrough"

def get_encoder(name: str):
    name = (name or "onehot").lower()
    if name == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif name == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

def build_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
    date_features: List[str],
    config: dict
):
    """Build sklearn ColumnTransformer.
    
    Date features are transformed into derived numerical features before pipeline.
    """
    prep_cfg = config.get("preprocessing", {}) if config else {}
    scaling = prep_cfg.get("scaling", "standard")
    encoding = prep_cfg.get("encoding", "onehot")

    transformers = []
    if numerical_features:
        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", get_scaler(scaling))
        ])
        transformers.append(("num", num_pipe, numerical_features))
    if categorical_features:
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", get_encoder(encoding))
        ])
        transformers.append(("cat", cat_pipe, categorical_features))

    # Date features will be dropped here; they should be engineered beforehand
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    return preprocessor

def engineer_date_features(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    """Convert ClaimDate into useful numeric features."""
    df = df.copy()
    for col in date_cols:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"] = dt.dt.year
            df[f"{col}_month"] = dt.dt.month
            df[f"{col}_day"] = dt.dt.day
            df[f"{col}_dayofweek"] = dt.dt.dayofweek
            df[f"{col}_quarter"] = dt.dt.quarter
            # Days since epoch or relative
            df[f"{col}_ordinal"] = dt.map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
            # Drop original to avoid leakage in pipeline if not handled
            # Keep original? We'll drop original date col for tabular modeling
            df = df.drop(columns=[col])
    return df

def outlier_analysis(df: pd.DataFrame, numerical_features: List[str]) -> dict:
    """IQR-based outlier count."""
    result = {}
    for col in numerical_features:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        outliers = ((s < lower) | (s > upper)).sum()
        result[col] = {"q1": float(q1), "q3": float(q3), "iqr": float(iqr), "outlier_count": int(outliers), "outlier_pct": float(outliers/len(s))}
    return result
