"""Leakage-aware feature selection and sklearn preprocessing assembly."""
from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
# High-cardinality hospital identity is deliberately excluded; only aggregate provider context may be added in a leakage-audited extension.
LEAKAGE={'is_fraud','fraud_type','claim_id','claimant_id','claim_date','hospital_id'}
def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create point-in-time ratios and remove identifiers/label metadata. Args: claims. Returns: X."""
    x=df.drop(columns=[c for c in LEAKAGE if c in df]).copy()
    x['claim_to_si_ratio']=x.total_claimed_amount_inr/x.sum_insured_inr
    x['premium_to_claim_ratio']=x.annual_premium_inr/x.total_claimed_amount_inr
    x['per_day_cost']=x.total_claimed_amount_inr/x.length_of_stay_days.clip(lower=1)
    return x
def transformer(x: pd.DataFrame) -> ColumnTransformer:
    """Build train-fitted mixed-type transformer. Args: X. Returns: transformer."""
    num=x.select_dtypes(include='number').columns.tolist(); cat=[c for c in x.columns if c not in num]
    return ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',RobustScaler())]),num),('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
