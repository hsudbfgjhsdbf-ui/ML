"""Leakage-aware feature engineering and preprocessing transformers."""

from .engineering import FeatureEngineeringResult, engineer_features
from .preprocessing import MatrixBundle, SplitData, encode_target, fit_transform_matrices, stratified_three_way_split

__all__ = [
    "FeatureEngineeringResult",
    "MatrixBundle",
    "SplitData",
    "encode_target",
    "engineer_features",
    "fit_transform_matrices",
    "stratified_three_way_split",
]
