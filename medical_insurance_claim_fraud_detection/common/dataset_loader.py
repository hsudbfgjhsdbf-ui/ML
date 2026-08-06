"""Dataset loading with validation for public medical fraud datasets."""
from pathlib import Path
from typing import Tuple, Optional, Dict
import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "ClaimID", "PatientID", "ProviderID", "ClaimAmount", "ClaimDate",
    "DiagnosisCode", "ProcedureCode", "PatientAge", "PatientGender",
    "ProviderSpecialty", "ClaimStatus", "PatientIncome", "PatientMaritalStatus",
    "PatientEmploymentStatus", "ProviderLocation", "ClaimType",
    "ClaimSubmissionMethod", "Cluster", "ClaimLegitimacy"
]

def find_dataset_file(preferred_paths: list[Path]) -> Optional[Path]:
    for p in preferred_paths:
        if p.exists():
            return p
    return None

def load_claims_dataset(raw_path: str | Path) -> pd.DataFrame:
    """Load claims dataset from Excel or CSV."""
    raw_path = Path(raw_path)
    if not raw_path.exists():
        # try alternatives
        alternatives = [
            Path("data/raw/Health_Insurance_Fraud_Claims.xlsx"),
            Path("data/raw/health_insurance_fraud_claims.xlsx"),
            Path("../Health Insurance Fraud Claims.xlsx"),
            Path("../../Health Insurance Fraud Claims.xlsx"),
            Path("/home/user/ML/Health Insurance Fraud Claims.xlsx"),
            Path("Health Insurance Fraud Claims.xlsx"),
        ]
        found = find_dataset_file(alternatives)
        if found:
            raw_path = found
            logger.info(f"Using alternative dataset path: {raw_path}")
        else:
            raise FileNotFoundError(f"Dataset not found at {raw_path}. Searched alternatives.")
    if raw_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(raw_path)
    elif raw_path.suffix.lower() == ".csv":
        df = pd.read_csv(raw_path)
    else:
        raise ValueError(f"Unsupported file extension: {raw_path.suffix}")
    logger.info(f"Loaded dataset with shape {df.shape} from {raw_path}")
    return df

def validate_schema(df: pd.DataFrame) -> Dict:
    """Basic schema validation."""
    report = {}
    report["rows"] = len(df)
    report["cols"] = len(df.columns)
    report["columns"] = list(df.columns)
    missing = df.isna().sum().to_dict()
    report["missing"] = missing
    # check target
    if "ClaimLegitimacy" in df.columns:
        vc = df["ClaimLegitimacy"].value_counts().to_dict()
        report["target_distribution"] = vc
    else:
        report["target_distribution"] = {}
    # check expected columns
    missing_expected = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    report["missing_expected_columns"] = missing_expected
    return report

def preprocess_target(df: pd.DataFrame, target_col: str = "ClaimLegitimacy",
                      fraud_value: str = "Fraud", legit_value: str = "Legitimate") -> Tuple[pd.DataFrame, pd.Series]:
    """Convert target to binary 0/1."""
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found")
    mapping = {legit_value: 0, fraud_value: 1}
    # Handle case insensitive
    df = df.copy()
    df["_target_binary"] = df[target_col].map(mapping)
    # If mapping fails, try lower case
    if df["_target_binary"].isna().any():
        lower_map = {k.lower(): v for k, v in mapping.items()}
        df["_target_binary"] = df[target_col].astype(str).str.lower().map(lower_map)
    if df["_target_binary"].isna().any():
        raise ValueError(f"Unable to map target values: unique values {df[target_col].unique()}")
    y = df["_target_binary"].astype(int)
    X = df.drop(columns=[target_col, "_target_binary"])
    return X, y

def get_feature_types(df: pd.DataFrame, config: dict):
    """Extract feature type lists from config and data."""
    cfg_prep = config.get("preprocessing", {}) if config else {}
    num_feats = cfg_prep.get("numerical_features", ["ClaimAmount","PatientAge","PatientIncome"])
    cat_feats = cfg_prep.get("categorical_features", [])
    date_feats = cfg_prep.get("date_features", ["ClaimDate"])
    drop_feats = cfg_prep.get("drop_features", ["ClaimID","PatientID","ProviderID"])
    # Filter to existing
    num_feats = [c for c in num_feats if c in df.columns]
    cat_feats = [c for c in cat_feats if c in df.columns]
    date_feats = [c for c in date_feats if c in df.columns]
    drop_feats = [c for c in drop_feats if c in df.columns]
    return num_feats, cat_feats, date_feats, drop_feats

def save_processed(df: pd.DataFrame, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved processed data to {path} with shape {df.shape}")
