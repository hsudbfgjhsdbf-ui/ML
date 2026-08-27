"""Data loading and exploration for the insurance claim fraud dataset.

Loads the raw claims spreadsheet, performs an initial sanity check, and exposes
a helper that summarises the dataset (shape, dtypes, class balance, missing
values) so the numbers can be reported in the documentation and presentation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import ROOT, ensure_dir, logging, setup_logging, load_config

logger = setup_logging()


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw claims dataset from the Excel file.

    Args:
        path: Optional explicit path. Defaults to config-driven raw location.

    Returns:
        pd.DataFrame: The raw claims data.

    Raises:
        FileNotFoundError: If the raw file cannot be located.
    """
    if path is None:
        cfg = load_config()
        path = ROOT / cfg["data"]["raw_path"]
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    logger.info("Loading raw dataset from %s", path)
    df = pd.read_excel(path)
    logger.info("Loaded %d rows x %d columns", df.shape[0], df.shape[1])
    return df


def describe_dataset(df: pd.DataFrame) -> dict:
    """Produce a structured summary of the dataset.

    Args:
        df: The raw dataset.

    Returns:
        dict: Summary including shape, dtypes, missing values, class counts.
    """
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(df[c].isna().sum()) for c in df.columns},
        "duplicates": int(df.duplicated().sum()),
        "class_counts": df.get("ClaimLegitimacy", pd.Series()).value_counts().to_dict(),
    }
    return summary


def save_summary(df: pd.DataFrame, out_dir: str | Path) -> str:
    """Persist the dataset summary to a JSON metadata file.

    Args:
        df: The dataset.
        out_dir: Directory for processed data.

    Returns:
        str: Path of the saved summary file.
    """
    out_dir = ensure_dir(out_dir)
    summary = describe_dataset(df)
    import json

    path = Path(out_dir) / "dataset_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved dataset summary to %s", path)
    return str(path)


if __name__ == "__main__":
    raw = load_raw_data()
    print(raw.shape)
    print(describe_dataset(raw))
