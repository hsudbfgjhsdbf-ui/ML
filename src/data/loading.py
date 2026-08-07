"""Load and validate the supplied medical-insurance claims workbook.

The repository contains one workbook rather than the multi-table Medicare
benchmark described in the planning documents. This module treats the
workbook as the authoritative supplied snapshot and records its limitations
instead of silently pretending it is a larger public claims source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_COLUMNS = [
    "ClaimID",
    "PatientID",
    "ProviderID",
    "ClaimAmount",
    "ClaimDate",
    "DiagnosisCode",
    "ProcedureCode",
    "PatientAge",
    "PatientGender",
    "ProviderSpecialty",
    "ClaimStatus",
    "PatientIncome",
    "PatientMaritalStatus",
    "PatientEmploymentStatus",
    "ProviderLocation",
    "ClaimType",
    "ClaimSubmissionMethod",
    "Cluster",
    "ClaimLegitimacy",
]


@dataclass(frozen=True)
class DatasetProfile:
    """Compact data-quality profile emitted before any modeling occurs.

    Args:
        rows: Number of records.
        columns: Number of columns.
        fraud_count: Number of positive-class records.
        legitimate_count: Number of negative-class records.
        duplicate_rows: Exact duplicate count.
        duplicate_claim_ids: Duplicate primary-key count.
        missing_cells: Total missing cells.
    """

    rows: int
    columns: int
    fraud_count: int
    legitimate_count: int
    duplicate_rows: int
    duplicate_claim_ids: int
    missing_cells: int

    @property
    def fraud_rate(self) -> float:
        """Return positive-class prevalence."""
        return self.fraud_count / self.rows if self.rows else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "rows": self.rows,
            "columns": self.columns,
            "fraud_count": self.fraud_count,
            "legitimate_count": self.legitimate_count,
            "fraud_rate": self.fraud_rate,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_claim_ids": self.duplicate_claim_ids,
            "missing_cells": self.missing_cells,
        }


def load_claims(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    """Read the supplied Excel workbook and normalize basic cell types.

    Args:
        path: Workbook path.
        max_rows: Optional deterministic row cap for development smoke runs.
    Returns:
        Claims dataframe preserving source column names.
    Raises:
        FileNotFoundError: If the workbook is absent.
        ValueError: If the workbook has no rows or an unexpected schema.
        RuntimeError: If pandas cannot parse the workbook.
    """
    if not path.exists():
        raise FileNotFoundError(f"Claims workbook not found: {path}")
    try:
        frame = pd.read_excel(path, engine="openpyxl")
    except Exception as exc:  # pragma: no cover - depends on malformed external file
        raise RuntimeError(f"Could not parse Excel workbook {path}: {exc}") from exc
    if max_rows is not None:
        if max_rows < 100:
            raise ValueError("max_rows must be at least 100 for a meaningful split")
        frame = frame.iloc[:max_rows].copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    validate_schema(frame)
    frame["ClaimDate"] = pd.to_datetime(frame["ClaimDate"], errors="coerce")
    if frame["ClaimDate"].isna().any():
        raise ValueError("ClaimDate contains values that cannot be parsed as dates")
    frame["ClaimLegitimacy"] = frame["ClaimLegitimacy"].astype(str).str.strip().str.title()
    frame["ClaimLegitimacy"] = frame["ClaimLegitimacy"].replace({"Fraudulent": "Fraud"})
    frame["PatientGender"] = frame["PatientGender"].astype(str).str.strip().str.upper()
    return frame.reset_index(drop=True)


def validate_schema(frame: pd.DataFrame) -> None:
    """Validate required columns, identifiers, target semantics, and basic ranges.

    Args:
        frame: Raw or normalized claims dataframe.
    Returns:
        None.
    Raises:
        ValueError: For a failed data-quality gate.
    """
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.empty:
        raise ValueError("The claims dataframe is empty")
    for key in ("ClaimID", "PatientID", "ProviderID"):
        if frame[key].isna().any() or (frame[key].astype(str).str.strip() == "").any():
            raise ValueError(f"Identifier column {key} contains null or blank values")
    if frame["ClaimID"].duplicated().any():
        raise ValueError("ClaimID must be unique; duplicate claims would invalidate evaluation")
    if frame["ClaimAmount"].isna().any() or (frame["ClaimAmount"] < 0).any():
        raise ValueError("ClaimAmount must be non-null and non-negative")
    if frame["PatientAge"].isna().any() or (~frame["PatientAge"].between(0, 120)).any():
        raise ValueError("PatientAge must lie between 0 and 120 years")
    labels = set(frame["ClaimLegitimacy"].dropna().astype(str).str.title())
    if not labels.issubset({"Fraud", "Legitimate"}):
        raise ValueError(f"Unexpected target labels: {sorted(labels)}")
    if len(labels) < 2:
        raise ValueError("Both Fraud and Legitimate classes are required")


def profile_dataset(frame: pd.DataFrame) -> DatasetProfile:
    """Compute a concise profile after schema validation.

    Args:
        frame: Validated claims dataframe.
    Returns:
        DatasetProfile with counts used in reports and gates.
    """
    return DatasetProfile(
        rows=len(frame),
        columns=len(frame.columns),
        fraud_count=int((frame["ClaimLegitimacy"] == "Fraud").sum()),
        legitimate_count=int((frame["ClaimLegitimacy"] == "Legitimate").sum()),
        duplicate_rows=int(frame.duplicated().sum()),
        duplicate_claim_ids=int(frame["ClaimID"].duplicated().sum()),
        missing_cells=int(frame.isna().sum().sum()),
    )


def missingness_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing counts and percentages for the data card.

    Args:
        frame: Claims dataframe.
    Returns:
        Table with one row per source feature.
    """
    return (
        pd.DataFrame(
            {
                "column": frame.columns,
                "missing_count": frame.isna().sum().values,
                "missing_percent": (frame.isna().mean().values * 100.0),
                "dtype": [str(dtype) for dtype in frame.dtypes],
                "unique_values": [int(frame[column].nunique(dropna=True)) for column in frame.columns],
            }
        )
        .sort_values(["missing_percent", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )
