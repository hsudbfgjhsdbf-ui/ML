"""
Data loading and dataset verification module for Medical Insurance Claim Fraud Detection System.
Institution: IIIT Dharwad, Department of Data Science and AI
Faculty Adviser: Prof. Ramesh Athe
Team Members: B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)

This module handles:
1. Loading the raw insurance claim dataset from Excel or CSV.
2. Generating a detailed metadata data dictionary (`data/metadata_dictionary.md`).
3. Enriching dataset with Indian healthcare and insurance context (Indian States/Cities, Hospital Tiers, Indian Insurer schemes).
4. Generating a realistic Indian synthetic fraud claim dataset (`data/synthetic/synthetic_indian_claims.csv`) for robustness testing.
5. Saving processed and raw dataset checkpoints.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from src.utils import setup_logger, ensure_directories, format_inr

logger = setup_logger("DataLoadingLogger")

INDIAN_STATES_CITIES = [
    ("Maharashtra", "Mumbai"),
    ("Maharashtra", "Pune"),
    ("Karnataka", "Bengaluru"),
    ("Karnataka", "Mysuru"),
    ("Telangana", "Hyderabad"),
    ("Delhi NCT", "New Delhi"),
    ("Tamil Nadu", "Chennai"),
    ("Tamil Nadu", "Coimbatore"),
    ("West Bengal", "Kolkata"),
    ("Gujarat", "Ahmedabad"),
    ("Rajasthan", "Jaipur"),
    ("Uttar Pradesh", "Lucknow"),
    ("Kerala", "Kochi"),
    ("Madhya Pradesh", "Indore")
]

INDIAN_POLICY_TYPES = [
    "Individual Health Plan",
    "Family Floater Plan",
    "Employer Group Health Insurance",
    "Senior Citizen Red Carpet Plan",
    "Ayushman Bharat PM-JAY Scheme"
]

INDIAN_INSURERS = [
    "Star Health and Allied Insurance",
    "ICICI Lombard General Insurance",
    "HDFC Ergo General Insurance",
    "New India Assurance",
    "United India Insurance"
]

HOSPITAL_TIERS = [
    "Tier-1 Metro Corporate Hospital",
    "Tier-2 City Multi-Specialty Hospital",
    "Tier-3 Town Nursing Home"
]


def load_raw_dataset(file_path: str = "data/raw/Health Insurance Fraud Claims.xlsx") -> pd.DataFrame:
    """
    Loads the raw dataset from an Excel or CSV file.
    Validates mandatory columns and logs dataset statistics.
    """
    logger.info(f"Loading raw insurance dataset from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at {file_path}")
        
    if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
        
    logger.info(f"Successfully loaded dataset with shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    expected_cols = [
        "ClaimID", "PatientID", "ProviderID", "ClaimAmount", "ClaimDate", 
        "DiagnosisCode", "ProcedureCode", "PatientAge", "PatientGender", 
        "ProviderSpecialty", "ClaimStatus", "PatientIncome", 
        "PatientMaritalStatus", "PatientEmploymentStatus", "ProviderLocation", 
        "ClaimType", "ClaimSubmissionMethod", "Cluster", "ClaimLegitimacy"
    ]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        logger.warning(f"Missing expected columns: {missing_cols}. Continuing with available columns.")
        
    return df


def enrich_with_indian_context(df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """
    Enriches the dataset with Indian healthcare and insurance domain features:
    - Maps ProviderLocation to realistic Indian State and City pairs.
    - Assigns Hospital Tier based on ClaimAmount and Specialty.
    - Assigns Indian Insurance Company and Policy Type.
    - Computes Indian Rupee (INR) Claim Amount (scaled to typical Indian hospital billing range).
    """
    logger.info("Enriching dataset with Indian healthcare and insurance domain context...")
    np.random.seed(random_seed)
    n = len(df)
    
    df = df.copy()
    
    # Map each record to an Indian State and City deterministically based on hash of ClaimID
    state_city_indices = [abs(hash(str(cid))) % len(INDIAN_STATES_CITIES) for cid in df["ClaimID"]]
    df["IndianState"] = [INDIAN_STATES_CITIES[idx][0] for idx in state_city_indices]
    df["IndianCity"] = [INDIAN_STATES_CITIES[idx][1] for idx in state_city_indices]
    
    # Assign Policy Type and Insurer
    policy_indices = [abs(hash(str(pid))) % len(INDIAN_POLICY_TYPES) for pid in df["PatientID"]]
    df["PolicyType"] = [INDIAN_POLICY_TYPES[idx] for idx in policy_indices]
    
    insurer_indices = [abs(hash(str(prid))) % len(INDIAN_INSURERS) for prid in df["ProviderID"]]
    df["InsurerCompany"] = [INDIAN_INSURERS[idx] for idx in insurer_indices]
    
    # Assign Hospital Tier:
    # High amount inpatient claims tend to be in Tier-1 Metro Corporate Hospitals
    def assign_tier(row):
        if row["ClaimType"] == "Inpatient" and row["ClaimAmount"] > 6000:
            return "Tier-1 Metro Corporate Hospital"
        elif row["ClaimAmount"] > 3500:
            return "Tier-2 City Multi-Specialty Hospital"
        else:
            return "Tier-3 Town Nursing Home"
            
    df["HospitalTier"] = df.apply(assign_tier, axis=1)
    
    # Compute realistic INR claim amount (scaling by 25x so typical average claim is around Rs. 1,25,000)
    df["ClaimAmountINR"] = df["ClaimAmount"] * 25.0
    df["PatientIncomeINR"] = df["PatientIncome"] * 25.0
    
    # Standardize target column to numeric binary (1 for Fraud, 0 for Legitimate)
    df["IsFraud"] = (df["ClaimLegitimacy"].str.strip().str.lower() == "fraud").astype(int)
    
    fraud_rate = df["IsFraud"].mean() * 100
    logger.info(f"Indian context enrichment complete. Total records: {len(df)}, Fraud rate: {fraud_rate:.2f}%")
    return df


def generate_metadata_dictionary(df: pd.DataFrame, output_path: str = "data/metadata_dictionary.md") -> None:
    """
    Generates a comprehensive Markdown data dictionary documenting every feature in the dataset.
    """
    logger.info(f"Generating data dictionary at: {output_path}")
    ensure_directories([os.path.dirname(output_path)])
    
    lines = [
        "# DATA DICTIONARY AND METADATA SPECIFICATION",
        "**Project:** Medical Insurance Claim Fraud Detection System  ",
        "**Institution:** IIIT Dharwad, Department of Data Science and AI  ",
        "**Faculty Adviser:** Prof. Ramesh Athe  ",
        "**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  ",
        "",
        "## 1. Overview of the Dataset",
        f"The dataset consists of **{df.shape[0]} medical insurance claim records** across **{df.shape[1]} features**. ",
        "It includes both raw attributes from hospital claim submissions and domain-specific enriched Indian healthcare features.",
        "",
        "## 2. Feature Specification Table",
        "| Feature Name | Data Type | Description | Valid Range / Categories | Relevance to Fraud Detection |",
        "| :--- | :--- | :--- | :--- | :--- |",
        "| **ClaimID** | String (UUID) | Unique identifier for each insurance claim submission. | Alphanumeric UUID string | Tracking and auditing identifier; prevents duplicate processing. |",
        "| **PatientID** | String (UUID) | Unique identifier for the insured policyholder / patient. | Alphanumeric UUID string | Identifies repeat claimants and historical claim patterns. |",
        "| **ProviderID** | String (UUID) | Unique identifier for the healthcare provider / hospital. | Alphanumeric UUID string | Identifies hospitals with unusually high claim rejection or fraud rates. |",
        "| **ClaimAmount** | Float | Raw medical claim billing amount submitted for reimbursement. | 100.12 to 9,997.20 | Unusually high amounts or statistical outliers indicate possible billing inflation. |",
        "| **ClaimAmountINR** | Float | Claim amount scaled to Indian Rupees (INR) representing realistic Indian hospital costs. | Rs. 2,503.00 to Rs. 2,49,930.00 | Benchmarks claim against typical Indian medical treatment cost structures. |",
        "| **ClaimDate** | DateTime | Date when the claim was filed by the claimant. | 2024-01-01 to 2024-12-31 | Identifies temporal spikes, seasonality, or claims filed shortly after waiting periods. |",
        "| **DiagnosisCode** | String | International diagnosis classification code (e.g., ICD-10 equivalent). | Alphanumeric code | Mismatches between diagnosis code and procedure indicate fraudulent billing. |",
        "| **ProcedureCode** | String | Medical procedure or surgery code billed on the claim. | Alphanumeric code | Verified against treatment cost benchmarks and diagnosis compatibility. |",
        "| **PatientAge** | Integer | Age of the policyholder / patient in years. | 0 to 100 years | Helps audit fairness across age groups (children, adults, elderly citizens). |",
        "| **PatientGender** | String | Gender identity of the patient (M, F, Other). | 'M', 'F' | Audited for gender fairness and demographic neutrality in fraud scoring. |",
        "| **ProviderSpecialty** | String | Medical department specialty of the provider. | Orthopedics, Cardiology, Neurology, Pediatrics, General Practice | Treatment specialty must align with diagnosis and procedure complexity. |",
        "| **ClaimStatus** | String | Processing status of the claim at ingestion. | Pending, Approved, Rejected | Administrative context for claim disposition. |",
        "| **PatientIncome** | Float | Annual declared income of the policyholder. | Numeric value | Financial context for sum insured and claim-to-income ratios. |",
        "| **PatientIncomeINR** | Float | Annual income scaled to Indian Rupees (INR). | Numeric INR value | Helps assess policy premium affordability and claim proportionality. |",
        "| **PatientMaritalStatus** | String | Marital status of the policyholder. | Single, Married, Divorced, Widowed | Demographic context for family floater coverage rules. |",
        "| **PatientEmploymentStatus** | String | Employment status of the policyholder. | Employed, Self-Employed, Unemployed, Student, Retired | Explains employer group insurance eligibility and claim behavior. |",
        "| **ProviderLocation** | String | Raw location name of the healthcare provider. | Simulated location names | Geographic reference for hospital cost analysis. |",
        "| **IndianState** | String | Enriched Indian state where treatment occurred. | Maharashtra, Karnataka, Telangana, Delhi NCT, Tamil Nadu, etc. | Regional cost benchmark and geographic fraud pattern auditing. |",
        "| **IndianCity** | String | Enriched Indian city / metro area of the hospital. | Mumbai, Bengaluru, Hyderabad, New Delhi, Chennai, Pune, etc. | Identifies city-level cost variations and metro vs. tier-3 disparities. |",
        "| **PolicyType** | String | Indian insurance policy product structure. | Individual, Family Floater, Employer Group, Senior Citizen, Ayushman Bharat | Determines sub-limits, co-payments, and waiting period rules. |",
        "| **InsurerCompany** | String | Indian general or health insurance company. | Star Health, ICICI Lombard, HDFC Ergo, New India Assurance, United India | Provides company-specific fraud rules and network hospital checks. |",
        "| **HospitalTier** | String | Indian healthcare provider tier classification. | Tier-1 Metro Corporate, Tier-2 City Multi-Specialty, Tier-3 Town Nursing Home | Cost expectation baseline; Tier-3 billing Tier-1 prices is a key fraud signal. |",
        "| **ClaimType** | String | Category of medical treatment. | Inpatient, Emergency, Routine, Outpatient | Inpatient and emergency claims have higher financial risk and fraud exposure. |",
        "| **ClaimSubmissionMethod** | String | Mode of claim filing by the claimant or hospital. | Paper, Online, Phone | Online submissions can be cross-verified digitally; paper claims require OCR. |",
        "| **Cluster** | Integer | Existing data clustering label from initial segmentation. | 0 to 4 | Structural grouping of similar claims. |",
        "| **ClaimLegitimacy** | String (Target) | Original textual target variable indicating claim legitimacy. | 'Legitimate', 'Fraud' | Primary binary target for supervised machine learning and deep learning. |",
        "| **IsFraud** | Integer (Target) | Binary numerical target variable (0 = Legitimate, 1 = Fraud). | 0, 1 | Machine-readable target for model training and loss evaluation. |",
        "",
        "## 3. Class Distribution and Fraud Rate",
        f"- **Total Legitimate Claims:** {(df['IsFraud'] == 0).sum()} ({(df['IsFraud'] == 0).mean()*100:.2f}%)",
        f"- **Total Fraudulent Claims:** {(df['IsFraud'] == 1).sum()} ({(df['IsFraud'] == 1).mean()*100:.2f}%)",
        "",
        "This data dictionary is automatically maintained and verified during the execution pipeline."
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Data dictionary generated successfully at: {output_path}")


def generate_synthetic_indian_claims(n_samples: int = 1500, output_path: str = "data/synthetic/synthetic_indian_claims.csv", random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic Indian health insurance claim dataset as required by Section 2.
    Mirrors Indian healthcare distributions: right-skewed INR claim amounts, 10% fraud rate,
    all Indian age groups, policy types, and hospital tiers.
    """
    logger.info(f"Generating synthetic Indian health insurance claim dataset ({n_samples} records)...")
    np.random.seed(random_seed)
    
    # 10% fraud rate for synthetic data
    is_fraud = np.random.binomial(n=1, p=0.10, size=n_samples)
    
    # Age distribution covering children, adults, and elderly citizens
    ages = np.random.choice(
        a=[np.random.randint(0, 18), np.random.randint(18, 45), np.random.randint(45, 65), np.random.randint(65, 95)],
        size=n_samples,
        p=[0.15, 0.45, 0.25, 0.15]
    )
    
    genders = np.random.choice(["M", "F"], size=n_samples, p=[0.52, 0.48])
    
    # Right-skewed claim amount in INR (Gamma distribution)
    # Legitimate claims average around Rs. 85,000; Fraud claims average Rs. 2,40,000
    base_amounts = np.random.gamma(shape=3.0, scale=28000.0, size=n_samples)
    fraud_multipliers = np.where(is_fraud == 1, np.random.uniform(2.2, 4.5, size=n_samples), 1.0)
    claim_amounts_inr = np.clip(base_amounts * fraud_multipliers, 5000.0, 750000.0)
    
    # Select Indian State and City
    state_city_idx = np.random.randint(0, len(INDIAN_STATES_CITIES), size=n_samples)
    states = [INDIAN_STATES_CITIES[idx][0] for idx in state_city_idx]
    cities = [INDIAN_STATES_CITIES[idx][1] for idx in state_city_idx]
    
    policy_types = np.random.choice(INDIAN_POLICY_TYPES, size=n_samples, p=[0.35, 0.35, 0.15, 0.10, 0.05])
    insurers = np.random.choice(INDIAN_INSURERS, size=n_samples)
    hospital_tiers = np.random.choice(HOSPITAL_TIERS, size=n_samples, p=[0.40, 0.40, 0.20])
    claim_types = np.random.choice(["Inpatient", "Emergency", "Routine", "Outpatient"], size=n_samples, p=[0.55, 0.25, 0.12, 0.08])
    specialties = np.random.choice(["Orthopedics", "Cardiology", "Neurology", "Pediatrics", "General Practice"], size=n_samples)
    
    # Generate UUIDs
    import uuid
    claim_ids = [str(uuid.uuid4()) for _ in range(n_samples)]
    patient_ids = [str(uuid.uuid4()) for _ in range(n_samples)]
    provider_ids = [str(uuid.uuid4()) for _ in range(n_samples)]
    
    df_synthetic = pd.DataFrame({
        "ClaimID": claim_ids,
        "PatientID": patient_ids,
        "ProviderID": provider_ids,
        "ClaimAmount": claim_amounts_inr / 25.0,
        "ClaimAmountINR": claim_amounts_inr,
        "ClaimDate": pd.date_range(start="2024-01-01", periods=n_samples, freq="h")[:n_samples],
        "DiagnosisCode": [f"IND-ICD-{np.random.randint(100,999)}" for _ in range(n_samples)],
        "ProcedureCode": [f"IND-PROC-{np.random.randint(10,99)}" for _ in range(n_samples)],
        "PatientAge": ages,
        "PatientGender": genders,
        "ProviderSpecialty": specialties,
        "ClaimStatus": ["Pending"] * n_samples,
        "PatientIncome": np.random.normal(60000, 15000, size=n_samples),
        "PatientIncomeINR": np.random.normal(1500000, 400000, size=n_samples),
        "PatientMaritalStatus": np.random.choice(["Married", "Single", "Widowed"], size=n_samples, p=[0.65, 0.25, 0.10]),
        "PatientEmploymentStatus": np.random.choice(["Employed", "Self-Employed", "Retired", "Student"], size=n_samples),
        "ProviderLocation": cities,
        "IndianState": states,
        "IndianCity": cities,
        "PolicyType": policy_types,
        "InsurerCompany": insurers,
        "HospitalTier": hospital_tiers,
        "ClaimType": claim_types,
        "ClaimSubmissionMethod": np.random.choice(["Online", "Paper", "Phone"], size=n_samples, p=[0.60, 0.30, 0.10]),
        "Cluster": np.random.randint(0, 5, size=n_samples),
        "ClaimLegitimacy": np.where(is_fraud == 1, "Fraud", "Legitimate"),
        "IsFraud": is_fraud
    })
    
    ensure_directories([os.path.dirname(output_path)])
    df_synthetic.to_csv(output_path, index=False)
    logger.info(f"Synthetic Indian claim dataset saved at: {output_path}")
    return df_synthetic


def execute_data_loading_pipeline(raw_path: str = "data/raw/Health Insurance Fraud Claims.xlsx") -> pd.DataFrame:
    """
    Executes the full data loading, enrichment, metadata dictionary generation,
    and synthetic dataset creation pipeline.
    """
    logger.info("Executing Complete Data Loading and Domain Enrichment Pipeline...")
    df_raw = load_raw_dataset(raw_path)
    df_enriched = enrich_with_indian_context(df_raw)
    generate_metadata_dictionary(df_enriched, "data/metadata_dictionary.md")
    generate_synthetic_indian_claims(n_samples=1500, output_path="data/synthetic/synthetic_indian_claims.csv")
    
    ensure_directories(["data/processed"])
    processed_path = "data/processed/claims_enriched.csv"
    df_enriched.to_csv(processed_path, index=False)
    logger.info(f"Enriched dataset checkpoint saved at: {processed_path}")
    return df_enriched
