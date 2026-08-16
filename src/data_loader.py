"""
Data Acquisition, Synthesis, and Ingestion Engine.
Loads the benchmark Excel dataset, generates high-fidelity Indian-context synthetic claims,
and provides stratified 70-15-15 train/val/test splits.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split

from src.config import (
    config, RANDOM_SEED, INDIAN_STATES, INDIAN_CITIES,
    HOSPITAL_TIERS, INSURANCE_PROVIDERS, POLICY_TYPES,
    DIAGNOSIS_CATEGORIES
)
from src.utils import logger

def load_raw_excel(file_path: Path = config.raw_data_path) -> pd.DataFrame:
    """Loads and standardizes the benchmark Excel dataset."""
    if not file_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {file_path}")
    
    logger.info(f"Loading raw dataset from {file_path}")
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    
    # Standardize column naming and target variable
    df["ClaimLegitimacy_Binary"] = df["ClaimLegitimacy"].apply(
        lambda x: 1 if str(x).strip().lower() == "fraud" else 0
    )
    logger.info(f"Loaded raw dataset with shape: {df.shape}, Fraud cases: {df['ClaimLegitimacy_Binary'].sum()} ({df['ClaimLegitimacy_Binary'].mean()*100:.2f}%)")
    return df

def generate_indian_synthetic_dataset(
    n_samples: int = 12000,
    fraud_rate: float = 0.105,
    random_state: int = RANDOM_SEED,
    output_path: Path = config.synthetic_data_path
) -> pd.DataFrame:
    """
    Generates a rich, domain-grounded synthetic medical insurance claims dataset
    contextualized for the Indian healthcare ecosystem.
    """
    np.random.seed(random_state)
    logger.info(f"Generating Indian-context synthetic claims dataset ({n_samples} records, target fraud rate: {fraud_rate*100:.1f}%)")
    
    # Determine fraud labels (binary: 1 = Fraud, 0 = Legitimate)
    is_fraud = np.random.binomial(1, fraud_rate, size=n_samples)
    
    # Demographics
    ages = np.random.randint(18, 85, size=n_samples)
    genders = np.random.choice(["Male", "Female", "Other"], size=n_samples, p=[0.51, 0.48, 0.01])
    
    # States and Cities
    states = np.random.choice(list(INDIAN_CITIES.keys()), size=n_samples)
    cities = [np.random.choice(INDIAN_CITIES[s]) for s in states]
    
    # Incomes in INR (Annual)
    # Right-skewed distribution from Rs 2.5 LPA to Rs 45 LPA
    incomes = np.round(np.random.lognormal(mean=13.0, sigma=0.65, size=n_samples), -2)
    incomes = np.clip(incomes, 180000, 5000000)
    
    # Policy Configurations
    policy_types = np.random.choice(POLICY_TYPES, size=n_samples, p=[0.30, 0.35, 0.15, 0.08, 0.07, 0.03, 0.02])
    insurers = np.random.choice(INSURANCE_PROVIDERS, size=n_samples)
    
    # Sum Insured (INR)
    sum_insured_options = np.array([200000, 300000, 500000, 750000, 1000000, 1500000, 2500000, 5000000])
    sums_insured = np.random.choice(sum_insured_options, size=n_samples, p=[0.15, 0.20, 0.30, 0.15, 0.10, 0.05, 0.03, 0.02])
    
    # Premiums (approx 2% - 5% of sum insured adjusted for age)
    annual_premiums = np.round(sums_insured * (0.025 + (ages / 85.0) * 0.025), -2)
    
    # Policy duration (months) and waiting period (months)
    policy_duration_months = np.random.randint(1, 120, size=n_samples)
    waiting_period_months = np.random.choice([1, 24, 36, 48], size=n_samples, p=[0.2, 0.4, 0.3, 0.1])
    waiting_period_completed = (policy_duration_months >= waiting_period_months).astype(int)
    copay_percentage = np.random.choice([0, 5, 10, 15, 20, 25], size=n_samples, p=[0.45, 0.20, 0.15, 0.10, 0.07, 0.03])
    
    # Hospital Tiers & Providers
    tier_names = list(HOSPITAL_TIERS.keys())
    hospital_tier_assignments = np.random.choice(tier_names, size=n_samples, p=[0.35, 0.45, 0.20])
    hospitals = [
        np.random.choice(HOSPITAL_TIERS[t]["hospitals"]) for t in hospital_tier_assignments
    ]
    
    # Clinical Categories & Diagnoses
    diag_cats = list(DIAGNOSIS_CATEGORIES.keys())
    assigned_cats = np.random.choice(diag_cats, size=n_samples)
    
    diagnosis_codes = []
    treatment_names = []
    base_costs = []
    hospital_stays = []
    
    for i in range(n_samples):
        cat = assigned_cats[i]
        info = DIAGNOSIS_CATEGORIES[cat]
        code = np.random.choice(info["codes"])
        treat = np.random.choice(info["treatments"])
        c_min, c_max = info["typical_inr_cost"]
        s_min, s_max = info["typical_stay_days"]
        
        tier_mult = HOSPITAL_TIERS[hospital_tier_assignments[i]]["cost_multiplier"]
        
        # Base realistic cost
        base_c = np.random.uniform(c_min, c_max) * tier_mult
        stay_d = max(0, int(np.random.uniform(s_min, s_max)))
        
        diagnosis_codes.append(code)
        treatment_names.append(treat)
        base_costs.append(base_c)
        hospital_stays.append(stay_d)
        
    base_costs = np.array(base_costs)
    hospital_stays = np.array(hospital_stays)
    
    # Claim Amounts with Fraud Patterns Injected
    claim_amounts = np.zeros(n_samples)
    fraud_pattern_types = []
    
    # Historical policyholder metrics
    prior_claims_count = np.random.poisson(lam=0.8, size=n_samples)
    total_prior_amount = np.zeros(n_samples)
    rejected_prior_claims = np.zeros(n_samples)
    
    for i in range(n_samples):
        if is_fraud[i] == 1:
            # Randomly select a dominant fraud typology
            ftype = np.random.choice([
                "Inflated_Billing",
                "Phantom_Hospitalization",
                "Upcoding_Procedure",
                "Waiting_Period_Exploit",
                "Repeat_Claim_Burst",
                "Collusive_Provider"
            ], p=[0.35, 0.15, 0.20, 0.15, 0.10, 0.05])
            fraud_pattern_types.append(ftype)
            
            if ftype == "Inflated_Billing":
                # Claim amount 2.2x to 4.5x normal
                inflation = np.random.uniform(2.2, 4.5)
                claim_amounts[i] = min(sums_insured[i] * 0.95, base_costs[i] * inflation)
                hospital_stays[i] = max(1, hospital_stays[i] + np.random.randint(2, 6))
            elif ftype == "Phantom_Hospitalization":
                # High cost claimed, 0 or 1 day actual needed stay, filed right after waiting period
                claim_amounts[i] = np.random.uniform(150000, min(sums_insured[i], 450000))
                policy_duration_months[i] = max(1, waiting_period_months[i] + np.random.randint(1, 3))
            elif ftype == "Upcoding_Procedure":
                # Billing for Tier 1 mega surgery in Tier 3 nursing home
                claim_amounts[i] = base_costs[i] * np.random.uniform(2.5, 3.8)
            elif ftype == "Waiting_Period_Exploit":
                # Claim filed within 15-45 days of policy purchase for pre-existing disease
                policy_duration_months[i] = np.random.randint(1, 2)
                claim_amounts[i] = base_costs[i] * np.random.uniform(1.2, 1.8)
            elif ftype == "Repeat_Claim_Burst":
                prior_claims_count[i] = np.random.randint(3, 7)
                rejected_prior_claims[i] = np.random.randint(1, 3)
                total_prior_amount[i] = np.random.uniform(200000, 600000)
                claim_amounts[i] = base_costs[i] * np.random.uniform(1.3, 2.2)
            elif ftype == "Collusive_Provider":
                claim_amounts[i] = base_costs[i] * np.random.uniform(1.8, 3.0)
                rejected_prior_claims[i] = np.random.randint(2, 4)
        else:
            fraud_pattern_types.append("None_Legitimate")
            # Legitimate variation +/- 15%
            noise = np.random.normal(1.0, 0.08)
            claim_amounts[i] = min(sums_insured[i], max(5000, base_costs[i] * noise))
            if prior_claims_count[i] > 0:
                total_prior_amount[i] = prior_claims_count[i] * np.random.uniform(25000, 90000)
                rejected_prior_claims[i] = 1 if np.random.rand() < 0.05 else 0

    claim_amounts = np.round(claim_amounts, 2)
    
    # Claim Submission Methods
    submission_methods = np.random.choice(["TPA_Cashless", "Reimbursement_Paper", "Digital_Portal", "Agent_Assisted"], size=n_samples, p=[0.45, 0.25, 0.20, 0.10])
    claim_types = np.random.choice(["Hospitalization", "DayCare_Procedure", "Outpatient_Surgery", "Pre_Post_Hospitalization"], size=n_samples, p=[0.60, 0.25, 0.10, 0.05])
    
    # Claim Dates over past 3 years
    start_date = pd.to_datetime("2023-01-01")
    date_offsets = np.random.randint(0, 1000, size=n_samples)
    claim_dates = start_date + pd.to_timedelta(date_offsets, unit="D")
    
    df_synthetic = pd.DataFrame({
        "Claim_ID": [f"CLM-IND-{i+10001}" for i in range(n_samples)],
        "Patient_ID": [f"PAT-{np.random.randint(10000, 99999)}" for _ in range(n_samples)],
        "Patient_Age": ages,
        "Patient_Gender": genders,
        "Patient_State": states,
        "Patient_City": cities,
        "Annual_Income_INR": incomes,
        "Insurance_Provider": insurers,
        "Policy_Type": policy_types,
        "Sum_Insured_INR": sums_insured,
        "Annual_Premium_INR": annual_premiums,
        "Policy_Duration_Months": policy_duration_months,
        "Waiting_Period_Months": waiting_period_months,
        "Waiting_Period_Completed": waiting_period_completed,
        "Copay_Percentage": copay_percentage,
        "Hospital_Name": hospitals,
        "Hospital_Tier": hospital_tier_assignments,
        "Diagnosis_Category": assigned_cats,
        "ICD10_Diagnosis_Code": diagnosis_codes,
        "Treatment_Name": treatment_names,
        "Hospitalization_Duration_Days": hospital_stays,
        "Claim_Type": claim_types,
        "Claim_Submission_Method": submission_methods,
        "Claim_Date": claim_dates,
        "Claim_Amount_INR": claim_amounts,
        "Prior_Claims_Count": prior_claims_count,
        "Total_Prior_Claimed_INR": total_prior_amount,
        "Rejected_Prior_Claims": rejected_prior_claims,
        "Fraud_Pattern_Type": fraud_pattern_types,
        "Is_Fraud": is_fraud
    })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_synthetic.to_csv(output_path, index=False)
    logger.info(f"Synthetic dataset successfully written to {output_path} with {len(df_synthetic)} rows and {len(df_synthetic.columns)} columns.")
    return df_synthetic

def get_unified_dataset() -> pd.DataFrame:
    """
    Returns the unified Indian health insurance dataset combining the Excel benchmark
    and the domain synthetic generator.
    """
    if not config.synthetic_data_path.exists():
        generate_indian_synthetic_dataset()
    
    df = pd.read_csv(config.synthetic_data_path)
    df["Claim_Date"] = pd.to_datetime(df["Claim_Date"])
    return df

def create_stratified_splits(
    df: pd.DataFrame,
    target_col: str = "Is_Fraud",
    train_size: float = config.train_split,
    val_size: float = config.val_split,
    test_size: float = config.test_split,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into stratified Train (70%), Validation (15%), and Test (15%) partitions.
    """
    logger.info(f"Splitting dataset of {len(df)} rows into Train ({train_size*100:.0f}%), Val ({val_size*100:.0f}%), Test ({test_size*100:.0f}%)")
    
    # First split: Train vs Temp (Val + Test)
    temp_size = val_size + test_size
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        stratify=df[target_col],
        random_state=random_state
    )
    
    # Second split: Val vs Test
    val_relative_ratio = val_size / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_relative_ratio),
        stratify=temp_df[target_col],
        random_state=random_state
    )
    
    logger.info(
        f"Split complete. Train: {len(train_df)} ({train_df[target_col].mean()*100:.2f}% fraud), "
        f"Val: {len(val_df)} ({val_df[target_col].mean()*100:.2f}% fraud), "
        f"Test: {len(test_df)} ({test_df[target_col].mean()*100:.2f}% fraud)"
    )
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
