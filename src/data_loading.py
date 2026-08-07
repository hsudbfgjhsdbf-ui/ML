"""Data provenance checks and a reproducible Indian-context synthetic claim generator.

The bundled workbook is preserved untouched as a reference source. It is inadequate
for the stated academic study because it has only 4,500 claims and generic, non-Indian
features. This module therefore creates a transparent, explicitly synthetic Indian
population; it does *not* claim to represent records from an actual insurer.
"""
from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import write_json

RAW_EXPECTED_COLUMNS = [
    "ClaimID", "PatientID", "ProviderID", "ClaimAmount", "ClaimDate", "DiagnosisCode",
    "ProcedureCode", "PatientAge", "PatientGender", "ProviderSpecialty", "ClaimStatus",
    "PatientIncome", "PatientMaritalStatus", "PatientEmploymentStatus", "ProviderLocation",
    "ClaimType", "ClaimSubmissionMethod", "Cluster", "ClaimLegitimacy",
]

STATES = {
    "Karnataka": [("Bengaluru", 1.55), ("Dharwad", 1.00), ("Mysuru", 1.14)],
    "Maharashtra": [("Mumbai", 1.70), ("Pune", 1.35), ("Nagpur", 1.05)],
    "Delhi": [("New Delhi", 1.72)],
    "Tamil Nadu": [("Chennai", 1.42), ("Coimbatore", 1.13), ("Madurai", 0.95)],
    "Telangana": [("Hyderabad", 1.38), ("Warangal", 0.93)],
    "Kerala": [("Kochi", 1.25), ("Thiruvananthapuram", 1.19)],
    "Gujarat": [("Ahmedabad", 1.18), ("Surat", 1.08)],
    "West Bengal": [("Kolkata", 1.23), ("Siliguri", 0.92)],
    "Uttar Pradesh": [("Lucknow", 0.94), ("Noida", 1.36), ("Varanasi", 0.78)],
    "Rajasthan": [("Jaipur", 0.98), ("Kota", 0.77)],
    "Bihar": [("Patna", 0.75)],
    "Odisha": [("Bhubaneswar", 0.86)],
}

TREATMENTS: dict[str, dict[str, Any]] = {
    "Cardiac hospitalization": {"base": 180000, "stay": 5.2, "practice": "Allopathic", "diagnosis": "I25.1", "procedure": "CABG/angioplasty"},
    "Orthopaedic surgery": {"base": 125000, "stay": 4.3, "practice": "Allopathic", "diagnosis": "M17.0", "procedure": "Knee arthroscopy"},
    "Maternity delivery": {"base": 70000, "stay": 3.1, "practice": "Allopathic", "diagnosis": "O80", "procedure": "Normal/C-section delivery"},
    "Dengue hospitalization": {"base": 51000, "stay": 4.0, "practice": "Allopathic", "diagnosis": "A90", "procedure": "Supportive inpatient care"},
    "Cataract day-care": {"base": 39000, "stay": 1.0, "practice": "Allopathic", "diagnosis": "H25.9", "procedure": "Phacoemulsification"},
    "Dialysis day-care": {"base": 26000, "stay": 1.0, "practice": "Allopathic", "diagnosis": "N18.6", "procedure": "Haemodialysis"},
    "Appendectomy": {"base": 62000, "stay": 2.8, "practice": "Allopathic", "diagnosis": "K35.8", "procedure": "Laparoscopic appendectomy"},
    "Respiratory outpatient": {"base": 7200, "stay": 0.0, "practice": "Allopathic", "diagnosis": "J18.9", "procedure": "Consultation and medicines"},
    "Ayurvedic panchakarma": {"base": 18000, "stay": 5.0, "practice": "Ayurvedic", "diagnosis": "M54.5", "procedure": "Panchakarma therapy"},
    "Ayurvedic outpatient": {"base": 4200, "stay": 0.0, "practice": "Ayurvedic", "diagnosis": "R53", "procedure": "Ayurvedic consultation"},
}


def sha256_file(path: str | Path) -> str:
    """Calculate a stable SHA-256 digest for a source file.

    Args:
        path: Existing file path.

    Returns:
        Lower-case hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_raw_workbook(path: str | Path) -> dict[str, Any]:
    """Profile the supplied source workbook without changing it.

    Args:
        path: Bundled Excel workbook path.

    Returns:
        Data-provenance and adequacy assessment suitable for JSON reporting.

    Raises:
        FileNotFoundError: If the workbook is missing.
        ValueError: If no readable rows or target label are present.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Bundled source workbook is missing: {source}")
    frame = pd.read_excel(source)
    if frame.empty or "ClaimLegitimacy" not in frame.columns:
        raise ValueError("Raw workbook must contain records and a ClaimLegitimacy target column.")
    missing_required = sorted(set(RAW_EXPECTED_COLUMNS) - set(frame.columns))
    adequate = len(frame) >= 10_000 and not missing_required
    return {
        "source_file": str(source),
        "sha256": sha256_file(source),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "column_names": list(frame.columns),
        "target_distribution": frame["ClaimLegitimacy"].value_counts(dropna=False).to_dict(),
        "missingness_fraction": {k: round(float(v), 6) for k, v in frame.isna().mean().items()},
        "adequacy_criteria": {
            "minimum_records": False,
            "clear_binary_label": frame["ClaimLegitimacy"].nunique(dropna=True) == 2,
            "indian_context": False,
            "policy_temporal_history_features": False,
        },
        "adequate_as_primary_training_data": adequate,
        "decision": "Preserve as raw reference and use transparent synthetic Indian fallback.",
        "limitations": [
            "Contains 4,500 records, below the required 10,000-record threshold.",
            "Locations and identifiers are generic rather than Indian-context provider/policy data.",
            "Does not provide policy limits, waiting periods, co-pay, or claim-history features.",
            "Claim currency and documentation conventions are not identified as Indian.",
        ],
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Compute a numerically stable logistic transform for synthetic fraud probabilities."""
    clipped = np.clip(values, -30, 30)
    return 1.0 / (1.0 + np.exp(-clipped))


def generate_indian_synthetic_claims(n_claims: int, seed: int) -> pd.DataFrame:
    """Generate a labelled synthetic Indian medical-insurance claim population.

    The generation process intentionally exposes the assumptions behind the label. It
    models legitimate regional price variation, policy structures and plausible fraud
    indicators. It must only be used for teaching, benchmarking and software testing.

    Args:
        n_claims: Number of unique claims before deliberate duplicate-quality tests.
        seed: Reproducibility seed.

    Returns:
        DataFrame containing ``n_claims`` unique claims plus 0.5% exact duplicates.

    Raises:
        ValueError: If fewer than 1,000 claims are requested.
    """
    if n_claims < 1_000:
        raise ValueError("At least 1,000 claims are required for a meaningful synthetic study.")
    rng = np.random.default_rng(seed)
    n_people, n_providers = max(4_500, n_claims // 2), 180
    state_names = np.array(list(STATES))
    state_prob = np.array([0.16, 0.13, 0.07, 0.12, 0.09, 0.07, 0.08, 0.07, 0.09, 0.06, 0.03, 0.03])
    state_prob = state_prob / state_prob.sum()

    # Policyholder master data persists across their several synthetic claims.
    person_state = rng.choice(state_names, size=n_people, p=state_prob)
    city_lookup, multiplier_lookup = {}, {}
    for state, cities in STATES.items():
        for city, multiplier in cities:
            city_lookup.setdefault(state, []).append(city)
            multiplier_lookup[city] = multiplier
    person_city = np.array([rng.choice(city_lookup[state]) for state in person_state])
    person_age = np.clip(rng.normal(41, 17, n_people).round().astype(int), 0, 90)
    person_gender = rng.choice(["Female", "Male", "Non-binary/Prefer not to say"], n_people, p=[0.47, 0.51, 0.02])
    income_bracket = rng.choice(["Low", "Lower-middle", "Middle", "Upper-middle", "High"], n_people, p=[0.17, .28, .31, .18, .06])
    income_map = {"Low": 21000, "Lower-middle": 43000, "Middle": 78000, "Upper-middle": 145000, "High": 285000}
    person_income = np.array([income_map[x] for x in income_bracket]) * rng.lognormal(0, .18, n_people)
    occupation = rng.choice(["Salaried", "Self-employed", "Student", "Retired", "Homemaker", "Daily-wage", "Agricultural"], n_people, p=[.36,.17,.12,.12,.10,.07,.06])
    disability = rng.choice(["No disclosed disability", "Mobility", "Visual", "Hearing", "Other"], n_people, p=[.91,.04,.015,.01,.025])
    policy_type = rng.choice(["Individual", "Family floater", "Employer group", "Ayushman Bharat", "ECHS"], n_people, p=[.29,.34,.20,.12,.05])
    insurer = rng.choice(["Star Health", "ICICI Lombard", "HDFC ERGO", "New India Assurance", "Government scheme"], n_people, p=[.25,.22,.20,.18,.15])
    sum_insured = rng.choice([300000, 500000, 750000, 1000000, 1500000, 2500000], n_people, p=[.12,.24,.16,.23,.15,.10])
    premium = np.maximum(2500, sum_insured * rng.uniform(.012, .048, n_people) * (1 + np.maximum(person_age - 45, 0) * .006))
    policy_start = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1460, n_people), unit="D")
    wait_days = np.where(person_age >= 60, 730, rng.choice([180, 365, 730], n_people, p=[.22,.55,.23]))
    copay = np.where(policy_type == "Ayushman Bharat", 0, rng.choice([0, 10, 20, 30], n_people, p=[.30,.35,.27,.08]))
    base_risk = rng.beta(1.4, 8.5, n_people)

    # Provider data has tier/region/risk properties but no direct label field.
    provider_state = rng.choice(state_names, n_providers, p=state_prob)
    provider_city = np.array([rng.choice(city_lookup[s]) for s in provider_state])
    provider_tier = rng.choice(["Government", "Nursing home", "Tier-2 private", "Corporate", "AYUSH centre"], n_providers, p=[.16,.19,.31,.23,.11])
    tier_multiplier = {"Government": .70, "Nursing home": .82, "Tier-2 private": 1.0, "Corporate": 1.55, "AYUSH centre": .72}
    provider_risk = rng.beta(1.7, 8.0, n_providers)
    provider_names = np.array([f"{provider_city[i]} {provider_tier[i].replace(' ', '-')} Hospital {i+1:03d}" for i in range(n_providers)])
    network_probability = np.where(np.isin(provider_tier, ["Government", "Corporate"]), .76, .58)

    patient_idx = rng.integers(0, n_people, n_claims)
    provider_idx = rng.integers(0, n_providers, n_claims)
    treatment_names = np.array(list(TREATMENTS))
    treatment = rng.choice(treatment_names, n_claims, p=[.10,.11,.13,.11,.10,.08,.09,.11,.08,.09])
    treatment_info = [TREATMENTS[x] for x in treatment]
    practice = np.array([x["practice"] for x in treatment_info])
    base_cost = np.array([x["base"] for x in treatment_info], dtype=float)
    diagnosis = np.array([x["diagnosis"] for x in treatment_info])
    procedure = np.array([x["procedure"] for x in treatment_info])
    claim_type = np.where(np.isin(treatment, ["Cataract day-care", "Dialysis day-care"]), "Day-care", np.where(np.isin(treatment, ["Respiratory outpatient", "Ayurvedic outpatient"]), "Outpatient", "Hospitalization"))
    claim_type = np.where(rng.random(n_claims) < .035, "Pre-authorization", claim_type)
    admission = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 731, n_claims), unit="D")
    policy_days = (admission - pd.DatetimeIndex(policy_start[patient_idx])).days.astype(int)
    waiting_completed = policy_days >= wait_days[patient_idx]
    stay_base = np.array([x["stay"] for x in treatment_info])
    stay = np.maximum(0, np.round(rng.normal(stay_base, np.maximum(.5, stay_base*.28)))).astype(int)
    stay[claim_type == "Day-care"] = 1
    stay[claim_type == "Outpatient"] = 0
    regional_multiplier = np.array([multiplier_lookup[c] for c in provider_city[provider_idx]])
    provider_multiplier = np.array([tier_multiplier[x] for x in provider_tier[provider_idx]])
    amount = base_cost * regional_multiplier * provider_multiplier * rng.lognormal(0, .30, n_claims)
    # Calculate proxy historical patterns before label assignment.
    history_count = rng.poisson(1.4 + 2.7 * base_risk[patient_idx])
    past_12m = np.minimum(history_count, rng.poisson(0.75 + 2.0 * base_risk[patient_idx]))
    last_claim_days = np.maximum(1, rng.gamma(2.4, 120, n_claims).astype(int) - 28 * past_12m)
    historical_avg = np.maximum(2000, amount * rng.lognormal(-.15, .45, n_claims))
    historical_std = historical_avg * rng.uniform(.12, .65, n_claims)
    distance = np.abs(rng.normal(38, 44, n_claims)) + (person_state[patient_idx] != provider_state[provider_idx]) * rng.uniform(150, 900, n_claims)
    procedures = np.where(np.isin(claim_type, ["Hospitalization", "Day-care"]), rng.integers(1, 5, n_claims), 1)
    doctor_credential = rng.choice(["MBBS", "MD/MS", "DNB", "BAMS", "Visiting consultant", "Registration unavailable"], n_claims, p=[.29,.32,.12,.14,.10,.03])
    network_hospital = rng.random(n_claims) < network_probability[provider_idx]
    # Latent probability is intentionally multi-factor and has unobserved noise.
    expected = base_cost * regional_multiplier * provider_multiplier
    amount_deviation = (amount - expected) / np.maximum(expected * .36, 1)
    risk_logit = (
        -3.65 + 1.55 * np.maximum(amount_deviation, 0) + 1.20 * (policy_days < 60)
        + .90 * (~waiting_completed) + .48 * (past_12m >= 3) + .62 * (distance > 250)
        + .78 * provider_risk[provider_idx] + .44 * base_risk[patient_idx]
        + .42 * (doctor_credential == "Registration unavailable") + rng.normal(0, .58, n_claims)
    )
    fraud_probability = _sigmoid(risk_logit)
    is_fraud = rng.random(n_claims) < fraud_probability
    # Inject plausible suspicious claims after label draw. Signals remain noisy, not deterministic.
    amount[is_fraud] *= rng.uniform(1.30, 3.50, is_fraud.sum())
    procedures[is_fraud] += rng.binomial(2, .48, is_fraud.sum())
    doctor_credential[np.where(is_fraud & (rng.random(n_claims) < .28))[0]] = "Registration unavailable"
    past_12m[is_fraud] += rng.binomial(3, .46, is_fraud.sum())
    rejected_history = rng.binomial(np.maximum(history_count, 1), np.clip(.06 + 1.9 * base_risk[patient_idx], .06, .72))
    rejected_history += rng.binomial(2, .40, n_claims) * is_fraud
    claim_to_premium = amount / np.maximum(premium[patient_idx], 1)
    duration_denominator = np.maximum(stay, 1)
    current_vs_history = amount / np.maximum(historical_avg, 1)
    seasonal = admission.month.map({1:"Winter",2:"Winter",3:"Summer",4:"Summer",5:"Summer",6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Monsoon",10:"Post-monsoon",11:"Winter",12:"Winter"}).to_numpy()
    gst_rate = np.where(practice == "Ayurvedic", 0.05, 0.00)
    gst_amount = amount * rng.uniform(.03, .11, n_claims) * (claim_type != "Hospitalization")
    # The output keeps raw component features alongside engineered-feature ingredients.
    frame = pd.DataFrame({
        "claim_id": [f"CLM-{202400000+i:09d}" for i in range(n_claims)],
        "policyholder_id": [f"PH-{i:05d}" for i in patient_idx],
        "provider_id": [f"PR-{i:04d}" for i in provider_idx],
        "claim_date": admission.strftime("%Y-%m-%d"),
        "state": person_state[patient_idx], "city": person_city[patient_idx], "age": person_age[patient_idx],
        "gender": person_gender[patient_idx], "income_bracket": income_bracket[patient_idx],
        "monthly_income_inr": person_income[patient_idx].round(2), "occupation_type": occupation[patient_idx],
        "disability_accommodation": disability[patient_idx], "policy_type": policy_type[patient_idx],
        "insurer": insurer[patient_idx], "sum_insured_inr": sum_insured[patient_idx],
        "annual_premium_inr": premium[patient_idx].round(2), "policy_start_date": pd.DatetimeIndex(policy_start[patient_idx]).strftime("%Y-%m-%d"),
        "policy_duration_days": policy_days, "waiting_period_days": wait_days[patient_idx],
        "waiting_period_completed": np.where(waiting_completed, "Yes", "No"), "copay_percent": copay[patient_idx],
        "claim_amount_inr": amount.round(2), "claim_type": claim_type, "treatment_type": treatment,
        "medical_practice": practice, "diagnosis_code": diagnosis, "procedure_code": procedure,
        "hospitalization_days": stay, "procedure_count": procedures, "doctor_credential": doctor_credential,
        "hospital_name": provider_names[provider_idx], "hospital_tier": provider_tier[provider_idx],
        "hospital_state": provider_state[provider_idx], "network_hospital": np.where(network_hospital, "Yes", "No"),
        "distance_to_hospital_km": distance.round(2), "time_since_last_claim_days": last_claim_days,
        "claims_past_12_months": past_12m, "total_historical_claims": history_count,
        "historical_claimed_amount_inr": (historical_avg * np.maximum(history_count, 1)).round(2),
        "historical_average_claim_inr": historical_avg.round(2), "historical_claim_std_inr": historical_std.round(2),
        "historical_max_claim_inr": (historical_avg + 1.5*historical_std).round(2),
        "rejected_claim_count": rejected_history, "provider_rejection_rate": provider_risk[provider_idx].round(4),
        "provider_average_claim_inr": (base_cost * regional_multiplier * provider_multiplier * 1.08).round(2),
        "provider_unique_patient_count": rng.integers(25, 1200, n_claims), "regional_treatment_baseline_inr": expected.round(2),
        "season": seasonal, "claim_submission_method": rng.choice(["Cashless", "Reimbursement", "TPA portal", "Branch"], n_claims, p=[.43,.30,.20,.07]),
        "gst_amount_inr": gst_amount.round(2), "document_completeness_score": np.clip(rng.normal(.91 - .13*is_fraud, .07), .42, 1).round(3),
        "is_fraud": is_fraud.astype(int), "claim_legitimacy": np.where(is_fraud, "Fraudulent", "Legitimate"),
    })
    # Test missing-data code paths without making critical fields unusable.
    for column, rate in {"occupation_type": .012, "monthly_income_inr": .018, "doctor_credential": .008}.items():
        frame.loc[rng.random(n_claims) < rate, column] = np.nan
    duplicate_n = max(1, round(n_claims * .005))
    duplicates = frame.sample(duplicate_n, random_state=seed).copy()
    return pd.concat([frame, duplicates], ignore_index=True)


def load_or_create_study_data(config: dict[str, Any], regenerate: bool = False) -> pd.DataFrame:
    """Load existing synthetic data or create the documented Indian-context fallback.

    Args:
        config: Loaded configuration dictionary.
        regenerate: If true, replace the synthetic CSV using the fixed seed.

    Returns:
        Synthetic study DataFrame before duplicate cleaning.
    """
    output = Path(config["paths"]["synthetic_data"])
    if regenerate or not output.exists():
        frame = generate_indian_synthetic_claims(
            int(config["synthetic_data"]["n_claims"]), int(config["project"]["random_seed"])
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        write_json(output.with_suffix(".metadata.json"), {
            "data_type": "synthetic educational data", "generator": "src.data_loading.generate_indian_synthetic_claims",
            "seed": config["project"]["random_seed"], "rows_before_cleaning": len(frame),
            "intentional_exact_duplicate_rows": int(frame.duplicated().sum()),
            "fraud_rate_before_cleaning": float(frame["is_fraud"].mean()),
            "warning": "This data is not an observed insurer dataset and must not be used for real claim decisions.",
        })
    return pd.read_csv(output)
