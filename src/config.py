"""
Global Configuration Module for Medical Insurance Claim Fraud Detection System.
IIIT Dharwad - B.Tech Data Science & AI
Adviser: Ramesh Athe
Team: B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Root Directory Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
MODELS_DIR = BASE_DIR / "saved_models"
EVALUATION_DIR = BASE_DIR / "evaluation"
DOCUMENTATION_DIR = BASE_DIR / "documentation"
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"
PRESENTATION_DIR = BASE_DIR / "presentation"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure all critical directories exist
for d in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR,
    MODELS_DIR, EVALUATION_DIR, DOCUMENTATION_DIR, VISUALIZATIONS_DIR,
    PRESENTATION_DIR, REPORTS_DIR
]:
    d.mkdir(parents=True, exist_ok=True)

# Random Seed for Complete Reproducibility
RANDOM_SEED = 42

# Train / Validation / Test Split Configuration
TRAIN_SPLIT_RATIO = 0.70
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15

# Evaluation & Optimization Settings
PRIMARY_METRIC = "f2_score"  # Prioritizes recall to catch fraudulent claims
SECONDARY_METRIC = "roc_auc"
CV_FOLDS = 5
ALPHA_SIGNIFICANCE = 0.05

# Indian Context Domain Mappings
INDIAN_STATES = [
    "Maharashtra", "Karnataka", "Tamil Nadu", "Delhi NCR", "Telangana",
    "Gujarat", "Uttar Pradesh", "West Bengal", "Kerala", "Rajasthan",
    "Madhya Pradesh", "Punjab", "Haryana", "Bihar", "Odisha", "Andhra Pradesh"
]

INDIAN_CITIES = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik"],
    "Karnataka": ["Bengaluru", "Dharwad", "Hubballi", "Mysuru", "Mangaluru"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli"],
    "Delhi NCR": ["New Delhi", "Gurugram", "Noida", "Faridabad"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Noida"],
    "West Bengal": ["Kolkata", "Howrah", "Siliguri"],
    "Kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur"],
}

HOSPITAL_TIERS = {
    "Tier 1 (Metro Super-Specialty / Corporate)": {
        "cost_multiplier": 2.2,
        "hospitals": ["Apollo Hospitals", "Fortis Healthcare", "Max Super Specialty", "Manipal Hospital", "Medanta The Medicity"]
    },
    "Tier 2 (City Multispecialty / District HQ)": {
        "cost_multiplier": 1.3,
        "hospitals": ["KIMS Hospital", "SDM College of Medical Sciences Dharwad", "Sahyadri Hospitals", "Care Hospitals", "Sparsh Hospital"]
    },
    "Tier 3 (Community / Nursing Home / Rural Clinic)": {
        "cost_multiplier": 0.8,
        "hospitals": ["Lifeline Nursing Home", "City Care Clinic", "Sanjeevani Nursing Home", "Janata Hospital", "Shree Ram Clinic"]
    }
}

INSURANCE_PROVIDERS = [
    "Star Health and Allied Insurance",
    "ICICI Lombard General Insurance",
    "HDFC ERGO General Insurance",
    "New India Assurance",
    "National Insurance Company",
    "Bajaj Allianz General Insurance",
    "Care Health Insurance",
    "Niva Bupa Health Insurance",
    "Ayushman Bharat PM-JAY (Government)"
]

POLICY_TYPES = [
    "Individual Health Plan",
    "Family Floater Plan",
    "Group Health Insurance (Corporate)",
    "Senior Citizen Red Carpet Plan",
    "Ayushman Bharat PM-JAY (State Supported)",
    "Critical Illness Cover",
    "Top-up Health Insurance"
]

DIAGNOSIS_CATEGORIES = {
    "Cardiovascular": {
        "codes": ["I21.9", "I25.1", "I50.9", "I10"],
        "treatments": ["Coronary Angioplasty (PTCA)", "Coronary Artery Bypass Graft (CABG)", "Pacemaker Implantation", "Medical Management for Heart Failure"],
        "typical_inr_cost": (120000, 450000),
        "typical_stay_days": (3, 8)
    },
    "Orthopedics": {
        "codes": ["M17.9", "S72.0", "M51.2", "M23.2"],
        "treatments": ["Total Knee Replacement (TKR)", "Total Hip Replacement (THR)", "Lumbar Spine Decompression", "Arthroscopic Meniscectomy"],
        "typical_inr_cost": (150000, 380000),
        "typical_stay_days": (4, 7)
    },
    "Gastroenterology & General Surgery": {
        "codes": ["K35.8", "K80.2", "K40.9", "K25.9"],
        "treatments": ["Laparoscopic Appendectomy", "Laparoscopic Cholecystectomy", "Hernioplasty", "Endoscopic GI Bleed Management"],
        "typical_inr_cost": (45000, 160000),
        "typical_stay_days": (2, 4)
    },
    "Neurology & Neurosurgery": {
        "codes": ["I63.9", "G40.9", "S06.0", "G43.9"],
        "treatments": ["Stroke Thrombolysis / ICU Care", "Craniotomy for Evacuation", "Epilepsy Monitoring & Stabilization", "Conservative Neuro Care"],
        "typical_inr_cost": (80000, 400000),
        "typical_stay_days": (4, 12)
    },
    "Infectious Diseases & General Medicine": {
        "codes": ["A90", "A09", "J18.9", "B54"],
        "treatments": ["Dengue Fever with Thrombocytopenia Management", "Acute Gastroenteritis with Severe Dehydration", "Community Acquired Pneumonia IV Antibiotic Therapy", "Malaria Management"],
        "typical_inr_cost": (25000, 95000),
        "typical_stay_days": (2, 5)
    },
    "Oncology": {
        "codes": ["C50.9", "C34.9", "C18.9", "C61"],
        "treatments": ["Chemotherapy Cycle Daycare", "Modified Radical Mastectomy", "Radiation Therapy Course", "Targeted Immunotherapy Session"],
        "typical_inr_cost": (75000, 500000),
        "typical_stay_days": (1, 6)
    },
    "Nephrology & Urology": {
        "codes": ["N18.9", "N20.1", "N40.0", "N39.0"],
        "treatments": ["Hemodialysis Multi-Session Package", "Laser Lithotripsy (URSL)", "Transurethral Resection of Prostate (TURP)", "Pyelonephritis IV Antibiotic Therapy"],
        "typical_inr_cost": (35000, 180000),
        "typical_stay_days": (1, 5)
    },
    "Ophthalmology & Daycare": {
        "codes": ["H25.9", "H40.1", "H33.2"],
        "treatments": ["Phacoemulsification with Foldable IOL (Cataract)", "Vitrectomy for Retinal Detachment", "Trabeculectomy for Glaucoma"],
        "typical_inr_cost": (28000, 95000),
        "typical_stay_days": (0, 1)
    }
}

# Financial Cost Matrix Parameters (in Indian Rupees INR)
# Cost of False Negative (approving a fraudulent claim) = Average fraud payout
# Cost of False Positive (rejecting a genuine claim) = Administrative audit & goodwill friction
COST_FALSE_NEGATIVE_INR = 185000.0  # Average fraud loss per undetected case
COST_FALSE_POSITIVE_INR = 12000.0   # Friction, re-evaluation, customer escalation cost
COST_TRUE_POSITIVE_SAVING_INR = 185000.0
COST_TRUE_NEGATIVE_COST_INR = 0.0

@dataclass
class ProjectConfig:
    random_seed: int = RANDOM_SEED
    train_split: float = TRAIN_SPLIT_RATIO
    val_split: float = VAL_SPLIT_RATIO
    test_split: float = TEST_SPLIT_RATIO
    primary_metric: str = PRIMARY_METRIC
    cv_folds: int = CV_FOLDS
    raw_data_path: Path = RAW_DATA_DIR / "health_insurance_fraud_claims.xlsx"
    synthetic_data_path: Path = SYNTHETIC_DATA_DIR / "indian_health_insurance_claims_12k.csv"
    processed_train_path: Path = PROCESSED_DATA_DIR / "train_processed.csv"
    processed_val_path: Path = PROCESSED_DATA_DIR / "val_processed.csv"
    processed_test_path: Path = PROCESSED_DATA_DIR / "test_processed.csv"
    db_path: Path = DATA_DIR / "insurance_claims.db"
    log_file: Path = BASE_DIR / "system_pipeline.log"

config = ProjectConfig()
