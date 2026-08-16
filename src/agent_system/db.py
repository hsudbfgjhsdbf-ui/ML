"""
Local Relational Database Engine (SQLite).
Manages tables for Users, Policies, Claims, Documents, Agent Results,
Fraud Rulebooks, Hospital Reference, and Medical Pricing Benchmarks.
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.config import config
from src.utils import logger

def get_db_connection(db_path: Path = config.db_path) -> sqlite3.Connection:
    """Creates a thread-safe connection to the local SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_local_database(db_path: Path = config.db_path) -> None:
    """Initializes all required relational tables and seeds domain reference data."""
    logger.info(f"Initializing local claims relational database at {db_path}")
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    
    # 1. Users Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        full_name TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        state TEXT NOT NULL,
        city TEXT NOT NULL,
        annual_income_inr REAL NOT NULL,
        aadhaar_hash TEXT NOT NULL,
        contact_phone TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 2. Policies Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS policies (
        policy_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        insurance_provider TEXT NOT NULL,
        policy_type TEXT NOT NULL,
        sum_insured_inr REAL NOT NULL,
        annual_premium_inr REAL NOT NULL,
        start_date DATE NOT NULL,
        duration_months INTEGER NOT NULL,
        waiting_period_months INTEGER NOT NULL,
        copay_percentage REAL NOT NULL,
        sub_limit_daycare_inr REAL,
        sub_limit_icu_per_day_inr REAL,
        is_active INTEGER DEFAULT 1,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    """)
    
    # 3. Claims Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS claims (
        claim_id TEXT PRIMARY KEY,
        policy_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        hospital_name TEXT NOT NULL,
        hospital_tier TEXT NOT NULL,
        admission_date DATE NOT NULL,
        discharge_date DATE NOT NULL,
        stay_duration_days INTEGER NOT NULL,
        diagnosis_category TEXT NOT NULL,
        icd10_code TEXT NOT NULL,
        treatment_name TEXT NOT NULL,
        claimed_amount_inr REAL NOT NULL,
        approved_amount_inr REAL DEFAULT 0.0,
        claim_status TEXT DEFAULT 'SUBMITTED',
        claim_submission_method TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        decision_summary TEXT,
        decision_hindi TEXT,
        FOREIGN KEY (policy_id) REFERENCES policies(policy_id),
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    """)
    
    # 4. Documents Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        claim_id TEXT NOT NULL,
        doc_type TEXT NOT NULL,
        file_path TEXT NOT NULL,
        extracted_text TEXT,
        extracted_data_json TEXT,
        extraction_confidence REAL,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
    );
    """)
    
    # 5. Agent Results Table (Audit Trail)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS agent_results (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim_id TEXT NOT NULL,
        agent_name TEXT NOT NULL,
        status TEXT NOT NULL,
        confidence_score REAL NOT NULL,
        findings_json TEXT NOT NULL,
        processing_time_ms REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
    );
    """)
    
    # 6. Medical Pricing Benchmarks Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS medical_pricing_benchmarks (
        benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
        diagnosis_category TEXT NOT NULL,
        treatment_name TEXT NOT NULL,
        hospital_tier TEXT NOT NULL,
        min_cost_inr REAL NOT NULL,
        max_cost_inr REAL NOT NULL,
        mean_cost_inr REAL NOT NULL,
        typical_stay_min_days INTEGER NOT NULL,
        typical_stay_max_days INTEGER NOT NULL
    );
    """)
    
    # 7. Fraud Rules Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS fraud_rules (
        rule_id TEXT PRIMARY KEY,
        rule_name TEXT NOT NULL,
        description TEXT NOT NULL,
        severity TEXT NOT NULL,
        threshold_value REAL
    );
    """)
    
    # 8. Hospital Reference Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS hospital_reference (
        hospital_id TEXT PRIMARY KEY,
        hospital_name TEXT NOT NULL,
        hospital_tier TEXT NOT NULL,
        state TEXT NOT NULL,
        city TEXT NOT NULL,
        accreditation TEXT NOT NULL,
        historical_rejection_rate REAL NOT NULL,
        is_network_hospital INTEGER DEFAULT 1
    );
    """)
    
    conn.commit()
    seed_reference_data(conn)
    conn.close()

def seed_reference_data(conn: sqlite3.Connection) -> None:
    """Populates domain benchmarks and rulebooks for RAG and agent queries."""
    cur = conn.cursor()
    
    # Check if already seeded
    cur.execute("SELECT COUNT(*) FROM medical_pricing_benchmarks")
    if cur.fetchone()[0] > 0:
        return
        
    logger.info("Seeding Indian healthcare reference benchmarks and fraud rulebooks...")
    
    # Seed Medical Benchmarks
    benchmarks = [
        ("Cardiovascular", "Coronary Angioplasty (PTCA)", "Tier 1 (Metro Super-Specialty)", 220000, 480000, 320000, 2, 5),
        ("Cardiovascular", "Coronary Angioplasty (PTCA)", "Tier 2 (City Multispecialty)", 140000, 260000, 195000, 2, 4),
        ("Cardiovascular", "Coronary Angioplasty (PTCA)", "Tier 3 (Nursing Home)", 90000, 160000, 125000, 2, 3),
        ("Orthopedics", "Total Knee Replacement (TKR)", "Tier 1 (Metro Super-Specialty)", 280000, 450000, 350000, 4, 7),
        ("Orthopedics", "Total Knee Replacement (TKR)", "Tier 2 (City Multispecialty)", 170000, 280000, 220000, 3, 6),
        ("Gastroenterology & General Surgery", "Laparoscopic Appendectomy", "Tier 1 (Metro Super-Specialty)", 80000, 160000, 115000, 2, 4),
        ("Gastroenterology & General Surgery", "Laparoscopic Appendectomy", "Tier 2 (City Multispecialty)", 45000, 95000, 68000, 1, 3),
        ("Gastroenterology & General Surgery", "Laparoscopic Appendectomy", "Tier 3 (Nursing Home)", 25000, 55000, 38000, 1, 2),
        ("Ophthalmology & Daycare", "Phacoemulsification with Foldable IOL (Cataract)", "Tier 1 (Metro Super-Specialty)", 45000, 95000, 65000, 0, 1),
        ("Ophthalmology & Daycare", "Phacoemulsification with Foldable IOL (Cataract)", "Tier 2 (City Multispecialty)", 25000, 50000, 35000, 0, 1)
    ]
    cur.executemany("""
    INSERT INTO medical_pricing_benchmarks 
    (diagnosis_category, treatment_name, hospital_tier, min_cost_inr, max_cost_inr, mean_cost_inr, typical_stay_min_days, typical_stay_max_days)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, benchmarks)
    
    # Seed Fraud Rules
    rules = [
        ("RULE-01", "Billing Inflation Exceeding Tier Norm", "Claim amount exceeds 2.2x the benchmark cost for treatment in given hospital tier", "HIGH", 2.2),
        ("RULE-02", "Waiting Period Violation", "Claim filed for pre-existing disease prior to completion of mandatory waiting period", "CRITICAL", 0.0),
        ("RULE-03", "Sum Insured Max-Out Velocity", "Single claim utilizes greater than 95% of total policy sum insured in first 6 months", "MEDIUM", 0.95),
        ("RULE-04", "Phantom Stay Mismatch", "Zero or 1 day inpatient stay claimed for major surgery requiring multi-day recovery", "CRITICAL", 1.0),
        ("RULE-05", "High Provider Rejection History", "Admitting healthcare provider has historical claim rejection rate exceeding 30%", "HIGH", 0.30)
    ]
    cur.executemany("""
    INSERT INTO fraud_rules (rule_id, rule_name, description, severity, threshold_value)
    VALUES (?, ?, ?, ?, ?)
    """, rules)
    
    # Seed Sample Hospitals
    hospitals = [
        ("HOSP-001", "Apollo Hospitals Bangalore", "Tier 1 (Metro Super-Specialty)", "Karnataka", "Bengaluru", "NABH", 0.04, 1),
        ("HOSP-002", "SDM College of Medical Sciences Dharwad", "Tier 2 (City Multispecialty)", "Karnataka", "Dharwad", "NABH", 0.06, 1),
        ("HOSP-003", "KIMS Hospital Hubballi", "Tier 2 (City Multispecialty)", "Karnataka", "Hubballi", "Government/NABH", 0.05, 1),
        ("HOSP-004", "Fortis Hospital Mumbai", "Tier 1 (Metro Super-Specialty)", "Maharashtra", "Mumbai", "JCI/NABH", 0.03, 1),
        ("HOSP-005", "City Care Nursing Home", "Tier 3 (Nursing Home)", "Uttar Pradesh", "Kanpur", "Unaccredited", 0.38, 0)
    ]
    cur.executemany("""
    INSERT INTO hospital_reference 
    (hospital_id, hospital_name, hospital_tier, state, city, accreditation, historical_rejection_rate, is_network_hospital)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, hospitals)
    
    # Seed Sample Test User & Policy for demonstration
    cur.execute("""
    INSERT OR REPLACE INTO users (user_id, full_name, age, gender, state, city, annual_income_inr, aadhaar_hash, contact_phone)
    VALUES ('USR-IND-8821', 'Ramesh Kumar Patil', 54, 'Male', 'Karnataka', 'Dharwad', 750000, 'XXXX-XXXX-8921', '+91-9845012345')
    """)
    cur.execute("""
    INSERT OR REPLACE INTO policies (policy_id, user_id, insurance_provider, policy_type, sum_insured_inr, annual_premium_inr, start_date, duration_months, waiting_period_months, copay_percentage, sub_limit_daycare_inr, sub_limit_icu_per_day_inr, is_active)
    VALUES ('POL-STAR-44912', 'USR-IND-8821', 'Star Health and Allied Insurance', 'Family Floater Plan', 500000, 18500, '2023-04-15', 28, 24, 10.0, 50000, 10000, 1)
    """)
    
    conn.commit()

def insert_claim_record(claim_data: Dict[str, Any], db_path: Path = config.db_path) -> str:
    """Inserts a new claim submission into the local database."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
    INSERT OR REPLACE INTO claims (
        claim_id, policy_id, user_id, hospital_name, hospital_tier,
        admission_date, discharge_date, stay_duration_days, diagnosis_category,
        icd10_code, treatment_name, claimed_amount_inr, approved_amount_inr,
        claim_status, claim_submission_method, decision_summary, decision_hindi
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        claim_data["claim_id"],
        claim_data["policy_id"],
        claim_data["user_id"],
        claim_data.get("hospital_name", "SDM College of Medical Sciences Dharwad"),
        claim_data.get("hospital_tier", "Tier 2 (City Multispecialty)"),
        claim_data.get("admission_date", "2024-06-10"),
        claim_data.get("discharge_date", "2024-06-13"),
        int(claim_data.get("stay_duration_days", 3)),
        claim_data.get("diagnosis_category", "Gastroenterology & General Surgery"),
        claim_data.get("icd10_code", "K35.8"),
        claim_data.get("treatment_name", "Laparoscopic Appendectomy"),
        float(claim_data["claimed_amount_inr"]),
        float(claim_data.get("approved_amount_inr", 0.0)),
        claim_data.get("claim_status", "SUBMITTED"),
        claim_data.get("claim_submission_method", "Digital_Portal"),
        claim_data.get("decision_summary", ""),
        claim_data.get("decision_hindi", "")
    ))
    conn.commit()
    conn.close()
    return claim_data["claim_id"]

def get_claim_with_details(claim_id: str, db_path: Path = config.db_path) -> Optional[Dict[str, Any]]:
    """Fetches a complete claim record joined with user and policy data."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
    SELECT c.*, u.full_name, u.age, u.gender, u.state, u.city, u.annual_income_inr,
           p.insurance_provider, p.policy_type, p.sum_insured_inr, p.annual_premium_inr,
           p.duration_months, p.waiting_period_months, p.copay_percentage
    FROM claims c
    JOIN users u ON c.user_id = u.user_id
    JOIN policies p ON c.policy_id = p.policy_id
    WHERE c.claim_id = ?
    """, (claim_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None
