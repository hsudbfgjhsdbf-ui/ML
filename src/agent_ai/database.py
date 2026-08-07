"""
Local SQLite database module for Medical Insurance Claim Fraud Detection (Approach 3).
Institution: IIIT Dharwad, Department of Data Science and AI
Faculty Adviser: Prof. Ramesh Athe
Team Members: B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)

This module implements the local database architecture required by Section 9:
1. `users`: Claimant personal profiles (Aadhaar, PAN, contact, age, gender).
2. `policies`: Insurance policy structures (Star Health, ICICI Lombard, HDFC Ergo, New India Assurance, Ayushman Bharat).
3. `claims`: Medical insurance claims with status and financial tracking.
4. `documents`: Metadata and JSON extraction payloads for uploaded claim documents.
5. `agent_results`: Complete audit trail of every AI agent's findings and confidence scores.
6. `fraud_reference`: Known Indian fraud patterns, blacklisted providers, and case studies.
7. `hospital_reference`: Indian healthcare provider directory with tier, cost range, and rejection statistics.
8. `medical_reference`: Standard Indian treatment protocols and typical INR cost benchmarks.
"""

import os
import json
import sqlite3
import logging
from typing import Dict, Any, List, Optional, Tuple
from src.utils import setup_logger, ensure_directories, format_inr

logger = setup_logger("AgentAIDatabaseLogger")


class InsuranceDatabaseManager:
    """
    Manages SQLite database lifecycle, schema initialization, and domain data population.
    """
    def __init__(self, db_path: str = "data/local_database.db"):
        self.db_path = db_path
        ensure_directories([os.path.dirname(db_path)])
        self._initialize_schema()
        self._seed_indian_domain_data()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_schema(self) -> None:
        """
        Creates all 8 required database tables if they do not exist.
        """
        logger.info(f"Initializing SQLite database schema at: {self.db_path}")
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 1. Users Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                aadhaar_number TEXT,
                pan_number TEXT,
                contact_number TEXT,
                email TEXT,
                address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. Policies Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS policies (
                policy_id TEXT PRIMARY KEY,
                policy_number TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                insurer_company TEXT NOT NULL,
                policy_type TEXT NOT NULL,
                sum_insured_inr REAL NOT NULL,
                annual_premium_inr REAL NOT NULL,
                co_pay_percentage REAL DEFAULT 0.0,
                room_rent_cap_inr REAL,
                waiting_period_months INTEGER DEFAULT 24,
                is_active INTEGER DEFAULT 1,
                start_date TEXT,
                end_date TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)
        
        # 3. Claims Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                policy_number TEXT NOT NULL,
                user_id TEXT NOT NULL,
                provider_id TEXT NOT NULL,
                hospital_name TEXT NOT NULL,
                treatment_type TEXT NOT NULL,
                diagnosis_code TEXT,
                procedure_code TEXT,
                admission_date TEXT,
                discharge_date TEXT,
                claimed_amount_inr REAL NOT NULL,
                approved_amount_inr REAL DEFAULT 0.0,
                status TEXT DEFAULT 'Submitted',
                decision_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (policy_number) REFERENCES policies (policy_number)
            )
        """)
        
        # 4. Documents Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                claim_id TEXT NOT NULL,
                document_type TEXT NOT NULL,
                file_name TEXT,
                file_path TEXT,
                extracted_json TEXT,
                extraction_confidence REAL DEFAULT 0.0,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (claim_id) REFERENCES claims (claim_id)
            )
        """)
        
        # 5. Agent Results Table (Audit Trail)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                input_snapshot TEXT,
                output_findings TEXT,
                confidence_score REAL,
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (claim_id) REFERENCES claims (claim_id)
            )
        """)
        
        # 6. Fraud Reference Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fraud_reference (
                reference_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT NOT NULL,
                description TEXT,
                risk_severity TEXT,
                applicable_specialties TEXT,
                indicator_keywords TEXT
            )
        """)
        
        # 7. Hospital Reference Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hospital_reference (
                provider_id TEXT PRIMARY KEY,
                hospital_name TEXT NOT NULL,
                hospital_tier TEXT NOT NULL,
                city TEXT NOT NULL,
                state TEXT NOT NULL,
                accreditation_status TEXT DEFAULT 'NABH Accredited',
                is_network_hospital INTEGER DEFAULT 1,
                historical_rejection_rate REAL DEFAULT 0.05,
                avg_inpatient_cost_inr REAL
            )
        """)
        
        # 8. Medical Reference Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS medical_reference (
                procedure_code TEXT PRIMARY KEY,
                procedure_name TEXT NOT NULL,
                specialty TEXT NOT NULL,
                tier1_avg_cost_inr REAL NOT NULL,
                tier2_avg_cost_inr REAL NOT NULL,
                tier3_avg_cost_inr REAL NOT NULL,
                standard_los_days INTEGER DEFAULT 3
            )
        """)
        
        conn.commit()
        conn.close()
        logger.debug("Database schema initialization complete.")

    def _seed_indian_domain_data(self) -> None:
        """
        Populates database tables with realistic Indian healthcare and insurance reference data.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Check if already seeded
        cursor.execute("SELECT COUNT(*) FROM hospital_reference")
        if cursor.fetchone()[0] > 0:
            conn.close()
            return
            
        logger.info("Seeding Indian insurance reference tables (Hospitals, Procedures, Fraud Patterns, Sample Policies)...")
        
        # 1. Seed Hospital Reference
        hospitals = [
            ("HOSP-MUM-01", "Apollo Hospitals Navi Mumbai", "Tier-1 Metro Corporate Hospital", "Mumbai", "Maharashtra", "NABH Accredited", 1, 0.04, 180000.0),
            ("HOSP-BLR-02", "Manipal Hospital HAL Airport Road", "Tier-1 Metro Corporate Hospital", "Bengaluru", "Karnataka", "NABH Accredited", 1, 0.03, 175000.0),
            ("HOSP-HYD-03", "Yashoda Hospitals Somajiguda", "Tier-1 Metro Corporate Hospital", "Hyderabad", "Telangana", "NABH Accredited", 1, 0.05, 165000.0),
            ("HOSP-PUN-04", "Sahyadri Super Specialty Hospital", "Tier-2 City Multi-Specialty Hospital", "Pune", "Maharashtra", "NABH Accredited", 1, 0.06, 110000.0),
            ("HOSP-JAI-05", "Fortis Escorts Jaipur", "Tier-2 City Multi-Specialty Hospital", "Jaipur", "Rajasthan", "NABH Accredited", 1, 0.07, 105000.0),
            ("HOSP-LUC-06", "Chandan Hospital Lucknow", "Tier-2 City Multi-Specialty Hospital", "Lucknow", "Uttar Pradesh", "NABH Accredited", 1, 0.08, 95000.0),
            ("HOSP-TWN-07", "Shree Krishna Nursing Home", "Tier-3 Town Nursing Home", "Mysuru", "Karnataka", "Non-Accredited", 0, 0.25, 45000.0),
            ("HOSP-TWN-08", "Jeevan Jyoti Hospital", "Tier-3 Town Nursing Home", "Coimbatore", "Tamil Nadu", "Non-Accredited", 0, 0.30, 40000.0)
        ]
        cursor.executemany("""
            INSERT OR REPLACE INTO hospital_reference VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, hospitals)
        
        # 2. Seed Medical Reference
        procedures = [
            ("IND-PROC-101", "Laparoscopic Appendectomy", "General Surgery", 140000.0, 85000.0, 45000.0, 2),
            ("IND-PROC-102", "Total Knee Replacement (Unilateral)", "Orthopedics", 320000.0, 210000.0, 130000.0, 5),
            ("IND-PROC-103", "Percutaneous Transluminal Coronary Angioplasty (PTCA)", "Cardiology", 280000.0, 190000.0, 120000.0, 3),
            ("IND-PROC-104", "Dengue Hemorrhagic Fever Management", "Internal Medicine", 95000.0, 55000.0, 30000.0, 4),
            ("IND-PROC-105", "Cataract Surgery with IOL Implantation", "Ophthalmology", 65000.0, 38000.0, 20000.0, 1),
            ("IND-PROC-106", "Cesarean Section Delivery", "Obstetrics & Gynecology", 110000.0, 68000.0, 38000.0, 3),
            ("IND-PROC-107", "MRI Brain with Contrast & Neuro-consultation", "Neurology", 35000.0, 22000.0, 15000.0, 1)
        ]
        cursor.executemany("""
            INSERT OR REPLACE INTO medical_reference VALUES (?, ?, ?, ?, ?, ?, ?)
        """, procedures)
        
        # 3. Seed Fraud Reference
        fraud_patterns = [
            ("Billing Inflation & Unbundled Charges", "Hospital bills items separately that are normally bundled into room rent or surgical package.", "High", "General Surgery, Orthopedics, Cardiology", "unbundled, consumables inflation, excessive PPE"),
            ("Tier-3 Hospital Charging Tier-1 Corporate Rates", "Non-accredited nursing home billing at metro corporate hospital rates without advanced ICU facilities.", "Critical", "All Specialties", "tier mismatch, inflated room rent, non-accredited premium"),
            ("Pre-Existing Disease (PED) Concealment", "Policyholder filing claim for chronic renal/cardiovascular condition within 24-month waiting period.", "High", "Cardiology, Nephrology", "early claim, chronic history, PED waiting period violation"),
            ("Organized Fraud Ring Collusion", "Multiple patients claiming identical diagnosis and treatment procedures at the same unlisted nursing home.", "Critical", "General Practice, Internal Medicine", "duplicate diagnosis, same doctor, repeated hospital claims"),
            ("Fake Document Fabrication", "Discharge summary or bills containing mismatched dates, invalid doctor registration numbers, or forged GSTIN.", "Critical", "All Specialties", "mismatched dates, invalid GSTIN, fabricated bills")
        ]
        cursor.executemany("""
            INSERT INTO fraud_reference (pattern_name, description, risk_severity, applicable_specialties, indicator_keywords)
            VALUES (?, ?, ?, ?, ?)
        """, fraud_patterns)
        
        # 4. Seed Sample User & Policy for Demo / Frontend
        cursor.execute("""
            INSERT OR REPLACE INTO users VALUES (
                'USR-IND-001', 'Rajesh Sharma', 48, 'M', '7845-1234-9012', 'ABCDE1234F', 
                '+91-9876543210', 'rajesh.sharma@iiitdwd.ac.in', 'Flat 402, Shanti Nagar, Bengaluru, Karnataka', datetime('now')
            )
        """)
        
        cursor.execute("""
            INSERT OR REPLACE INTO policies VALUES (
                'POL-STAR-001', 'STAR-HLTH-2024-8871', 'USR-IND-001', 'Star Health and Allied Insurance',
                'Family Floater Plan', 500000.0, 18500.0, 0.10, 5000.0, 24, 1, '2023-01-15', '2025-01-14'
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Indian domain reference data seeded successfully.")

    def insert_claim(self, claim_dict: Dict[str, Any]) -> str:
        """
        Inserts a new insurance claim record into the database.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO claims (
                claim_id, policy_number, user_id, provider_id, hospital_name, treatment_type,
                diagnosis_code, procedure_code, admission_date, discharge_date,
                claimed_amount_inr, approved_amount_inr, status, decision_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            claim_dict["claim_id"], claim_dict["policy_number"], claim_dict["user_id"],
            claim_dict["provider_id"], claim_dict["hospital_name"], claim_dict["treatment_type"],
            claim_dict.get("diagnosis_code", ""), claim_dict.get("procedure_code", ""),
            claim_dict.get("admission_date", ""), claim_dict.get("discharge_date", ""),
            float(claim_dict["claimed_amount_inr"]), 0.0, claim_dict.get("status", "Submitted"),
            claim_dict.get("decision_reason", "")
        ))
        conn.commit()
        conn.close()
        logger.debug(f"Inserted claim {claim_dict['claim_id']} into database.")
        return claim_dict["claim_id"]

    def record_agent_result(self, claim_id: str, agent_name: str, findings: Dict[str, Any], conf_score: float = 0.95) -> None:
        """
        Records an agent's processing output into the audit trail table.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO agent_results (claim_id, agent_name, output_findings, confidence_score)
            VALUES (?, ?, ?, ?)
        """, (claim_id, agent_name, json.dumps(findings, indent=2), conf_score))
        conn.commit()
        conn.close()

    def get_claim_details(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves complete claim, policy, and user details.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.*, p.policy_type, p.sum_insured_inr, p.co_pay_percentage, p.room_rent_cap_inr,
                   p.waiting_period_months, p.insurer_company, u.full_name, u.age as patient_age, u.gender
            FROM claims c
            LEFT JOIN policies p ON c.policy_number = p.policy_number
            LEFT JOIN users u ON c.user_id = u.user_id
            WHERE c.claim_id = ?
        """, (claim_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return dict(row)

    def get_hospital_info(self, provider_id_or_name: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM hospital_reference 
            WHERE provider_id = ? OR hospital_name LIKE ?
        """, (provider_id_or_name, f"%{provider_id_or_name}%"))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_procedure_benchmark(self, procedure_code_or_name: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM medical_reference 
            WHERE procedure_code = ? OR procedure_name LIKE ?
        """, (procedure_code_or_name, f"%{procedure_code_or_name}%"))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
