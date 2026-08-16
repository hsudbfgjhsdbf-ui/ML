"""
FastAPI REST API Server for Medical Insurance Claim Verification and Inference.
Exposes endpoints for claim submission, document analysis, agent pipeline orchestration,
ML/DL model predictions, vector RAG search, and benchmarking.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import os
import json
import sqlite3
from pathlib import Path

from src.config import config, INDIAN_STATES, HOSPITAL_TIERS, INSURANCE_PROVIDERS, POLICY_TYPES
from src.agent_system.coordinator import MultiAgentCoordinator
from src.agent_system.db import (
    initialize_local_database, get_db_connection,
    insert_claim_record, get_claim_with_details
)
from src.agent_system.rag_engine import InsuranceKnowledgeRAG
from src.utils import logger

# Initialize database tables on startup
initialize_local_database()
coordinator = MultiAgentCoordinator()
rag_engine = InsuranceKnowledgeRAG()

app = FastAPI(
    title="Medical Insurance Claim Fraud Detection & Explainable AI Verification API",
    description="IIIT Dharwad B.Tech DS&AI — Multi-Agent Cognitive Platform & Benchmarking Suite",
    version="1.0.0"
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClaimSubmissionSchema(BaseModel):
    full_name: str = Field(default="Ramesh Kumar Patil")
    age: int = Field(default=54, ge=0, le=120)
    gender: str = Field(default="Male")
    state: str = Field(default="Karnataka")
    city: str = Field(default="Dharwad")
    annual_income_inr: float = Field(default=750000.0)
    insurance_provider: str = Field(default="Star Health and Allied Insurance")
    policy_type: str = Field(default="Family Floater Plan")
    sum_insured_inr: float = Field(default=500000.0)
    annual_premium_inr: float = Field(default=18500.0)
    duration_months: int = Field(default=28)
    waiting_period_months: int = Field(default=24)
    copay_percentage: float = Field(default=10.0)
    hospital_name: str = Field(default="SDM College of Medical Sciences Dharwad")
    hospital_tier: str = Field(default="Tier 2 (City Multispecialty)")
    diagnosis_category: str = Field(default="Gastroenterology & General Surgery")
    icd10_code: str = Field(default="K35.8")
    treatment_name: str = Field(default="Laparoscopic Appendectomy")
    stay_duration_days: int = Field(default=3)
    claimed_amount_inr: float = Field(default=78000.0)
    claim_submission_method: str = Field(default="Digital_Portal")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Medical Insurance Fraud Detection API",
        "institution": "IIIT Dharwad",
        "faculty_adviser": "Ramesh Athe",
        "team": ["B Varshith (23BDS011)", "M Jagadeshwar (23BDS033)", "J Ganesh (23BDS024)"]
    }

@app.post("/api/claims/submit")
def submit_and_verify_claim(claim_in: ClaimSubmissionSchema):
    """
    Submits a new insurance claim, persists to local database, and executes
    the full multi-agent cognitive verification pipeline.
    """
    import time
    claim_dict = claim_in.dict()
    claim_id = f"CLM-LIVE-{int(time.time()*1000)%1000000}"
    user_id = f"USR-{int(time.time()*10)%100000}"
    policy_id = f"POL-{claim_in.insurance_provider[:4].upper()}-{int(time.time())%10000}"
    
    claim_dict["claim_id"] = claim_id
    claim_dict["user_id"] = user_id
    claim_dict["policy_id"] = policy_id
    
    # Save claim into SQLite database
    insert_claim_record(claim_dict)
    
    # Execute Multi-Agent Verification Workflow
    verification_report = coordinator.process_claim_end_to_end(claim_dict)
    return verification_report

@app.get("/api/claims/{claim_id}")
def get_claim_status(claim_id: str):
    """Fetches details and audit logs for an existing claim."""
    claim = get_claim_with_details(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found in database.")
    return claim

@app.get("/api/rag/search")
def search_knowledge_base(query: str = Query(..., min_length=2), top_k: int = 3):
    """Searches vector knowledge base for policy clauses and medical tariffs."""
    results = rag_engine.retrieve(query, top_k=top_k)
    return {"query": query, "matches": results}

@app.get("/api/benchmarks")
def get_benchmark_summary():
    """Returns benchmark comparison table across ML and DL models."""
    summary_md_path = config.raw_data_path.parent.parent / "evaluation" / "benchmark_summary.md"
    if summary_md_path.exists():
        with open(summary_md_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"benchmark_markdown": content}
    return {"status": "Run pipeline to generate complete benchmarks."}
