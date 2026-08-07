"""
Database models using SQLAlchemy for Approach 3 (Agent AI System).
Stores users, policies, claims, uploaded documents, and agent verification results.
"""

from sqlalchemy import create_engine, Column, String, Float, Integer, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

class UserDB(Base):
    __tablename__ = 'users'
    user_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    phone = Column(String)
    role = Column(String, default='claimant') # claimant, agent, admin
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class PolicyDB(Base):
    __tablename__ = 'policies'
    policy_number = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    policy_type = Column(String, nullable=False) # Family Floater, Individual, Group, Senior Citizen
    sum_insured = Column(Float, nullable=False)
    premium_amount = Column(Float, nullable=False)
    waiting_period_months = Column(Integer, default=24)
    start_date = Column(String)
    end_date = Column(String)
    copay_percentage = Column(Float, default=10.0)

class ClaimDB(Base):
    __tablename__ = 'claims'
    claim_id = Column(String, primary_key=True)
    policy_number = Column(String, nullable=False)
    patient_name = Column(String, nullable=False)
    treatment_type = Column(String, nullable=False)
    hospital_name = Column(String, nullable=False)
    hospital_tier = Column(String, default='Tier 2')
    claimed_amount = Column(Float, nullable=False)
    approved_amount = Column(Float, default=0.0)
    diagnosis = Column(String)
    admission_date = Column(String)
    discharge_date = Column(String)
    status = Column(String, default='Submitted') # Submitted, Under Review, Approved, Flagged, Rejected
    decision = Column(String, default='Pending')
    reasoning_explanation = Column(Text, default='')
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class DocumentDB(Base):
    __tablename__ = 'documents'
    doc_id = Column(String, primary_key=True)
    claim_id = Column(String, nullable=False)
    doc_type = Column(String, nullable=False) # Bill, Prescription, Discharge Summary, Lab Report
    file_path = Column(String, nullable=False)
    extracted_data = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class AgentResultDB(Base):
    __tablename__ = 'agent_results'
    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_id = Column(String, nullable=False)
    agent_name = Column(String, nullable=False) # DocumentAgent, PolicyAgent, AnomalyAgent, HistoryAgent, ReasoningAgent
    findings = Column(JSON, default={})
    confidence_score = Column(Float, default=0.95)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

engine = create_engine('sqlite:///data/agent_system.db', echo=False)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    import os
    os.makedirs('data', exist_ok=True)
    Base.metadata.create_all(bind=engine)
    print("Agent AI SQLite Database initialized at data/agent_system.db")

if __name__ == '__main__':
    init_db()
