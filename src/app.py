"""
FastAPI Backend and Web Interface for Medical Insurance Claim Fraud Detection System.
Supports Approach 1 (ML), Approach 2 (DL), and Approach 3 (Agent AI Multi-Agent System)
with live preview endpoints, interactive claim submission, document processing, and explainable AI decisions.
"""

import os
import uuid
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.database.models import init_db, SessionLocal, ClaimDB, PolicyDB, UserDB, DocumentDB
from src.agents.workflow import MultiAgentClaimProcessor

app = FastAPI(
    title="Medical Insurance Claim Fraud Detection System (IIIT Dharwad)",
    description="End-to-end multi-approach platform: Traditional ML, Deep Learning, and Agent AI Multi-Agent System.",
    version="1.0.0"
)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()
    db = SessionLocal()
    # Add a sample policy and user if empty
    if db.query(PolicyDB).count() == 0:
        sample_user = UserDB(user_id="U-1001", name="Ramesh Kumar", email="ramesh.kumar@example.com", phone="+91-9876543210", role="claimant")
        sample_policy = PolicyDB(policy_number="POL-998877", user_id="U-1001", policy_type="Family Floater", sum_insured=1000000.0, premium_amount=25000.0, copay_percentage=10.0)
        db.add(sample_user)
        db.add(sample_policy)
        db.commit()
    db.close()

class ClaimSubmitRequest(BaseModel):
    policy_number: str
    patient_name: str
    treatment_type: str
    hospital_name: str
    hospital_tier: str = "Tier 2"
    claimed_amount: float
    diagnosis: str
    admission_date: str
    discharge_date: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medical Insurance Claim Fraud Detection - IIIT Dharwad</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f4f8; margin: 0; padding: 0; color: #333; }
            header { background: #1e3a8a; color: white; padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            header h1 { margin: 0; font-size: 2.2rem; }
            header p { margin: 0.5rem 0 0; font-size: 1.1rem; color: #93c5fd; }
            .container { max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }
            .card { background: white; border-radius: 12px; padding: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 2rem; }
            h2 { color: #1e3a8a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }
            .form-group { margin-bottom: 1.2rem; }
            label { display: block; font-weight: 600; margin-bottom: 0.4rem; color: #475569; }
            input, select { width: 100%; padding: 0.75rem; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 1rem; box-sizing: border-box; }
            button { background: #2563eb; color: white; border: none; padding: 0.85rem 1.5rem; font-size: 1rem; font-weight: 600; border-radius: 8px; cursor: pointer; transition: background 0.2s; }
            button:hover { background: #1d4ed8; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
            #result { margin-top: 1.5rem; padding: 1.5rem; background: #f8fafc; border-left: 5px solid #2563eb; border-radius: 8px; display: none; }
            .badge { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 9999px; font-weight: 600; font-size: 0.9rem; }
            .badge-Approved { background: #dcfce7; color: #166534; }
            .badge-Flagged { background: #fef9c3; color: #854d0e; }
            .badge-Rejected { background: #fee2e2; color: #991b1b; }
            footer { text-align: center; padding: 2rem; color: #64748b; font-size: 0.9rem; }
        </style>
    </head>
    <body>
        <header>
            <h1>Medical Insurance Claim Fraud Detection System</h1>
            <p>IIIT Dharwad | Faculty Adviser: Ramesh Athe | Team: B Varshith, M Jagadeshwar, J Ganesh</p>
        </header>
        <div class="container">
            <div class="card">
                <h2>Submit Insurance Claim (Agent AI Multi-Agent Verification)</h2>
                <form id="claimForm">
                    <div class="grid">
                        <div class="form-group">
                            <label>Policy Number</label>
                            <input type="text" id="policy_number" value="POL-998877" required>
                        </div>
                        <div class="form-group">
                            <label>Patient Name</label>
                            <input type="text" id="patient_name" value="Ananya Sharma" required>
                        </div>
                    </div>
                    <div class="grid">
                        <div class="form-group">
                            <label>Treatment Type</label>
                            <select id="treatment_type">
                                <option value="Appendectomy">Appendectomy</option>
                                <option value="Cardiology Bypass">Cardiology Bypass</option>
                                <option value="Orthopedic Surgery">Orthopedic Surgery</option>
                                <option value="Routine Health Checkup">Routine Health Checkup</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Hospital Name & Location</label>
                            <input type="text" id="hospital_name" value="Apollo Hospital, Mumbai" required>
                        </div>
                    </div>
                    <div class="grid">
                        <div class="form-group">
                            <label>Hospital Tier</label>
                            <select id="hospital_tier">
                                <option value="Tier 1">Tier 1 (Metro Corporate)</option>
                                <option value="Tier 2" selected>Tier 2 (City Multispecialty)</option>
                                <option value="Tier 3">Tier 3 (Local Nursing Home)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Claimed Amount (INR)</label>
                            <input type="number" id="claimed_amount" value="125000" required>
                        </div>
                    </div>
                    <div class="grid">
                        <div class="form-group">
                            <label>Diagnosis Code</label>
                            <input type="text" id="diagnosis" value="ICD-10: K35.3 Acute Appendicitis" required>
                        </div>
                        <div class="form-group">
                            <label>Admission & Discharge Dates</label>
                            <input type="text" id="admission_date" value="2026-08-01 to 2026-08-04" required>
                        </div>
                    </div>
                    <button type="submit">Run Multi-Agent Verification Pipeline</button>
                </form>
                <div id="result">
                    <h3>Verification Verdict</h3>
                    <p><strong>Claim ID:</strong> <span id="res_id"></span></p>
                    <p><strong>Decision:</strong> <span id="res_decision"></span></p>
                    <p><strong>Approved Amount:</strong> Rs <span id="res_amount"></span></p>
                    <p><strong>Explainable AI Reasoning:</strong> <span id="res_reasoning"></span></p>
                </div>
            </div>
            
            <div class="card">
                <h2>Project Benchmarks & Approaches</h2>
                <ul>
                    <li><strong>Approach 1: Traditional Machine Learning</strong> (12 algorithms evaluated, top model: Random Forest / LightGBM with F2-Score > 0.92). <a href="/docs" target="_blank">View Docs</a></li>
                    <li><strong>Approach 2: Deep Learning</strong> (10 architectures: MLP, Transformer, ResNet, TabNet, LSTM, Autoencoder, VAE).</li>
                    <li><strong>Approach 3: Agent AI Multi-Agent System</strong> (LangGraph orchestration, OCR, RAG policy verification, explainable AI).</li>
                </ul>
            </div>
        </div>
        <footer>
            &copy; 2026 IIIT Dharwad | Department of Data Science and AI
        </footer>
        <script>
            document.getElementById('claimForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const payload = {
                    policy_number: document.getElementById('policy_number').value,
                    patient_name: document.getElementById('patient_name').value,
                    treatment_type: document.getElementById('treatment_type').value,
                    hospital_name: document.getElementById('hospital_name').value,
                    hospital_tier: document.getElementById('hospital_tier').value,
                    claimed_amount: parseFloat(document.getElementById('claimed_amount').value),
                    diagnosis: document.getElementById('diagnosis').value,
                    admission_date: document.getElementById('admission_date').value.split(' to ')[0],
                    discharge_date: document.getElementById('admission_date').value.split(' to ')[1] || "2026-08-04"
                };
                
                const response = await fetch('/api/submit_claim', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                
                document.getElementById('res_id').innerText = data.claim_id;
                const decSpan = document.getElementById('res_decision');
                decSpan.innerText = data.decision;
                decSpan.className = 'badge badge-' + data.decision;
                document.getElementById('res_amount').innerText = data.approved_amount.toLocaleString('en-IN');
                document.getElementById('res_reasoning').innerText = data.reasoning;
                document.getElementById('result').style.display = 'block';
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/api/submit_claim")
def submit_claim(req: ClaimSubmitRequest):
    db = SessionLocal()
    try:
        claim_id = f"CL-{uuid.uuid4().hex[:8].upper()}"
        claim = ClaimDB(
            claim_id=claim_id,
            policy_number=req.policy_number,
            patient_name=req.patient_name,
            treatment_type=req.treatment_type,
            hospital_name=req.hospital_name,
            hospital_tier=req.hospital_tier,
            claimed_amount=req.claimed_amount,
            diagnosis=req.diagnosis,
            admission_date=req.admission_date,
            discharge_date=req.discharge_date,
            status='Under Review'
        )
        db.add(claim)
        db.commit()
        
        # Run Multi-Agent Processor
        processor = MultiAgentClaimProcessor()
        result = processor.process_claim(claim_id)
        return result
    finally:
        db.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
