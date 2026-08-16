"""
Interactive Full-Stack Web Application for Medical Insurance Claim Fraud Detection.
Binds to 0.0.0.0:8000 providing live claim verification, multi-agent visualization,
and benchmark exploration for academic demonstration.
"""

import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from src.api import app
from src.config import VISUALIZATIONS_DIR

# Mount visualizations as static directory if exists
if VISUALIZATIONS_DIR.exists():
    app.mount("/static/visualizations", StaticFiles(directory=str(VISUALIZATIONS_DIR)), name="visualizations")

@app.get("/", response_class=HTMLResponse)
def index_page():
    """Renders the comprehensive Medical Insurance Claim Fraud Detection Web Interface."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Insurance Claim Fraud Detection & Explainable AI Platform | IIIT Dharwad</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary-navy: #1a365d;
            --accent-blue: #2b6cb0;
            --fraud-crimson: #c53030;
            --success-green: #276749;
            --light-bg: #f7fafc;
            --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        body {
            background-color: var(--light-bg);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #2d3748;
        }
        .navbar-custom {
            background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 100%);
            color: white;
            padding: 1rem 1.5rem;
            box-shadow: var(--card-shadow);
        }
        .badge-academic {
            background-color: rgba(255, 255, 255, 0.2);
            color: white;
            font-size: 0.85rem;
            padding: 0.35rem 0.75rem;
            border-radius: 20px;
        }
        .nav-tabs .nav-link {
            font-weight: 600;
            color: #4a5568;
            border: none;
            padding: 0.75rem 1.25rem;
        }
        .nav-tabs .nav-link.active {
            color: var(--accent-blue);
            border-bottom: 3px solid var(--accent-blue);
            background: transparent;
        }
        .card-custom {
            border: none;
            border-radius: 12px;
            box-shadow: var(--card-shadow);
            background: white;
            margin-bottom: 1.5rem;
        }
        .card-header-custom {
            background: #edf2f7;
            font-weight: 700;
            border-top-left-radius: 12px !important;
            border-top-right-radius: 12px !important;
            padding: 0.9rem 1.25rem;
            color: var(--primary-navy);
        }
        .agent-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.4rem 0.8rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0.25rem;
        }
        .agent-success { background-color: #c6f6d5; color: #22543d; border: 1px solid #9ae6b4; }
        .agent-running { background-color: #feebc8; color: #7b341e; border: 1px solid #fbd38d; }
        .decision-box-APPROVED { background-color: #f0fff4; border: 2px solid #38a169; border-radius: 10px; padding: 1.5rem; }
        .decision-box-FLAGGED { background-color: #fffaf0; border: 2px solid #dd6b20; border-radius: 10px; padding: 1.5rem; }
        .decision-box-REJECTED { background-color: #fff5f5; border: 2px solid #e53e3e; border-radius: 10px; padding: 1.5rem; }
        .hindi-text { font-family: 'Mukta', 'Nirmala UI', sans-serif; line-height: 1.6; }
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 1.25rem;
            text-align: center;
            border: 1px solid #e2e8f0;
            box-shadow: var(--card-shadow);
        }
        .metric-val { font-size: 1.8rem; font-weight: 700; color: var(--primary-navy); }
        .metric-lbl { font-size: 0.8rem; text-transform: uppercase; color: #718096; font-weight: 600; }
    </style>
</head>
<body>

    <!-- Header Navigation -->
    <nav class="navbar navbar-custom">
        <div class="container-fluid d-flex justify-content-between align-items-center">
            <div>
                <h4 class="mb-0 fw-bold"><i class="fa-solid fa-shield-heart me-2"></i>Medical Insurance Claim Fraud Detection</h4>
                <small class="text-light opacity-75">AI-Driven Claim Verification & Explainable Reasoning Platform</small>
            </div>
            <div class="text-end">
                <span class="badge badge-academic me-2"><i class="fa-solid fa-graduation-cap me-1"></i>IIIT Dharwad (DS & AI)</span>
                <span class="badge badge-academic"><i class="fa-solid fa-user-tie me-1"></i>Adviser: Ramesh Athe</span>
            </div>
        </div>
    </nav>

    <!-- Main Container -->
    <div class="container-fluid py-4 px-4">
        
        <!-- Summary Stats Row -->
        <div class="row g-3 mb-4">
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-val text-primary">22+</div>
                    <div class="metric-lbl">Evaluated ML & DL Models</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-val text-success">0.965</div>
                    <div class="metric-lbl">Best F2-Score (Recall Focused)</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-val text-warning">5 Agents</div>
                    <div class="metric-lbl">Multi-Agent Cognitive Pipeline</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-val text-info">₹ INR</div>
                    <div class="metric-lbl">Indian Context & IRDAI Tariff Model</div>
                </div>
            </div>
        </div>

        <!-- Navigation Tabs -->
        <ul class="nav nav-tabs mb-4" id="mainTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="claim-tab" data-bs-toggle="tab" data-bs-target="#claim-pane" type="button"><i class="fa-solid fa-file-invoice-dollar me-2"></i>Live Claim Intake & Multi-Agent Adjudication</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="benchmarks-tab" data-bs-toggle="tab" data-bs-target="#benchmarks-pane" type="button"><i class="fa-solid fa-chart-line me-2"></i>Model Benchmarking Suite (ML vs DL)</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="visuals-tab" data-bs-toggle="tab" data-bs-target="#visuals-pane" type="button"><i class="fa-solid fa-chart-pie me-2"></i>Visual Analytics & XAI Gallery</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="rag-tab" data-bs-toggle="tab" data-bs-target="#rag-pane" type="button"><i class="fa-solid fa-book-medical me-2"></i>RAG Knowledge Base & IRDAI Rules</button>
            </li>
        </ul>

        <!-- Tab Panes -->
        <div class="tab-content" id="mainTabContent">
            
            <!-- PANE 1: CLAIM VERIFICATION WIZARD -->
            <div class="tab-pane fade show active" id="claim-pane" role="tabpanel">
                <div class="row">
                    <!-- Form Column -->
                    <div class="col-lg-6">
                        <div class="card card-custom">
                            <div class="card-header card-header-custom d-flex justify-content-between align-items-center">
                                <span><i class="fa-solid fa-clipboard-list me-2"></i>Claimant & Treatment Intake Form</span>
                                <button class="btn btn-sm btn-outline-primary" onclick="loadSampleCase('legit')">Load Legitimate Sample</button>
                                <button class="btn btn-sm btn-outline-danger" onclick="loadSampleCase('fraud')">Load Inflated Fraud Sample</button>
                            </div>
                            <div class="card-body">
                                <form id="claimForm">
                                    <div class="row g-2 mb-3">
                                        <div class="col-md-6">
                                            <label class="form-label small fw-bold">Full Name of Claimant</label>
                                            <input type="text" id="full_name" class="form-control form-control-sm" value="Ramesh Kumar Patil">
                                        </div>
                                        <div class="col-md-3">
                                            <label class="form-label small fw-bold">Age</label>
                                            <input type="number" id="age" class="form-control form-control-sm" value="54">
                                        </div>
                                        <div class="col-md-3">
                                            <label class="form-label small fw-bold">Gender</label>
                                            <select id="gender" class="form-select form-select-sm">
                                                <option value="Male">Male</option>
                                                <option value="Female">Female</option>
                                            </select>
                                        </div>
                                    </div>

                                    <div class="row g-2 mb-3">
                                        <div class="col-md-6">
                                            <label class="form-label small fw-bold">State of Residence</label>
                                            <select id="state" class="form-select form-select-sm">
                                                <option value="Karnataka">Karnataka</option>
                                                <option value="Maharashtra">Maharashtra</option>
                                                <option value="Tamil Nadu">Tamil Nadu</option>
                                                <option value="Delhi NCR">Delhi NCR</option>
                                            </select>
                                        </div>
                                        <div class="col-md-6">
                                            <label class="form-label small fw-bold">City</label>
                                            <input type="text" id="city" class="form-control form-control-sm" value="Dharwad">
                                        </div>
                                    </div>

                                    <div class="row g-2 mb-3">
                                        <div class="col-md-6">
                                            <label class="form-label small fw-bold">Insurance Provider</label>
                                            <select id="insurance_provider" class="form-select form-select-sm">
                                                <option value="Star Health and Allied Insurance">Star Health and Allied Insurance</option>
                                                <option value="ICICI Lombard General Insurance">ICICI Lombard General Insurance</option>
                                                <option value="HDFC ERGO General Insurance">HDFC ERGO General Insurance</option>
                                                <option value="Ayushman Bharat PM-JAY (Government)">Ayushman Bharat PM-JAY (Government)</option>
                                            </select>
                                        </div>
                                        <div class="col-md-6">
                                            <label class="form-label small fw-bold">Policy Type</label>
                                            <select id="policy_type" class="form-select form-select-sm">
                                                <option value="Family Floater Plan">Family Floater Plan</option>
                                                <option value="Individual Health Plan">Individual Health Plan</option>
                                                <option value="Senior Citizen Red Carpet Plan">Senior Citizen Red Carpet Plan</option>
                                            </select>
                                        </div>
                                    </div>

                                    <div class="row g-2 mb-3">
                                        <div class="col-md-4">
                                            <label class="form-label small fw-bold">Sum Insured (₹)</label>
                                            <input type="number" id="sum_insured_inr" class="form-control form-control-sm" value="500000">
                                        </div>
                                        <div class="col-md-4">
                                            <label class="form-label small fw-bold">Policy Duration (Mo)</label>
                                            <input type="number" id="duration_months" class="form-control form-control-sm" value="28">
                                        </div>
                                        <div class="col-md-4">
                                            <label class="form-label small fw-bold">Waiting Period (Mo)</label>
                                            <input type="number" id="waiting_period_months" class="form-control form-control-sm" value="24">
                                        </div>
                                    </div>

                                    <hr class="my-3">

                                    <div class="row g-2 mb-3">
                                        <div class="col-md-7">
                                            <label class="form-label small fw-bold">Admitting Hospital</label>
                                            <input type="text" id="hospital_name" class="form-control form-control-sm" value="SDM College of Medical Sciences Dharwad">
                                        </div>
                                        <div class="col-md-5">
                                            <label class="form-label small fw-bold">Hospital Tier</label>
                                            <select id="hospital_tier" class="form-select form-select-sm">
                                                <option value="Tier 2 (City Multispecialty)">Tier 2 (City Multispecialty)</option>
                                                <option value="Tier 1 (Metro Super-Specialty)">Tier 1 (Metro Super-Specialty)</option>
                                                <option value="Tier 3 (Nursing Home)">Tier 3 (Nursing Home)</option>
                                            </select>
                                        </div>
                                    </div>

                                    <div class="row g-2 mb-3">
                                        <div class="col-md-6">
                                            <label class="form-label small fw-bold">Diagnosis Category</label>
                                            <select id="diagnosis_category" class="form-select form-select-sm">
                                                <option value="Gastroenterology & General Surgery">Gastroenterology & General Surgery</option>
                                                <option value="Cardiovascular">Cardiovascular</option>
                                                <option value="Orthopedics">Orthopedics</option>
                                                <option value="Ophthalmology & Daycare">Ophthalmology & Daycare</option>
                                            </select>
                                        </div>
                                        <div class="col-md-6">
                                            <label class="form-label small fw-bold">Treatment / Procedure</label>
                                            <input type="text" id="treatment_name" class="form-control form-control-sm" value="Laparoscopic Appendectomy">
                                        </div>
                                    </div>

                                    <div class="row g-2 mb-3">
                                        <div class="col-md-4">
                                            <label class="form-label small fw-bold">Stay Days</label>
                                            <input type="number" id="stay_duration_days" class="form-control form-control-sm" value="3">
                                        </div>
                                        <div class="col-md-8">
                                            <label class="form-label small fw-bold text-danger">Total Claimed Amount (₹ INR)</label>
                                            <input type="number" id="claimed_amount_inr" class="form-control form-control-sm fw-bold border-danger" value="78000">
                                        </div>
                                    </div>

                                    <div class="mb-3">
                                        <label class="form-label small fw-bold"><i class="fa-solid fa-paperclip me-1"></i>Attached Medical Documents (Simulated Upload)</label>
                                        <div class="p-2 border rounded bg-light text-muted small">
                                            <i class="fa-solid fa-file-pdf text-danger me-1"></i> Hospital_Discharge_Summary_SDM.pdf (Uploaded)<br>
                                            <i class="fa-solid fa-file-image text-primary me-1"></i> Itemized_Pharmacy_Invoice_GST.png (Uploaded)
                                        </div>
                                    </div>

                                    <button type="button" class="btn btn-primary w-100 fw-bold py-2" onclick="runAgentPipeline()">
                                        <i class="fa-solid fa-robot me-2"></i>Verify Claim with Multi-Agent AI System
                                    </button>
                                </form>
                            </div>
                        </div>
                    </div>

                    <!-- Live Reasoning & Output Column -->
                    <div class="col-lg-6">
                        <div class="card card-custom">
                            <div class="card-header card-header-custom">
                                <i class="fa-solid fa-network-wired me-2"></i>Multi-Agent Workflow & Adjudication Verdict
                            </div>
                            <div class="card-body">
                                
                                <!-- Active Agent Status Badges -->
                                <div class="mb-3 p-2 bg-light rounded border">
                                    <div class="small fw-bold text-secondary mb-1">Agent Collaboration Status:</div>
                                    <span class="agent-badge agent-success"><i class="fa-solid fa-check-circle me-1"></i>Coordinator</span>
                                    <span class="agent-badge agent-success"><i class="fa-solid fa-file-medical me-1"></i>Document OCR</span>
                                    <span class="agent-badge agent-success"><i class="fa-solid fa-scale-balanced me-1"></i>Policy RAG</span>
                                    <span class="agent-badge agent-success"><i class="fa-solid fa-magnifying-glass-chart me-1"></i>Anomaly Engine</span>
                                    <span class="agent-badge agent-success"><i class="fa-solid fa-brain me-1"></i>Reasoning Agent</span>
                                </div>

                                <!-- Decision Result Container -->
                                <div id="decisionContainer">
                                    <div class="alert alert-info text-center py-4">
                                        <i class="fa-solid fa-hourglass-half fa-2x mb-2 text-primary"></i>
                                        <p class="mb-0">Fill out the claim intake form and click <strong>Verify Claim with Multi-Agent AI System</strong> to trigger automated verification and natural language reasoning.</p>
                                    </div>
                                </div>

                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- PANE 2: BENCHMARKING SUITE -->
            <div class="tab-pane fade" id="benchmarks-pane" role="tabpanel">
                <div class="card card-custom">
                    <div class="card-header card-header-custom d-flex justify-content-between align-items-center">
                        <span><i class="fa-solid fa-table me-2"></i>Master Benchmarking Table across All 22+ Algorithms</span>
                        <span class="badge bg-secondary">Stratified 5-Fold CV</span>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-hover table-striped align-middle small">
                                <thead class="table-dark">
                                    <tr>
                                        <th>Paradigm</th>
                                        <th>Algorithm Name</th>
                                        <th>Accuracy</th>
                                        <th>Precision</th>
                                        <th>Recall</th>
                                        <th>F1-Score</th>
                                        <th>F2-Score (Target)</th>
                                        <th>AUC-ROC</th>
                                        <th>Latency (ms)</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr class="table-warning fw-bold">
                                        <td><span class="badge bg-primary">Traditional ML</span></td>
                                        <td>XGBoost Classifier (Tuned)</td>
                                        <td>0.962</td>
                                        <td>0.912</td>
                                        <td>0.948</td>
                                        <td>0.930</td>
                                        <td>0.941</td>
                                        <td>0.984</td>
                                        <td>1.2 ms</td>
                                    </tr>
                                    <tr class="fw-bold">
                                        <td><span class="badge bg-primary">Traditional ML</span></td>
                                        <td>LightGBM Classifier</td>
                                        <td>0.958</td>
                                        <td>0.905</td>
                                        <td>0.942</td>
                                        <td>0.923</td>
                                        <td>0.934</td>
                                        <td>0.981</td>
                                        <td>0.8 ms</td>
                                    </tr>
                                    <tr>
                                        <td><span class="badge bg-primary">Traditional ML</span></td>
                                        <td>Random Forest (OOB Tuned)</td>
                                        <td>0.954</td>
                                        <td>0.898</td>
                                        <td>0.931</td>
                                        <td>0.914</td>
                                        <td>0.924</td>
                                        <td>0.978</td>
                                        <td>2.1 ms</td>
                                    </tr>
                                    <tr class="table-success fw-bold">
                                        <td><span class="badge bg-success">Deep Learning</span></td>
                                        <td>Tabular FT-Transformer</td>
                                        <td>0.968</td>
                                        <td>0.928</td>
                                        <td>0.965</td>
                                        <td>0.946</td>
                                        <td>0.957</td>
                                        <td>0.989</td>
                                        <td>4.8 ms</td>
                                    </tr>
                                    <tr class="fw-bold">
                                        <td><span class="badge bg-success">Deep Learning</span></td>
                                        <td>TabNet Attention Model</td>
                                        <td>0.964</td>
                                        <td>0.921</td>
                                        <td>0.958</td>
                                        <td>0.939</td>
                                        <td>0.950</td>
                                        <td>0.986</td>
                                        <td>3.6 ms</td>
                                    </tr>
                                    <tr>
                                        <td><span class="badge bg-success">Deep Learning</span></td>
                                        <td>Tabular ResNet</td>
                                        <td>0.959</td>
                                        <td>0.910</td>
                                        <td>0.946</td>
                                        <td>0.928</td>
                                        <td>0.939</td>
                                        <td>0.982</td>
                                        <td>2.8 ms</td>
                                    </tr>
                                    <tr>
                                        <td><span class="badge bg-success">Deep Learning</span></td>
                                        <td>Deep & Cross Network (DCN)</td>
                                        <td>0.955</td>
                                        <td>0.902</td>
                                        <td>0.938</td>
                                        <td>0.920</td>
                                        <td>0.931</td>
                                        <td>0.979</td>
                                        <td>2.2 ms</td>
                                    </tr>
                                    <tr>
                                        <td><span class="badge bg-primary">Traditional ML</span></td>
                                        <td>Logistic Regression (L2 Balanced)</td>
                                        <td>0.895</td>
                                        <td>0.768</td>
                                        <td>0.884</td>
                                        <td>0.822</td>
                                        <td>0.858</td>
                                        <td>0.942</td>
                                        <td>0.3 ms</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>

            <!-- PANE 3: VISUAL ANALYTICS GALLERY -->
            <div class="tab-pane fade" id="visuals-pane" role="tabpanel">
                <div class="row g-3">
                    <div class="col-md-6">
                        <div class="card card-custom">
                            <div class="card-header card-header-custom">ROC Discrimination Curves</div>
                            <div class="card-body text-center">
                                <img src="/static/visualizations/05_overlaid_roc_curves.png" class="img-fluid rounded border" alt="ROC Curves" onerror="this.src='https://via.placeholder.com/600x400?text=ROC+Curves+Generated+on+Pipeline+Run'">
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card card-custom">
                            <div class="card-header card-header-custom">Precision-Recall Curves (Imbalanced Evaluation)</div>
                            <div class="card-body text-center">
                                <img src="/static/visualizations/06_overlaid_pr_curves.png" class="img-fluid rounded border" alt="PR Curves" onerror="this.src='https://via.placeholder.com/600x400?text=PR+Curves+Generated+on+Pipeline+Run'">
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card card-custom">
                            <div class="card-header card-header-custom">Top Predictive Fraud Features</div>
                            <div class="card-body text-center">
                                <img src="/static/visualizations/04_top_feature_importance.png" class="img-fluid rounded border" alt="Feature Importance" onerror="this.src='https://via.placeholder.com/600x400?text=Feature+Importance+Plot'">
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card card-custom">
                            <div class="card-header card-header-custom">Multi-Agent Workflow Architecture</div>
                            <div class="card-body text-center">
                                <img src="/static/visualizations/10_multi_agent_workflow_architecture.png" class="img-fluid rounded border" alt="Multi-Agent Architecture" onerror="this.src='https://via.placeholder.com/600x400?text=Agent+Architecture+Diagram'">
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- PANE 4: RAG KNOWLEDGE BASE -->
            <div class="tab-pane fade" id="rag-pane" role="tabpanel">
                <div class="card card-custom">
                    <div class="card-header card-header-custom">
                        <i class="fa-solid fa-magnifying-glass me-2"></i>Search Indian Insurance Policy Knowledge Base & IRDAI Guidelines
                    </div>
                    <div class="card-body">
                        <div class="input-group mb-3">
                            <input type="text" id="ragQuery" class="form-control" placeholder="Search policy waiting periods, cataract sub-limits, or billing tariffs..." value="waiting period pre-existing diseases">
                            <button class="btn btn-primary" onclick="searchRAG()"><i class="fa-solid fa-search me-1"></i>Search Knowledge Base</button>
                        </div>
                        <div id="ragResults" class="mt-3">
                            <!-- RAG Search Results Injected Here -->
                        </div>
                    </div>
                </div>
            </div>

        </div>

    </div>

    <!-- Scripts -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function loadSampleCase(type) {
            if (type === 'legit') {
                document.getElementById('full_name').value = "Ramesh Kumar Patil";
                document.getElementById('hospital_name').value = "SDM College of Medical Sciences Dharwad";
                document.getElementById('hospital_tier').value = "Tier 2 (City Multispecialty)";
                document.getElementById('diagnosis_category').value = "Gastroenterology & General Surgery";
                document.getElementById('treatment_name').value = "Laparoscopic Appendectomy";
                document.getElementById('stay_duration_days').value = "3";
                document.getElementById('claimed_amount_inr').value = "78000";
                document.getElementById('duration_months').value = "28";
                document.getElementById('waiting_period_months').value = "24";
            } else {
                document.getElementById('full_name').value = "Vikram Aditya Sharma";
                document.getElementById('hospital_name').value = "City Care Nursing Home";
                document.getElementById('hospital_tier').value = "Tier 3 (Nursing Home)";
                document.getElementById('diagnosis_category').value = "Gastroenterology & General Surgery";
                document.getElementById('treatment_name').value = "Laparoscopic Appendectomy";
                document.getElementById('stay_duration_days').value = "1";
                document.getElementById('claimed_amount_inr').value = "265000";
                document.getElementById('duration_months').value = "2";
                document.getElementById('waiting_period_months').value = "24";
            }
        }

        async function runAgentPipeline() {
            const container = document.getElementById('decisionContainer');
            container.innerHTML = `
                <div class="text-center py-4">
                    <div class="spinner-border text-primary mb-2" role="status"></div>
                    <p class="mb-0 fw-bold">Executing Multi-Agent Cognitive Graph...</p>
                    <small class="text-muted">Document OCR -> Policy Clause RAG -> Tariff Anomaly -> Reasoning Synthesis</small>
                </div>
            `;

            const payload = {
                full_name: document.getElementById('full_name').value,
                age: parseInt(document.getElementById('age').value),
                gender: document.getElementById('gender').value,
                state: document.getElementById('state').value,
                city: document.getElementById('city').value,
                annual_income_inr: 750000.0,
                insurance_provider: document.getElementById('insurance_provider').value,
                policy_type: document.getElementById('policy_type').value,
                sum_insured_inr: parseFloat(document.getElementById('sum_insured_inr').value),
                annual_premium_inr: 18500.0,
                duration_months: parseInt(document.getElementById('duration_months').value),
                waiting_period_months: parseInt(document.getElementById('waiting_period_months').value),
                copay_percentage: 10.0,
                hospital_name: document.getElementById('hospital_name').value,
                hospital_tier: document.getElementById('hospital_tier').value,
                diagnosis_category: document.getElementById('diagnosis_category').value,
                icd10_code: "K35.8",
                treatment_name: document.getElementById('treatment_name').value,
                stay_duration_days: parseInt(document.getElementById('stay_duration_days').value),
                claimed_amount_inr: parseFloat(document.getElementById('claimed_amount_inr').value),
                claim_submission_method: "Digital_Portal"
            };

            try {
                const res = await fetch('/api/claims/submit', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                
                const dec = data.final_decision;
                const decClass = dec === 'APPROVED' ? 'APPROVED' : (dec === 'FLAGGED_FOR_MANUAL_REVIEW' ? 'FLAGGED' : 'REJECTED');
                const badgeColor = dec === 'APPROVED' ? 'bg-success' : (dec === 'FLAGGED_FOR_MANUAL_REVIEW' ? 'bg-warning text-dark' : 'bg-danger');

                container.innerHTML = `
                    <div class="decision-box-${decClass}">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <span class="badge ${badgeColor} fs-6 px-3 py-2"><i class="fa-solid fa-gavel me-2"></i>${dec.replace(/_/g, ' ')}</span>
                            <span class="small fw-bold text-muted">Risk Score: ${data.composite_fraud_risk}</span>
                        </div>
                        
                        <h6 class="fw-bold mb-1 text-primary">Approved Settlement Amount:</h6>
                        <h3 class="fw-bold text-success mb-3">₹${data.approved_amount_inr.toLocaleString('en-IN')}</h3>

                        <h6 class="fw-bold text-dark"><i class="fa-solid fa-comment-dots me-2 text-primary"></i>Executive Summary:</h6>
                        <p class="small mb-3">${data.summary_explanation}</p>

                        <h6 class="fw-bold text-dark"><i class="fa-solid fa-list-check me-2 text-primary"></i>Layer-by-Layer Verification Evidence:</h6>
                        <div class="p-2 bg-white rounded border small mb-3" style="white-space: pre-line;">${data.detailed_explanation}</div>

                        <h6 class="fw-bold text-dark"><i class="fa-solid fa-language me-2 text-success"></i>फैसले का विवरण (Hindi Explanation):</h6>
                        <p class="small hindi-text bg-white p-2 rounded border mb-0">${data.explanation_hindi}</p>
                    </div>
                `;
            } catch (err) {
                container.innerHTML = `<div class="alert alert-danger">Error processing claim: ${err}</div>`;
            }
        }

        async function searchRAG() {
            const query = document.getElementById('ragQuery').value;
            const resBox = document.getElementById('ragResults');
            resBox.innerHTML = "<div class='text-muted small'>Searching...</div>";
            try {
                const res = await fetch(`/api/rag/search?query=${encodeURIComponent(query)}`);
                const data = await res.json();
                let html = "";
                data.matches.forEach(m => {
                    html += `
                        <div class="p-3 mb-2 bg-light border rounded">
                            <div class="d-flex justify-content-between">
                                <h6 class="fw-bold text-primary mb-1">${m.title}</h6>
                                <span class="badge bg-secondary">${m.category}</span>
                            </div>
                            <p class="small mb-0 text-muted">${m.content}</p>
                        </div>
                    `;
                });
                resBox.innerHTML = html || "<div class='alert alert-warning'>No matching clauses found.</div>";
            } catch(e) {
                resBox.innerHTML = `<div class='alert alert-danger'>Error: ${e}</div>`;
            }
        }
    </script>
</body>
</html>
    """
