import { useState, useEffect } from 'react';

export default function Home() {
  const [activeTab, setActiveTab] = useState('submission');
  const [claims, setClaims] = useState([]);
  const [selectedClaim, setSelectedClaim] = useState(null);
  const [filterStatus, setFilterStatus] = useState('ALL');
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(1);

  // Form State
  const [formData, setFormData] = useState({
    policyNumber: 'STAR-HLTH-2024-8871',
    patientName: 'Rajesh Sharma',
    patientAge: '48',
    gender: 'M',
    contactNumber: '+91-9876543210',
    aadhaarNumber: '7845-1234-9012',
    hospitalName: 'Apollo Hospitals Navi Mumbai',
    hospitalTier: 'Tier-1 Metro Corporate Hospital',
    treatmentType: 'Inpatient',
    procedureCode: 'IND-PROC-101',
    procedureName: 'Laparoscopic Appendectomy',
    claimedAmountInr: '145000',
    daysSinceInception: '300',
    documentType: 'Hospital Bill',
    fileName: 'Apollo_Discharge_Bill_2024.pdf'
  });

  useEffect(() => {
    fetch('/api/claims')
      .then((res) => res.json())
      .then((data) => {
        if (data && data.claims) {
          setClaims(data.claims);
        }
      })
      .catch((err) => console.error("Failed loading claims:", err));
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleClaimSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch('/api/submit-claim', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      const data = await res.json();
      if (data && data.claim) {
        setClaims((prev) => [data.claim, ...prev]);
        setSelectedClaim(data.claim);
        setActiveTab('dashboard');
        setStep(1);
      }
    } catch (err) {
      console.error("Submission failed:", err);
      alert("Error submitting claim for verification.");
    } finally {
      setLoading(false);
    }
  };

  const filteredClaims = filterStatus === 'ALL'
    ? claims
    : claims.filter(c => c.status && c.status.includes(filterStatus));

  return (
    <div className="container">
      {/* HEADER */}
      <header>
        <div className="container header-content">
          <div className="header-title">
            <h1>Medical Insurance Claim Fraud Detection System</h1>
            <p>An End-to-End Three-Approach AI Investigation in the Indian Healthcare Ecosystem</p>
          </div>
          <div className="header-meta">
            <div><strong>Institution:</strong> IIIT Dharwad | B.Tech Data Science &amp; AI</div>
            <div><strong>Faculty Adviser:</strong> Prof. Ramesh Athe</div>
            <div><strong>Team:</strong> B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)</div>
          </div>
        </div>
      </header>

      {/* NAVIGATION TABS */}
      <div className="nav-tabs">
        <button
          className={`tab-btn ${activeTab === 'submission' ? 'active' : ''}`}
          onClick={() => setActiveTab('submission')}
        >
          New Claim Submission Workflow
        </button>
        <button
          className={`tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          Claim Status Dashboard &amp; Explainable AI
        </button>
        <button
          className={`tab-btn ${activeTab === 'benchmark' ? 'active' : ''}`}
          onClick={() => setActiveTab('benchmark')}
        >
          AI Approaches &amp; Model Benchmarking
        </button>
      </div>

      {/* TAB 1: NEW CLAIM SUBMISSION WORKFLOW */}
      {activeTab === 'submission' && (
        <div className="card">
          <h2 className="card-title">4-Step Guided Medical Insurance Claim Verification Wizard</h2>
          
          <div className="step-indicator">
            <div className={`step-item ${step >= 1 ? 'active' : ''}`}>
              <div className="step-num">1</div>
              <div className="step-label">Policyholder Details</div>
            </div>
            <div className={`step-item ${step >= 2 ? 'active' : ''}`}>
              <div className="step-num">2</div>
              <div className="step-label">Policy Coverage</div>
            </div>
            <div className={`step-item ${step >= 3 ? 'active' : ''}`}>
              <div className="step-num">3</div>
              <div className="step-label">Hospital &amp; Treatment</div>
            </div>
            <div className={`step-item ${step >= 4 ? 'active' : ''}`}>
              <div className="step-num">4</div>
              <div className="step-label">Document Upload &amp; AI Audit</div>
            </div>
          </div>

          <form onSubmit={handleClaimSubmit}>
            {step === 1 && (
              <div className="form-grid">
                <div className="form-group">
                  <label>Full Name of Insured Policyholder</label>
                  <input
                    type="text"
                    name="patientName"
                    value={formData.patientName}
                    onChange={handleInputChange}
                    className="form-input"
                    required
                  />
                </div>
                <div className="form-group">
                  <label>Patient Age (Years)</label>
                  <input
                    type="number"
                    name="patientAge"
                    value={formData.patientAge}
                    onChange={handleInputChange}
                    className="form-input"
                    required
                  />
                </div>
                <div className="form-group">
                  <label>Gender Identity</label>
                  <select
                    name="gender"
                    value={formData.gender}
                    onChange={handleInputChange}
                    className="form-select"
                  >
                    <option value="M">Male (M)</option>
                    <option value="F">Female (F)</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Contact Number (India)</label>
                  <input
                    type="text"
                    name="contactNumber"
                    value={formData.contactNumber}
                    onChange={handleInputChange}
                    className="form-input"
                  />
                </div>
                <div className="form-group">
                  <label>Aadhaar / Gov ID Verification Number</label>
                  <input
                    type="text"
                    name="aadhaarNumber"
                    value={formData.aadhaarNumber}
                    onChange={handleInputChange}
                    className="form-input"
                  />
                </div>
              </div>
            )}

            {step === 2 && (
              <div className="form-grid">
                <div className="form-group">
                  <label>Policy Reference Number</label>
                  <input
                    type="text"
                    name="policyNumber"
                    value={formData.policyNumber}
                    onChange={handleInputChange}
                    className="form-input"
                    required
                  />
                </div>
                <div className="form-group">
                  <label>Days Since Policy Inception</label>
                  <input
                    type="number"
                    name="daysSinceInception"
                    value={formData.daysSinceInception}
                    onChange={handleInputChange}
                    className="form-input"
                    required
                  />
                </div>
              </div>
            )}

            {step === 3 && (
              <div className="form-grid">
                <div className="form-group">
                  <label>Hospital / Healthcare Provider Name</label>
                  <input
                    type="text"
                    name="hospitalName"
                    value={formData.hospitalName}
                    onChange={handleInputChange}
                    className="form-input"
                    required
                  />
                </div>
                <div className="form-group">
                  <label>Indian Hospital Tier Classification</label>
                  <select
                    name="hospitalTier"
                    value={formData.hospitalTier}
                    onChange={handleInputChange}
                    className="form-select"
                  >
                    <option value="Tier-1 Metro Corporate Hospital">Tier-1 Metro Corporate Hospital</option>
                    <option value="Tier-2 City Multi-Specialty Hospital">Tier-2 City Multi-Specialty Hospital</option>
                    <option value="Tier-3 Town Nursing Home">Tier-3 Town Nursing Home</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Type of Treatment</label>
                  <select
                    name="treatmentType"
                    value={formData.treatmentType}
                    onChange={handleInputChange}
                    className="form-select"
                  >
                    <option value="Inpatient">Inpatient Hospitalization</option>
                    <option value="Emergency">Emergency Room</option>
                    <option value="Routine">Routine Day Care</option>
                    <option value="Outpatient">Outpatient Consultation</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Medical Procedure Billed</label>
                  <select
                    name="procedureName"
                    value={formData.procedureName}
                    onChange={handleInputChange}
                    className="form-select"
                  >
                    <option value="Laparoscopic Appendectomy">Laparoscopic Appendectomy (IND-PROC-101)</option>
                    <option value="Total Knee Replacement">Total Knee Replacement (IND-PROC-102)</option>
                    <option value="PTCA Coronary Angioplasty">PTCA Coronary Angioplasty (IND-PROC-103)</option>
                    <option value="Dengue Hemorrhagic Fever Management">Dengue Fever Management (IND-PROC-104)</option>
                    <option value="Cataract Surgery">Cataract Surgery (IND-PROC-105)</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Total Billed Amount in Indian Rupees (INR / Rs.)</label>
                  <input
                    type="number"
                    name="claimedAmountInr"
                    value={formData.claimedAmountInr}
                    onChange={handleInputChange}
                    className="form-input"
                    required
                  />
                </div>
              </div>
            )}

            {step === 4 && (
              <div>
                <div className="form-grid">
                  <div className="form-group">
                    <label>Primary Document Type</label>
                    <select
                      name="documentType"
                      value={formData.documentType}
                      onChange={handleInputChange}
                      className="form-select"
                    >
                      <option value="Hospital Bill">Hospital Itemized Bill &amp; Invoice</option>
                      <option value="Prescription">Doctor Prescription &amp; Pharmacy Bill</option>
                      <option value="Discharge Summary">Hospital Discharge Summary</option>
                      <option value="Lab Report">Pathology / Laboratory Test Report</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Selected Document File</label>
                    <input
                      type="text"
                      name="fileName"
                      value={formData.fileName}
                      onChange={handleInputChange}
                      className="form-input"
                    />
                  </div>
                </div>
                <div style={{ background: '#f8fafc', padding: '1.2rem', borderRadius: '6px', border: '1px dashed #cbd5e1', marginBottom: '1.5rem' }}>
                  <h4 style={{ marginBottom: '0.4rem', color: '#1b5e20' }}>Multi-Agent AI Document Verification Notice</h4>
                  <p style={{ fontSize: '0.9rem', color: '#475569' }}>
                    Upon submission, our <strong>DocumentProcessingAgent</strong> will extract itemized charges using OCR &amp; Vision Language Models.
                    The <strong>PolicyVerificationAgent</strong> will audit sub-limits against RAG policy clauses, and the <strong>AnomalyDetectionAgent</strong> will benchmark against Indian regional hospital tier pricing.
                  </p>
                </div>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1.5rem' }}>
              {step > 1 ? (
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setStep(step - 1)}
                >
                  &larr; Previous Step
                </button>
              ) : <div></div>}

              {step < 4 ? (
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => setStep(step + 1)}
                >
                  Next Step &rarr;
                </button>
              ) : (
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={loading}
                >
                  {loading ? 'Executing Multi-Agent Verification...' : 'Submit Claim & Run Multi-Agent Audit'}
                </button>
              )}
            </div>
          </form>
        </div>
      )}

      {/* TAB 2: CLAIM STATUS DASHBOARD */}
      {activeTab === 'dashboard' && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
            <h2 className="card-title" style={{ margin: 0, border: 'none' }}>Live Medical Insurance Claim Tracking Dashboard</h2>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 'bold' }}>Filter Status:</span>
              {['ALL', 'APPROVED', 'FLAGGED', 'REJECTED'].map((st) => (
                <button
                  key={st}
                  className={`btn ${filterStatus === st ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}
                  onClick={() => setFilterStatus(st)}
                >
                  {st}
                </button>
              ))}
            </div>
          </div>

          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Claim ID</th>
                  <th>Policy Number</th>
                  <th>Insured Patient</th>
                  <th>Hospital Provider</th>
                  <th>Claimed Amount (INR)</th>
                  <th>Approved (INR)</th>
                  <th>Status</th>
                  <th>Explainable AI Reasoning</th>
                </tr>
              </thead>
              <tbody>
                {filteredClaims.map((c) => (
                  <tr key={c.claimId}>
                    <td><strong>{c.claimId}</strong></td>
                    <td>{c.policyNumber}</td>
                    <td>{c.patientName} ({c.patientAge}y, {c.gender})</td>
                    <td>{c.hospitalName}<br/><small style={{ color: '#64748b' }}>{c.hospitalTier}</small></td>
                    <td>Rs. {c.claimedAmountInr?.toLocaleString('en-IN')}.00</td>
                    <td>Rs. {c.approvedAmountInr?.toLocaleString('en-IN')}.00</td>
                    <td>
                      <span className={`badge ${
                        c.status === 'APPROVED' ? 'badge-approved' :
                        c.status === 'REJECTED' ? 'badge-rejected' : 'badge-flagged'
                      }`}>
                        {c.status}
                      </span>
                    </td>
                    <td>
                      <button
                        className="btn btn-secondary"
                        style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}
                        onClick={() => setSelectedClaim(c)}
                      >
                        View AI Explanation
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 3: MODEL BENCHMARKING EXPLORER */}
      {activeTab === 'benchmark' && (
        <div className="card">
          <h2 className="card-title">Multi-Approach Algorithm Benchmarking &amp; Financial Impact Explorer</h2>
          <p style={{ marginBottom: '1.5rem', color: '#475569' }}>
            Comparison of all 12 Classical Machine Learning Algorithms (Approach 1), 10 Deep Tabular Neural Network Architectures (Approach 2),
            and our cognitive Agent AI Multi-Agent System (Approach 3) on the Indian health insurance dataset (4,500 claims, 6.0% fraud rate).
          </p>

          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Algorithm Name</th>
                  <th>Approach Pillar</th>
                  <th>F2 Score</th>
                  <th>Recall</th>
                  <th>Precision</th>
                  <th>AUC-ROC</th>
                  <th>Latency (ms)</th>
                  <th>Total Financial Cost (INR)</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>1</strong></td>
                  <td><strong>Agent AI Multi-Agent System</strong></td>
                  <td><span className="badge badge-approved">Approach 3 (Agent AI)</span></td>
                  <td><strong>1.0000</strong></td>
                  <td>1.0000</td>
                  <td>1.0000</td>
                  <td>1.0000</td>
                  <td>1250.0</td>
                  <td><strong>Rs. 0 (100% verified)</strong></td>
                </tr>
                <tr>
                  <td><strong>2</strong></td>
                  <td><strong>AdaBoost Classifier</strong></td>
                  <td><span className="badge" style={{ background: '#e0e7ff', color: '#3730a3' }}>Approach 1 (Classical ML)</span></td>
                  <td><strong>1.0000</strong></td>
                  <td>1.0000</td>
                  <td>1.0000</td>
                  <td>1.0000</td>
                  <td>0.280</td>
                  <td><strong>Rs. 0</strong></td>
                </tr>
                <tr>
                  <td><strong>3</strong></td>
                  <td><strong>LightGBM Classifier</strong></td>
                  <td><span className="badge" style={{ background: '#e0e7ff', color: '#3730a3' }}>Approach 1 (Classical ML)</span></td>
                  <td><strong>0.9950</strong></td>
                  <td>1.0000</td>
                  <td>0.9756</td>
                  <td>0.9998</td>
                  <td>0.080</td>
                  <td><strong>Rs. 5,000</strong></td>
                </tr>
                <tr>
                  <td><strong>4</strong></td>
                  <td><strong>XGBoost Classifier</strong></td>
                  <td><span className="badge" style={{ background: '#e0e7ff', color: '#3730a3' }}>Approach 1 (Classical ML)</span></td>
                  <td><strong>0.9901</strong></td>
                  <td>1.0000</td>
                  <td>0.9524</td>
                  <td>0.9998</td>
                  <td>0.030</td>
                  <td><strong>Rs. 10,000</strong></td>
                </tr>
                <tr>
                  <td><strong>5</strong></td>
                  <td><strong>Tabular Transformer (Self-Attention)</strong></td>
                  <td><span className="badge" style={{ background: '#f3e8ff', color: '#6b21a8' }}>Approach 2 (Deep Learning)</span></td>
                  <td><strong>0.9799</strong></td>
                  <td>0.9750</td>
                  <td>1.0000</td>
                  <td>0.9994</td>
                  <td>3.450</td>
                  <td><strong>Rs. 1,50,000</strong></td>
                </tr>
                <tr>
                  <td><strong>6</strong></td>
                  <td><strong>HistGradientBoosting Classifier</strong></td>
                  <td><span className="badge" style={{ background: '#e0e7ff', color: '#3730a3' }}>Approach 1 (Classical ML)</span></td>
                  <td><strong>0.9799</strong></td>
                  <td>0.9750</td>
                  <td>1.0000</td>
                  <td>0.9999</td>
                  <td>0.950</td>
                  <td><strong>Rs. 1,50,000</strong></td>
                </tr>
                <tr>
                  <td><strong>7</strong></td>
                  <td><strong>Random Forest Classifier</strong></td>
                  <td><span className="badge" style={{ background: '#e0e7ff', color: '#3730a3' }}>Approach 1 (Classical ML)</span></td>
                  <td><strong>0.9750</strong></td>
                  <td>0.9750</td>
                  <td>0.9750</td>
                  <td>1.0000</td>
                  <td>0.310</td>
                  <td><strong>Rs. 155,000</strong></td>
                </tr>
                <tr>
                  <td><strong>8</strong></td>
                  <td><strong>TabNet-Style Attentive Network</strong></td>
                  <td><span className="badge" style={{ background: '#f3e8ff', color: '#6b21a8' }}>Approach 2 (Deep Learning)</span></td>
                  <td><strong>0.9466</strong></td>
                  <td>0.9750</td>
                  <td>0.8478</td>
                  <td>0.9956</td>
                  <td>1.850</td>
                  <td><strong>Rs. 1,85,000</strong></td>
                </tr>
                <tr>
                  <td><strong>9</strong></td>
                  <td><strong>Deep &amp; Cross Network (DCN)</strong></td>
                  <td><span className="badge" style={{ background: '#f3e8ff', color: '#6b21a8' }}>Approach 2 (Deep Learning)</span></td>
                  <td><strong>0.9375</strong></td>
                  <td>0.9750</td>
                  <td>0.8125</td>
                  <td>0.9961</td>
                  <td>1.430</td>
                  <td><strong>Rs. 1,95,000</strong></td>
                </tr>
                <tr>
                  <td><strong>10</strong></td>
                  <td><strong>ResNet for Tabular Data</strong></td>
                  <td><span className="badge" style={{ background: '#f3e8ff', color: '#6b21a8' }}>Approach 2 (Deep Learning)</span></td>
                  <td><strong>0.9314</strong></td>
                  <td>0.9500</td>
                  <td>0.8636</td>
                  <td>0.9965</td>
                  <td>2.100</td>
                  <td><strong>Rs. 3,30,000</strong></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* EXPLAINABLE AI DECISION MODAL */}
      {selectedClaim && (
        <div className="modal-overlay" onClick={() => setSelectedClaim(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelectedClaim(null)}>&times;</button>
            <h3 style={{ fontSize: '1.3rem', color: '#1b5e20', marginBottom: '0.5rem' }}>
              Explainable AI (XAI) Claim Decision Report
            </h3>
            <div style={{ marginBottom: '1.2rem', fontSize: '0.9rem', color: '#475569' }}>
              Claim ID: <strong>{selectedClaim.claimId}</strong> | Patient: <strong>{selectedClaim.patientName}</strong> | Hospital: <strong>{selectedClaim.hospitalName}</strong>
            </div>

            <div style={{
              padding: '1.25rem',
              borderRadius: '6px',
              backgroundColor: selectedClaim.status === 'APPROVED' ? '#f0fdf4' :
                               selectedClaim.status === 'REJECTED' ? '#fef2f2' : '#fefce8',
              border: `1px solid ${selectedClaim.status === 'APPROVED' ? '#bbf7d0' :
                                   selectedClaim.status === 'REJECTED' ? '#fecaca' : '#fef08a'}`,
              marginBottom: '1.5rem'
            }}>
              <h4 style={{
                color: selectedClaim.status === 'APPROVED' ? '#166534' :
                       selectedClaim.status === 'REJECTED' ? '#991b1b' : '#854d0e',
                marginBottom: '0.5rem'
              }}>
                Verdict: {selectedClaim.status} (Confidence Score: {(selectedClaim.confidenceScore * 100).toFixed(1)}%)
              </h4>
              <p style={{ fontSize: '0.95rem', fontWeight: 500 }}>
                {selectedClaim.executiveSummary}
              </p>
            </div>

            <div style={{
              whiteSpace: 'pre-line',
              fontSize: '0.9rem',
              color: '#1e293b',
              lineHeight: 1.6,
              background: '#f8fafc',
              padding: '1.25rem',
              borderRadius: '6px',
              border: '1px solid #cbd5e1'
            }}>
              {selectedClaim.detailedExplanation}
            </div>

            {selectedClaim.evidenceCitations && selectedClaim.evidenceCitations.length > 0 && (
              <div style={{ marginTop: '1.25rem' }}>
                <h5 style={{ fontSize: '0.85rem', color: '#475569', marginBottom: '0.4rem', textTransform: 'uppercase' }}>
                  Policy Clauses &amp; Regulatory Citations:
                </h5>
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                  {selectedClaim.evidenceCitations.map((cit, idx) => (
                    <span key={idx} style={{
                      background: '#e2e8f0', color: '#334155', fontSize: '0.8rem',
                      padding: '0.3rem 0.6rem', borderRadius: '4px', fontWeight: 600
                    }}>
                      {cit}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
