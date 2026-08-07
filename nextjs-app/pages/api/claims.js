// API Route: /api/claims
// Returns initial seed claims and in-memory submitted claims with explainable AI reasoning

let initialClaims = [
  {
    claimId: "CLM-IND-1001",
    policyNumber: "STAR-HLTH-2024-8871",
    patientName: "Rajesh Sharma",
    patientAge: 48,
    gender: "M",
    hospitalName: "Apollo Hospitals Navi Mumbai",
    hospitalTier: "Tier-1 Metro Corporate Hospital",
    treatmentType: "Inpatient",
    procedureCode: "IND-PROC-101",
    procedureName: "Laparoscopic Appendectomy",
    claimedAmountInr: 135000,
    approvedAmountInr: 135000,
    status: "APPROVED",
    confidenceScore: 0.96,
    submittedDate: "2024-07-15",
    executiveSummary: "Claim CLM-IND-1001 for Rs. 1,35,000.00 is APPROVED. All submitted medical documents are verified, the treatment cost aligns with Indian regional hospital benchmarks, and the claim complies fully with policy coverage terms.",
    detailedExplanation: `### EXPLAINABLE AI DECISION REPORT FOR CLAIM CLM-IND-1001
**Final Decision:** APPROVED (Confidence Score: 96.0%)  
**Claimed Reimbursement Amount:** Rs. 1,35,000.00  
**Approved Amount:** Rs. 1,35,000.00  

#### 1. Executive Summary
Claim CLM-IND-1001 for Rs. 1,35,000.00 is APPROVED. All submitted medical documents are verified, the treatment cost aligns with Indian regional hospital benchmarks, and the claim complies fully with policy coverage terms.

#### 2. Detailed Evidence & Verification Findings
**A. Document Processing Verification:**
- All medical bills, prescriptions, and discharge summaries were processed with an average OCR/Vision confidence of 96.2%.
- No billing unbundling or document fabrication anomalies detected.

**B. Indian Policy Coverage Compliance:**
- **[PASSED] Sum Insured Coverage Limit:** Claimed amount Rs. 1,35,000.00 is within Policy Sum Insured of Rs. 5,00,000.00.
- **[PASSED] Room Rent Sub-Limit Compliance:** Room rent billed (Rs. 18,000.00) complies with 1% daily cap. [CLAUSE-ROOM-001]

**C. Fraud Indicator & Cost Benchmark Analysis:**
- No billing inflation, tier mismatch, or temporal fraud anomalies detected. Treatment cost aligns with Indian Regional Tier benchmarks.

#### 3. Claimant & Regulatory Notice
In accordance with IRDAI Claim Settlement Guidelines [IRDAI-REG-101], policyholders have the right to request clarification or raise a grievance through the Insurance Grievance Redressal Mechanism within 30 days of this decision letter.`,
    evidenceCitations: ["[CLAUSE-ROOM-001] Room Rent Capping", "[IRDAI-REG-101] IRDAI Claim Turnaround"]
  },
  {
    claimId: "CLM-IND-1002",
    policyNumber: "ICICI-HLTH-2024-5542",
    patientName: "Sunita Verma",
    patientAge: 52,
    gender: "F",
    hospitalName: "Shree Krishna Nursing Home",
    hospitalTier: "Tier-3 Town Nursing Home",
    treatmentType: "Inpatient",
    procedureCode: "IND-PROC-101",
    procedureName: "Laparoscopic Appendectomy",
    claimedAmountInr: 245000,
    approvedAmountInr: 0,
    status: "REJECTED",
    confidenceScore: 0.94,
    submittedDate: "2024-07-20",
    executiveSummary: "Claim CLM-IND-1002 for Rs. 2,45,000.00 is REJECTED due to identified fraud indicators: Tier-3 Town Nursing Home billing corporate rate of Rs. 2,45,000.00; Claim amount is +430% above Tier-3 benchmark.",
    detailedExplanation: `### EXPLAINABLE AI DECISION REPORT FOR CLAIM CLM-IND-1002
**Final Decision:** REJECTED (Confidence Score: 94.0%)  
**Claimed Reimbursement Amount:** Rs. 2,45,000.00  
**Approved Amount:** Rs. 0.00  

#### 1. Executive Summary
Claim CLM-IND-1002 for Rs. 2,45,000.00 is REJECTED due to identified fraud indicators: Tier-3 Town Nursing Home billing corporate rate of Rs. 2,45,000.00; Claim amount is +430% above Tier-3 benchmark.

#### 2. Detailed Evidence & Verification Findings
**A. Document Processing Verification:**
- All medical bills, prescriptions, and discharge summaries were processed with an average OCR/Vision confidence of 94.1%.
- *Observation:* High consumables billing percentage (22.4% > 15% threshold).

**B. Indian Policy Coverage Compliance:**
- **[PASSED] Sum Insured Coverage Limit:** Claimed amount Rs. 2,45,000.00 is within Policy Sum Insured of Rs. 3,00,000.00.

**C. Fraud Indicator & Cost Benchmark Analysis:**
- **[CRITICAL] Billing Inflation & Cost Deviation:** Claim amount Rs. 2,45,000.00 is +430% above the typical Indian Tier-3 Town Nursing Home benchmark (Rs. 45,000.00) for Laparoscopic Appendectomy. [FRAUD-RULE-201]
- **[CRITICAL] Hospital Tier Pricing Mismatch:** Non-accredited Tier-3 Nursing Home charging corporate Tier-1 Metro prices without ICU infrastructure. [FRAUD-RULE-202]

#### 3. Claimant & Regulatory Notice
In accordance with IRDAI Claim Settlement Guidelines [IRDAI-REG-101], policyholders have the right to request clarification or raise a grievance through the Insurance Grievance Redressal Mechanism within 30 days of this decision letter.`,
    evidenceCitations: ["[FRAUD-RULE-201] Billing Inflation", "[FRAUD-RULE-202] Tier-3 Corporate Pricing"]
  },
  {
    claimId: "CLM-IND-1003",
    policyNumber: "HDFC-HLTH-2024-3310",
    patientName: "Vikramaditya Deshmukh",
    patientAge: 64,
    gender: "M",
    hospitalName: "Sahyadri Super Specialty Hospital",
    hospitalTier: "Tier-2 City Multi-Specialty Hospital",
    treatmentType: "Inpatient",
    procedureCode: "IND-PROC-102",
    procedureName: "Total Knee Replacement",
    claimedAmountInr: 235000,
    approvedAmountInr: 0,
    status: "FLAGGED FOR MANUAL REVIEW",
    confidenceScore: 0.88,
    submittedDate: "2024-07-28",
    executiveSummary: "Claim CLM-IND-1003 for Rs. 2,35,000.00 is FLAGGED FOR MANUAL REVIEW due to moderate cost deviation (+12% above Tier-2 benchmark) and Senior Citizen Co-Payment verification requirements.",
    detailedExplanation: `### EXPLAINABLE AI DECISION REPORT FOR CLAIM CLM-IND-1003
**Final Decision:** FLAGGED FOR MANUAL REVIEW (Confidence Score: 88.0%)  
**Claimed Reimbursement Amount:** Rs. 2,35,000.00  
**Approved Amount:** Pending investigator verification  

#### 1. Executive Summary
Claim CLM-IND-1003 for Rs. 2,35,000.00 is FLAGGED FOR MANUAL REVIEW due to moderate cost deviation (+12% above Tier-2 benchmark) and Senior Citizen Co-Payment verification requirements.

#### 2. Detailed Evidence & Verification Findings
**A. Document Processing Verification:**
- All medical bills, prescriptions, and discharge summaries were processed with an average OCR/Vision confidence of 95.8%.

**B. Indian Policy Coverage Compliance:**
- **[PASSED] Sum Insured Coverage Limit:** Claimed amount Rs. 2,35,000.00 is within Policy Sum Insured of Rs. 10,00,000.00.
- **[PASSED] Senior Citizen Co-Payment Rule:** Patient age 64 (>=60): mandatory 15% co-payment applies on final settlement. [CLAUSE-COPAY-003]

**C. Fraud Indicator & Cost Benchmark Analysis:**
- **[MEDIUM] Moderate Cost Deviation:** Claimed amount Rs. 2,35,000.00 is +12% higher than regional Tier-2 average (Rs. 2,10,000.00).

#### 3. Claimant & Regulatory Notice
In accordance with IRDAI Claim Settlement Guidelines [IRDAI-REG-101], policyholders have the right to request clarification or raise a grievance through the Insurance Grievance Redressal Mechanism within 30 days of this decision letter.`,
    evidenceCitations: ["[CLAUSE-COPAY-003] Senior Citizen Co-Payment", "[IRDAI-REG-101] IRDAI Turnaround Time"]
  }
];

export default function handler(req, res) {
  if (req.method === 'GET') {
    res.status(200).json({ claims: initialClaims });
  } else if (req.method === 'POST') {
    const newClaim = req.body;
    initialClaims = [newClaim, ...initialClaims];
    res.status(201).json({ claim: newClaim });
  } else {
    res.setHeader('Allow', ['GET', 'POST']);
    res.status(405).end(`Method ${req.method} Not Allowed`);
  }
}
