# Data Dictionary: Indian Medical Insurance Claim Fraud Dataset

**Institution:** Indian Institute of Information Technology (IIIT), Dharwad  
**Faculty Adviser:** Ramesh Athe  
**Team Members:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

---

## 1. Raw and Synthetic Feature Definitions

| Feature Name | Data Type | Valid Range / Categories | Description | Fraud Relevance & Clinical Intuition |
|---|---|---|---|---|
| `Claim_ID` | String | `CLM-IND-10001` to `CLM-IND-99999` | Unique identifier for the medical insurance claim | Primary tracking identifier. |
| `Patient_ID` | String | `PAT-10000` to `PAT-99999` | Unique policyholder / patient identifier | Used to detect repeat claim bursts and multi-hospital fraud rings. |
| `Patient_Age` | Integer | 18 to 85 years | Age of the claimant in completed years | Detects age-treatment mismatch (e.g., pediatric treatments claimed for geriatric patients). |
| `Patient_Gender` | Categorical | `Male`, `Female`, `Other` | Self-reported gender of the patient | Used to monitor demographic parity and fairness across protected groups. |
| `Patient_State` | Categorical | 16 Major Indian States | State of primary residence in India | Detects geographical hopping (e.g., resident in Bihar claiming in high-tier Mumbai hospital). |
| `Patient_City` | Categorical | Metro, Tier-1, Tier-2, Tier-3 Cities | City of residence | Evaluates proximity to claimed healthcare provider. |
| `Annual_Income_INR` | Continuous (Float) | ₹1,80,000 to ₹50,00,000 | Annual household income in Indian Rupees (INR) | Cross-checked against policy sum insured and premium affordability. |
| `Insurance_Provider` | Categorical | 9 Insurers (Star Health, ICICI Lombard, HDFC ERGO, Ayushman Bharat, etc.) | Underwriting insurance company or government scheme | Identifies provider-specific claim fraud concentrations. |
| `Policy_Type` | Categorical | Individual, Family Floater, Corporate, Senior Citizen, Ayushman Bharat, etc. | Structure of insurance coverage | Family floaters and senior citizen plans show different fraud typologies. |
| `Sum_Insured_INR` | Continuous (Float) | ₹2,00,000 to ₹50,00,000 | Total maximum annual coverage limit in INR | Unusually high claims nearing 95%+ of sum insured are statistically flagged. |
| `Annual_Premium_INR` | Continuous (Float) | ₹5,000 to ₹1,50,000 | Annual premium paid for the policy in INR | Crucial for the `Claim_to_Premium_Ratio` domain feature. |
| `Policy_Duration_Months` | Integer | 1 to 120 months | Number of active months since policy inception | Detects early fraud: claims filed within 1–2 months of policy issuance. |
| `Waiting_Period_Months` | Integer | 1, 24, 36, 48 months | Mandatory waiting period for specific/pre-existing illnesses | Contractual baseline under IRDAI guidelines. |
| `Waiting_Period_Completed` | Binary (0 or 1) | 0 (No), 1 (Yes) | Status indicating if waiting period has elapsed | Claims for specified ailments filed prior to completion are contractual violations. |
| `Copay_Percentage` | Continuous (Float) | 0% to 25% | Mandatory out-of-pocket co-payment percentage | High co-pay reduces moral hazard; zero co-pay claims require deeper auditing. |
| `Hospital_Name` | Categorical | 15+ Named Indian Hospitals & Clinics | Name of admitting medical facility | Tracks hospital fraud collusion, blacklisted clinics, and repeat rejections. |
| `Hospital_Tier` | Categorical | `Tier 1 (Metro Super-Specialty)`, `Tier 2 (City Multispecialty)`, `Tier 3 (Nursing Home)` | Facility tier rating reflecting infrastructure & pricing | Critical for billing inflation detection (e.g., Tier 3 billing at Tier 1 rates). |
| `Diagnosis_Category` | Categorical | Cardiovascular, Orthopedics, Gastroenterology, Oncology, Nephrology, Ophthalmology, etc. | Broad medical specialty classification | Determines expected cost benchmarks and typical hospital stay length. |
| `ICD10_Diagnosis_Code` | Categorical (String) | Standard ICD-10 Codes (e.g., `I21.9`, `M17.9`, `K35.8`) | WHO standard diagnosis classification code | Medical coding standard used to cross-reference procedure legitimacy. |
| `Treatment_Name` | Categorical (String) | PTCA, Total Knee Replacement, Laparoscopic Appendectomy, Chemotherapy, etc. | Exact surgical or medical intervention conducted | Used for treatment-cost deviation modeling. |
| `Hospitalization_Duration_Days` | Integer | 0 to 30 days | Length of inpatient hospital stay | Disproportionately short or long stays relative to surgery type indicate phantom claims. |
| `Claim_Type` | Categorical | Hospitalization, DayCare, Outpatient, Pre-Post Hospitalization | Nature of the insurance claim filing | Daycare procedures (e.g., Cataract, Dialysis) have specific cost ceilings. |
| `Claim_Submission_Method` | Categorical | TPA Cashless, Reimbursement Paper, Digital Portal, Agent Assisted | Channel through which claim was lodged | Paper reimbursement claims exhibit 3.2x higher fraud incidence than cashless TPAs. |
| `Claim_Date` | Datetime | 2023-01-01 to 2026-08-01 | Date of claim submission | Used to compute temporal spikes and holiday/weekend claim bursts. |
| `Claim_Amount_INR` | Continuous (Float) | ₹5,000 to ₹45,00,000 | Total claimed reimbursement amount in INR | Primary monetary target for fraud inflation detection. |
| `Prior_Claims_Count` | Integer | 0 to 10 claims | Number of claims filed by policyholder in past 12 months | Detects velocity bursts and organized claimant syndicates. |
| `Total_Prior_Claimed_INR` | Continuous (Float) | ₹0 to ₹35,00,000 | Cumulative monetary amount claimed across policy lifetime | High cumulative claims relative to premium paid signify risk. |
| `Rejected_Prior_Claims` | Integer | 0 to 5 claims | Number of previously denied claims for this policyholder | Strong historical predictor of suspicious claimant behavior. |
| `Fraud_Pattern_Type` | Categorical (Metadata) | `Inflated_Billing`, `Phantom_Hospitalization`, `Upcoding_Procedure`, etc. | Specific fraud typology injected in ground truth | Used for multi-class fraud typology evaluation. |
| `Is_Fraud` | Binary Target (0 or 1) | `0` (Legitimate / Approve), `1` (Fraudulent / Reject) | Primary ground-truth classification target | Binary classification label across all ML, DL, and Agent models. |

---

## 2. Engineered Domain Features

| Engineered Feature | Mathematical Formula | Purpose & Fraud Indicator |
|---|---|---|
| `Claim_to_Premium_Ratio` | $\frac{\text{Claim\_Amount\_INR}}{\text{Annual\_Premium\_INR} + 1}$ | Captures claims exceeding multiple annual premiums; ratios $> 15$ trigger heightened scrutiny. |
| `Cost_Per_Day_INR` | $\frac{\text{Claim\_Amount\_INR}}{\text{Hospitalization\_Duration\_Days} + 1}$ | Detects inflated daily room and nursing rates exceeding IRDAI standard schedule of charges. |
| `Sum_Insured_Utilization` | $\frac{\text{Claim\_Amount\_INR}}{\text{Sum\_Insured\_INR}}$ | Values $> 0.90$ often indicate attempts to maximize policy payout before policy expiration. |
| `Treatment_Cost_Deviation` | $\frac{\text{Claim\_Amount\_INR} - \mu_{\text{tier, specialty}}}{\sigma_{\text{tier, specialty}} + \epsilon}$ | Z-score of claim amount relative to benchmark cost for specific diagnosis in the specific hospital tier. |
| `Claim_Velocity_Risk` | $\text{Prior\_Claims\_Count} \times \log(1 + \text{Total\_Prior\_Claimed\_INR})$ | Composite metric measuring claimant claiming frequency and monetary volume. |
| `Early_Claim_Flag` | $\mathbb{I}(\text{Policy\_Duration\_Months} \le \text{Waiting\_Period\_Months} + 2)$ | Binary indicator for claims lodged immediately upon waiting period expiry (moral hazard indicator). |
| `Prior_Rejection_Ratio` | $\frac{\text{Rejected\_Prior\_Claims}}{\text{Prior\_Claims\_Count} + 1}$ | Proportion of previously submitted claims that were rejected by auditors. |
| `Income_to_Claim_Ratio` | $\frac{\text{Claim\_Amount\_INR}}{\text{Annual\_Income\_INR} + 1}$ | Evaluates medical expense proportionality relative to claimant's socioeconomic profile. |
| `High_Risk_Submission_Method` | $\mathbb{I}(\text{Claim\_Submission\_Method} == \text{"Reimbursement\_Paper"})$ | Binary indicator for paper reimbursements which lack real-time hospital verification. |
