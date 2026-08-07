# DATA DICTIONARY AND METADATA SPECIFICATION
**Project:** Medical Insurance Claim Fraud Detection System  
**Institution:** IIIT Dharwad, Department of Data Science and AI  
**Faculty Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

## 1. Overview of the Dataset
The dataset consists of **4500 medical insurance claim records** across **27 features**. 
It includes both raw attributes from hospital claim submissions and domain-specific enriched Indian healthcare features.

## 2. Feature Specification Table
| Feature Name | Data Type | Description | Valid Range / Categories | Relevance to Fraud Detection |
| :--- | :--- | :--- | :--- | :--- |
| **ClaimID** | String (UUID) | Unique identifier for each insurance claim submission. | Alphanumeric UUID string | Tracking and auditing identifier; prevents duplicate processing. |
| **PatientID** | String (UUID) | Unique identifier for the insured policyholder / patient. | Alphanumeric UUID string | Identifies repeat claimants and historical claim patterns. |
| **ProviderID** | String (UUID) | Unique identifier for the healthcare provider / hospital. | Alphanumeric UUID string | Identifies hospitals with unusually high claim rejection or fraud rates. |
| **ClaimAmount** | Float | Raw medical claim billing amount submitted for reimbursement. | 100.12 to 9,997.20 | Unusually high amounts or statistical outliers indicate possible billing inflation. |
| **ClaimAmountINR** | Float | Claim amount scaled to Indian Rupees (INR) representing realistic Indian hospital costs. | Rs. 2,503.00 to Rs. 2,49,930.00 | Benchmarks claim against typical Indian medical treatment cost structures. |
| **ClaimDate** | DateTime | Date when the claim was filed by the claimant. | 2024-01-01 to 2024-12-31 | Identifies temporal spikes, seasonality, or claims filed shortly after waiting periods. |
| **DiagnosisCode** | String | International diagnosis classification code (e.g., ICD-10 equivalent). | Alphanumeric code | Mismatches between diagnosis code and procedure indicate fraudulent billing. |
| **ProcedureCode** | String | Medical procedure or surgery code billed on the claim. | Alphanumeric code | Verified against treatment cost benchmarks and diagnosis compatibility. |
| **PatientAge** | Integer | Age of the policyholder / patient in years. | 0 to 100 years | Helps audit fairness across age groups (children, adults, elderly citizens). |
| **PatientGender** | String | Gender identity of the patient (M, F, Other). | 'M', 'F' | Audited for gender fairness and demographic neutrality in fraud scoring. |
| **ProviderSpecialty** | String | Medical department specialty of the provider. | Orthopedics, Cardiology, Neurology, Pediatrics, General Practice | Treatment specialty must align with diagnosis and procedure complexity. |
| **ClaimStatus** | String | Processing status of the claim at ingestion. | Pending, Approved, Rejected | Administrative context for claim disposition. |
| **PatientIncome** | Float | Annual declared income of the policyholder. | Numeric value | Financial context for sum insured and claim-to-income ratios. |
| **PatientIncomeINR** | Float | Annual income scaled to Indian Rupees (INR). | Numeric INR value | Helps assess policy premium affordability and claim proportionality. |
| **PatientMaritalStatus** | String | Marital status of the policyholder. | Single, Married, Divorced, Widowed | Demographic context for family floater coverage rules. |
| **PatientEmploymentStatus** | String | Employment status of the policyholder. | Employed, Self-Employed, Unemployed, Student, Retired | Explains employer group insurance eligibility and claim behavior. |
| **ProviderLocation** | String | Raw location name of the healthcare provider. | Simulated location names | Geographic reference for hospital cost analysis. |
| **IndianState** | String | Enriched Indian state where treatment occurred. | Maharashtra, Karnataka, Telangana, Delhi NCT, Tamil Nadu, etc. | Regional cost benchmark and geographic fraud pattern auditing. |
| **IndianCity** | String | Enriched Indian city / metro area of the hospital. | Mumbai, Bengaluru, Hyderabad, New Delhi, Chennai, Pune, etc. | Identifies city-level cost variations and metro vs. tier-3 disparities. |
| **PolicyType** | String | Indian insurance policy product structure. | Individual, Family Floater, Employer Group, Senior Citizen, Ayushman Bharat | Determines sub-limits, co-payments, and waiting period rules. |
| **InsurerCompany** | String | Indian general or health insurance company. | Star Health, ICICI Lombard, HDFC Ergo, New India Assurance, United India | Provides company-specific fraud rules and network hospital checks. |
| **HospitalTier** | String | Indian healthcare provider tier classification. | Tier-1 Metro Corporate, Tier-2 City Multi-Specialty, Tier-3 Town Nursing Home | Cost expectation baseline; Tier-3 billing Tier-1 prices is a key fraud signal. |
| **ClaimType** | String | Category of medical treatment. | Inpatient, Emergency, Routine, Outpatient | Inpatient and emergency claims have higher financial risk and fraud exposure. |
| **ClaimSubmissionMethod** | String | Mode of claim filing by the claimant or hospital. | Paper, Online, Phone | Online submissions can be cross-verified digitally; paper claims require OCR. |
| **Cluster** | Integer | Existing data clustering label from initial segmentation. | 0 to 4 | Structural grouping of similar claims. |
| **ClaimLegitimacy** | String (Target) | Original textual target variable indicating claim legitimacy. | 'Legitimate', 'Fraud' | Primary binary target for supervised machine learning and deep learning. |
| **IsFraud** | Integer (Target) | Binary numerical target variable (0 = Legitimate, 1 = Fraud). | 0, 1 | Machine-readable target for model training and loss evaluation. |

## 3. Class Distribution and Fraud Rate
- **Total Legitimate Claims:** 4230 (94.00%)
- **Total Fraudulent Claims:** 270 (6.00%)

This data dictionary is automatically maintained and verified during the execution pipeline.