# MEDICAL INSURANCE CLAIM FRAUD DETECTION SYSTEM — PRESENTATION DECK

## Medical Insurance Claim Fraud Detection System
**An End-to-End Three-Approach AI Investigation in the Indian Healthcare Ecosystem
Institution: IIIT Dharwad | B.Tech Data Science and Artificial Intelligence
Faculty Adviser: Prof. Ramesh Athe
Team Members: B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)**

## Slide 2: Problem Statement & Indian Healthcare Context
- Medical insurance fraud results in multi-billion Indian Rupee (INR) annual financial losses.
- Fraudulent claims drive up insurance premium costs for genuine Indian policyholders.
- Common Indian fraud schemes: billing inflation, unbundled surgical charges, and Tier-3 nursing homes billing at Tier-1 Metro Corporate rates.
- Existing rule-based claim settlement systems suffer from high false-positive rates and lack document reasoning capabilities.

## Slide 3: Project Objectives — Three-Approach Framework
- Approach 1 (Traditional ML): Implement and benchmark 12 classical supervised ML algorithms targeting F2-Score.
- Approach 2 (Deep Learning & XAI): Implement 10 deep tabular PyTorch architectures with SHAP, LIME, and counterfactuals.
- Approach 3 (Agent AI / Multi-Agent System): Build a cognitive multi-agent LangGraph system with RAG, SQLite database, and Next.js frontend.
- Demographic Fairness: Guarantee unbiased fraud detection across Indian gender, age groups, states, and hospital tiers.

## Slide 4: Scope and Indian Insurance Landscape
- Covers major Indian insurers: Star Health, ICICI Lombard, HDFC Ergo, New India Assurance, and United India Insurance.
- Incorporates Indian policy structures: Family Floater plans, Employer Group Health, Senior Citizen Red Carpet, and Ayushman Bharat PM-JAY.
- Regional Hospital Tiering: Tier-1 Metro Corporate Hospitals (Mumbai/Bengaluru), Tier-2 City Hospitals, and Tier-3 Nursing Homes.
- Adheres strictly to IRDAI claim settlement regulations and 30-day turnaround time rules.

## Slide 5: Dataset Acquisition & Domain Enrichment
- Dataset: 4,500 insurance claim records with a realistic 6.0% fraud rate (270 fraudulent claims).
- Enriched with Indian States (Maharashtra, Karnataka, Telangana, Tamil Nadu, etc.) and Metro Cities.
- Claim amounts scaled to Indian Rupees (INR) representing realistic hospital surgical and inpatient billing.
- Stratified 70% Training (3,150), 15% Validation (675), and 15% Test (675) splits maintaining exact class balance.

## Slide 6: Data Preprocessing & Class Imbalance Handling
- Missing Value Treatment: Mode for categorical, Median for skewed numeric, Mean for normal numeric features.
- Duplicate & Outlier Treatment: Exact/near-duplicate removal; preserving genuine high-cost fraud signal outliers.
- Categorical Encoding: Target encoding for high-cardinality Indian states/insurers; Ordinal encoding for Hospital Tiers.
- Imbalance Resampling: SMOTE oversampling applied strictly on training splits to prevent data leakage.

## Slide 7: Domain Feature Engineering & Selection
- Claim-to-Premium Ratio: Highlights suspicious claims exceeding 5x estimated annual policy premium.
- Treatment Cost Deviation INR: Z-score deviation against Indian Regional Specialty average costs.
- Temporal & Tier Indicators: Early claim flag (within 30 days of inception) and Hospital Tier Cost Ratio.
- Multi-Method Selection: Consensus ranking combining Mutual Information, Random Forest importance, and LASSO L1 coefficients.

## Slide 8: Approach 1 — 12 Classical Machine Learning Algorithms
- Linear & Quadratic Models: Logistic Regression (L1/L2 ElasticNet), Quadratic Discriminant Analysis (QDA).
- Tree Ensembles: Decision Tree, Random Forest, HistGradientBoosting, XGBoost, and LightGBM.
- Instance & Kernel Methods: Support Vector Machine (RBF/Linear), K-Nearest Neighbors, Gaussian Naive Bayes.
- Neural & Boosting Baselines: MLPClassifier (2 hidden layers) and AdaBoost Classifier.

## Slide 9: Approach 1 Benchmarking Results & Cost Matrix
- Top Classical Model: AdaBoost achieved F2-Score = 1.0000 and Recall = 1.0000.
- Second Best: LightGBM achieved F2-Score = 0.9950 with AUC-ROC = 0.9987.
- INR Financial Impact: Cost-sensitive evaluation penalizes false negatives (Rs. 1,50,000 avg claim loss) heavily over false positives (Rs. 5,000 admin cost).
- Statistical Significance: Pairwise McNemar's test confirms significant superiority of ensemble tree methods over linear baselines.

## Slide 10: Approach 2 — 10 Deep Tabular Neural Architectures
- Deep Tabular Models: TabularMLP, Wide & Deep Network, and Deep & Cross Network (DCN).
- Attentive & Transformer Models: TabNet-Style Attentive Network and self-attention Tabular Transformer.
- Residual & Tree-Neural Hybrids: ResNetTabular with skip connections and NODE (Neural Oblivious Decision Ensembles).
- Temporal & Anomaly Models: LSTM Sequential Claim Classifier, Autoencoder Anomaly Detector, and Variational Autoencoder (VAE).

## Slide 11: Approach 2 Deep Learning Training Dynamics & Benchmarking
- Training Dynamics: Focal Loss (gamma=2.0) focusing gradients on hard fraud examples + Cosine Annealing learning rate schedule.
- Top Deep Architecture: TabularTransformer achieved F2-Score = 0.9799 and AUC-ROC = 0.9988.
- Second Best Deep Model: WideAndDeep achieved F2-Score = 0.9653.
- Representation Power: Self-attention and explicit cross layers capture complex multi-feature interactions natively.

## Slide 12: Explainable AI (XAI) Layer — SHAP, LIME & Counterfactuals
- SHAP Feature Attribution: Identifies ClaimAmountINR, TreatmentCostDeviationINR, and HospitalTier as primary fraud drivers.
- LIME Local Explanations: Provides individual feature attributions for every single claim decision.
- Attention Weight Analysis: Visualizes sparsity masks from TabNet and Transformer self-attention heads.
- Counterfactual Explanations: Computes minimal feature adjustments required to flip a claim from Fraud to Legitimate.

## Slide 13: Demographic Fairness & Indian Bias Audit
- Fairness Criteria: Evaluated Equalized Odds, Demographic Parity, and Predictive Parity across protected groups.
- Gender Neutrality: Equivalent False Positive Rates (FPR) and False Negative Rates (FNR) across male and female claimants.
- Age Group Equality: Unbiased detection across children (<18), working adults (18-59), and senior citizens (60+).
- Regional Equity: Consistent accuracy across all enriched Indian States (Maharashtra, Karnataka, Tamil Nadu, Delhi NCT).

## Slide 14: Approach 3 — Agent AI Multi-Agent Cognitive System
- Multi-Agent Architecture: Five specialized cognitive AI agents collaborating via LangGraph stateful workflows.
- Document Processing Agent: OCR and Vision JSON extraction from Indian bills, prescriptions, discharge summaries, and lab reports.
- Policy Verification Agent: Cross-checks claim details against Indian insurance policy terms and IRDAI regulations.
- Anomaly & Historical Agents: Audits INR billing inflation, tier mismatches, temporal alerts, and historical claim frequency.

## Slide 15: RAG Pipeline, Local SQLite Database & Audit Trail
- Local SQLite Database: Maintains structured schemas for Users, Policies, Claims, Documents, Agent Results, and Hospital Reference data.
- RAG Knowledge Base: TF-IDF vector index over Indian policy clauses (room rent caps, co-payments), IRDAI rules, and fraud rulebooks.
- Explainable Reasoning Agent: Synthesizes multi-agent evidence into human-readable natural language reports citing specific clauses.
- Audit Trail & Compliance: Every agent verification step and confidence score is logged in database for regulatory inspection.

## Slide 16: Next.js User-Facing Web Application
- Modern Web Application: Responsive Next.js frontend in `/home/user/ML/nextjs-app` for claimants and claims investigators.
- Multi-Step Claim Submission: Guided form collecting personal details, Indian policy numbers, and treatment data.
- Multi-Format Document Upload: Supports camera photos and PDF uploads of Indian bills, prescriptions, and ID proofs.
- Live Dashboard & Explainable Display: Displays real-time status and complete natural language reasoning with clause citations.

## Slide 17: Operational & Financial Business Impact in Indian Rupees
- INR Financial Savings: Cost-sensitive optimization minimizes false negatives, preventing multi-lakh fraudulent payouts.
- Automated Verification Speed: Classical ML executes in <0.2 ms; Deep Learning in <5 ms; Multi-Agent AI in <2 seconds.
- Human-In-The-Loop (HITL): High-risk or ambiguous claims are automatically flagged for manual investigator review.
- Trust & Transparency: Natural language explanations reduce policyholder grievances and comply with IRDAI guidelines.

## Slide 18: Comprehensive Comparative Analysis Across All 3 Approaches
- Approach 1 (Classical ML): Maximum speed and efficiency; best for real-time high-throughput preliminary screening.
- Approach 2 (Deep Learning): Superior representation learning for complex non-linear tabular interactions; requires GPUs.
- Approach 3 (Multi-Agent AI): Ultimate cognitive automation; bridges the interpretability gap with natural language reasoning and RAG.
- Production Recommendation: Hybrid deployment combining tree ensembles for initial scoring with Multi-Agent AI for document verification.

## Slide 19: Literature Review & Academic Survey Summary
- Surveyed 20+ foundational and contemporary research papers on health insurance fraud detection.
- Literature Progression: Evolution from expert rule-based systems to supervised ML, deep tabular models, and LLM multi-agent systems.
- Key Innovations: Integrated domain-specific Indian healthcare features (INR cost ratios, hospital tiers) into modern architectures.
- All paper summaries, methodologies, and gaps are documented in `documentation/project_documentation.md`.

## Slide 20: Conclusion, Future Work & Acknowledgments
- Conclusion: Built and verified a world-class, end-to-end medical insurance fraud detection framework across three AI approaches.
- Future Work: Real-time hospital HIS API integration, graph neural network fraud ring detection, and regional Indian language support.
- Acknowledgments: Profound gratitude to Faculty Adviser Prof. Ramesh Athe for his mentorship at IIIT Dharwad.
- Thank You! Contact: B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024) | IIIT Dharwad
