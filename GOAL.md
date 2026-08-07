# Medical Insurance Claim Fraud Detection — Delivery Charter

**Institution:** IIIT Dharwad, Department of Data Science and AI  
**Faculty Adviser:** Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  
**Date:** 7 August 2026  
**Project form:** Three complementary, reproducible approaches to explainable medical-insurance claim fraud detection in India.

---

## 1. Definition of Done

The completed repository is a submission-ready academic project, rather than a collection of notebooks or unverified claims. It must be possible for an evaluator to create an environment from the pinned requirements, run a documented command, reproduce datasets and data splits, train/evaluate the applicable models, inspect evidence and decisions, and open the accompanying presentation and research report.

A claim outcome has an operational meaning throughout the project:

- **Legitimate / approve:** the model estimates no fraud signal at the chosen operational threshold. This is a screening recommendation, not an automatic insurance settlement.
- **Fraudulent / reject:** the model estimates material fraud risk and must emit understandable contributing factors. It is a high-risk recommendation requiring insurer review, not a replacement for due process.
- **Flagged (Agent AI):** evidence is incomplete, contradictory, or risk is sufficiently high that a human reviewer must decide.

All components use Indian healthcare and insurance terminology: rupee-denominated amounts; private insurers (Star Health, ICICI Lombard, HDFC ERGO and New India Assurance); government schemes (Ayushman Bharat and ECHS); individual, family-floater and group policies; allopathic and Ayurvedic care; Aadhaar/PAN-aware document handling; and geographically variable provider costs. Protected characteristics are used only for post-hoc fairness audit, never as a direct decision rule in a deployed scoring interface.

---

## 2. Repository Read-Through and Dataset Decision

The initial repository read-through covered every pre-existing file:

- `Health Insurance Fraud Claims.xlsx`: 4,500 claim records plus a header (19 columns). It has a binary `ClaimLegitimacy` label but contains generic/non-Indian provider locations and identifiers, claim amounts in an unspecified currency scale, and lacks essential policy, temporal and historical fraud fields.
- `Mini Project.pdf`: a two-page project statement defining end-to-end intake, document intelligence, RAG, multi-agent decisioning and explainability.
- `# Medical Insurance Claim Fraud Det.txt`, all three approach prompts, their duplicate variants, explainability prompt, and the three recordings: detailed specifications for the traditional-ML, deep-learning and Agent-AI tracks.
- `README.md`: only the original repository title.

The supplied workbook will be preserved unchanged under `data/raw/` and profiled in the data documentation. It fails the stated adequacy threshold (4,500 rather than at least 10,000 claims) and is not Indian-context complete. The reproducible fallback is therefore a **clearly labelled synthetic Indian claim population** created from transparent domain assumptions. It will contain at least 12,000 claims, a 5–15% fraud prevalence, right-skewed INR claim amounts, deliberate but auditable fraud signals, Indian states/cities/providers/policies, and all required policy, temporal, provider and claim-history features. The original workbook remains an audit/reference source; it is not silently represented as Indian real-world data.

---

## 3. Shared Engineering and Research Deliverables

### 3.1 Reproducible foundation

- [ ] A clear `README.md`, pinned `requirements.txt`, `.env.example`, `.gitignore`, configuration files and one-command entry points.
- [ ] Immutable raw-source manifest with SHA-256 checksum, data provenance, adequacy assessment and privacy statement.
- [ ] Reproducible synthetic-data generator and data dictionary with feature type, valid values/range, origin and fraud relevance.
- [ ] Fixed random seeds, structured logging, robust input/schema validation and safe artifact paths.
- [ ] Deterministic 70% / 15% / 15% stratified train/validation/test split used by ML and DL approaches.
- [ ] No data leakage: transformations, encoders, sampling, calibration and threshold choice are fitted from training/validation data only; the test set is untouched until final evaluation.
- [ ] Generated assets organised below `data/`, `models/`, `evaluation/`, `documentation/`, `visualizations/`, `presentation/` and `reports/`.

### 3.2 Responsible use requirements

- [ ] Use synthetic or de-identified data only; never commit Aadhaar/PAN values, API keys, patient images or personally identifying claimant information.
- [ ] Score recommendations must carry a human-review disclaimer and never become an automatic denial.
- [ ] Report per-group accuracy, FPR, FNR, precision/recall and selection rate across gender, age bracket, geography, income bracket and treatment type.
- [ ] Investigate material disparities; compare a mitigation alternative if a predefined disparity guardrail is crossed; document the accuracy/fairness trade-off.
- [ ] Every rejection/flag recommendation receives model-appropriate explanations and evidence limitations.

### 3.3 Academic communications

- [ ] Technical diagrams are created as local assets and embedded in relevant Markdown documents.
- [ ] Evaluation claims are generated from actual result artifacts rather than invented values.
- [ ] Each approach receives a professional ~20-slide `.pptx` deck and an IEEE-style `.pdf` report with IIIT Dharwad affiliation, team, Ramesh Athe acknowledgement, sufficient figures/tables, and verified citations.
- [ ] The final comparison names only comparable metrics and explicitly separates tabular classification performance from Agent-AI workflow/document evidence quality.

---

## 4. Approach 1 — Traditional Machine Learning Baseline

### Done means

Approach 1 is complete when a configurable pipeline creates/loads the selected data, verifies the schema, engineers domain features, saves train/validation/test datasets, fits serializable preprocessing and model artifacts, tunes and evaluates the full agreed model suite, selects a validation-derived probability threshold, and writes all metrics/plots/reports from the actual run.

The minimum evaluated classifiers are Logistic Regression (L1/L2), Decision Tree, Random Forest, HistGradientBoosting, XGBoost when installed, LightGBM when installed, linear/RBF SVM, KNN, Gaussian and Multinomial-compatible Naive Bayes variants, shallow MLP, AdaBoost and QDA. Optional dependencies must degrade gracefully and be disclosed; a missing optional library is never fabricated as a completed benchmark.

The approach includes missingness/duplicate/outlier audits; train-only imputation/encoding/scaling; comparison of legal imbalance strategies (class weights, random undersampling, SMOTE/Tomek/SMOTEENN where `imbalanced-learn` is present); limited domain and interaction features; correlation/MI/RFE/embedded selection evidence; F2-first tuning; calibrated/thresholded held-out evaluation; costs in INR; McNemar/Wilcoxon tests where their assumptions are meaningful; SHAP/permutation/tree/coefficient explanations; and a fairness analysis.

### Deliverables

- [ ] `src/` modules for configuration, loading, synthesis, validation, preprocessing, feature engineering, modelling, training, evaluation, visualization and utilities, all typed and documented.
- [ ] Raw/processed/synthetic data, split metadata, dictionary and data-quality reports.
- [ ] Saved preprocessing pipeline, selected models and machine-readable metadata per trained model.
- [ ] Actual benchmarks: accuracy, precision, recall, F1, F2, ROC-AUC, PR-AUC, MCC, threshold, confusion matrix, INR cost, train time, inference latency, artifact size and tuned-parameter count.
- [ ] EDA, relationship, feature-importance, ROC/PR, comparison, calibration, fairness and computational-efficiency visualizations.
- [ ] Comprehensive methodology/evaluation documents, including results limitations and reproducibility instructions.
- [ ] A 20-slide traditional-ML deck and IEEE-format traditional-ML report PDF.
- [ ] An end-to-end verification record confirming artifact existence and that report/deck numbers match the latest run.

---

## 5. Approach 2 — Deep Learning on the Same Split

### Done means

Approach 2 reuses the exact frozen entity/split definition from Approach 1. A unified PyTorch interface prepares numeric inputs with train-only standardisation and categorical embedding vocabularies, uses class-aware sampling/losses, records reproducible training runs, and evaluates on precisely the same held-out test claims. It must clearly distinguish implemented/reproducibly runnable architectures from future-work designs.

The target architecture suite is MLP, Wide & Deep, Deep & Cross Network, TabNet-style attentive tabular model, feature-token Transformer, tabular ResNet, NODE-inspired differentiable-oblivious-tree model, attention LSTM over policyholder sequences, legitimate-only Autoencoder anomaly detector and a VAE. Training supports weighted BCE/focal loss, early stopping, Adam/AdamW, clipping, dropout/norms, checkpoint recovery, learning curves and seed/stability runs. Architecture-specific dependencies not available in a fresh environment must be replaced with self-contained implementations or stated accurately.

### Deliverables

- [ ] DL configuration YAMLs and one unified data/training/evaluation command.
- [ ] Common model interface plus individually testable architecture implementations.
- [ ] Checkpoints, exact experiment configurations, per-epoch histories and device/resource logs.
- [ ] Actual DL benchmark, bootstrap intervals, calibration/Brier/ECE, ablations, robustness checks and fairness analysis.
- [ ] Architecture, training dynamics, embedding, attention/importance, calibration and ML-vs-DL figures.
- [ ] Deep-learning documentation and detailed evaluation analysis grounded in the generated outputs.
- [ ] A ~20-slide DL deck and IEEE-style DL report PDF.
- [ ] Verification record that metric claims use the frozen common test set and match files.

---

## 6. Approach 3 — Evidence-Grounded Agent AI Claim Workflow

### Done means

Approach 3 is a safe, demonstrable local claim-assessment system—not a promise of an autonomous insurer. It contains a local database with seeded synthetic claimant/policy/claim/provider/medical/fraud-reference records; a local retriever over versioned policy, medical-cost, fraud-rule and regulatory reference documents; typed inter-agent messages; an auditable workflow; and a role-aware frontend/API or reproducible demo interface.

Its coordinator orchestrates five specialist stages: document extraction/quality gating, policy verification with clause citations, anomaly detection using regional/tier baselines, historical pattern analysis, and transparent final reasoning. Document processing and policy verification may execute independently; anomaly analysis waits for both. A missing document, a deterministic coverage violation, low-confidence extraction, a failed tool call and a fraud flag all follow explicit state transitions. Fraud flags route to human review; rejection recommendations explain evidence, uncertainty, next actions and grievance route.

The system must operate in deterministic offline demo mode without an API key. Gemini/LangChain/LangGraph adapters are optional integrations, enabled only from environment variables; they must implement redaction, schema validation, retries/backoff, rate/cost logs and never reveal secrets. No real Aadhaar/PAN data, real medical documentation or live production claim decisioning is included.

### Deliverables

- [ ] Database schema/migrations, safely seeded demo data and indexes.
- [ ] Versioned RAG corpus, chunk manifest, retrieval tests and cited sources.
- [ ] Typed state/messages, tools, agent prompts/adapters, deterministic fallback agents and workflow/error/human-review tests.
- [ ] Claim-submission, status, reviewer and decision-explanation interface/API design with role/accessibility/localisation considerations.
- [ ] Workflow diagram, system architecture, ER diagram, RAG diagram and demo evidence.
- [ ] Agent quality/evidence-completeness/latency/cost analysis and scenario-based fairness/safety review.
- [ ] Agent-AI documentation, deck and IEEE-style report PDF, all separating demo evidence from unvalidated production claims.
- [ ] Verification record for a complete approved, flagged and rejected synthetic scenario.

---

## 7. Final Cross-Approach Comparison

- [ ] One explicit shared claim-population/split protocol for ML and DL; Agent-AI is evaluated on scenario/workflow evidence and is not misrepresented as a directly trained classifier unless a separately defined test is run.
- [ ] Comparison of accuracy, precision, recall, F1, F2, ROC-AUC, PR-AUC, MCC, threshold cost, latency, training/computing cost, model size, explanation capability and operating constraints.
- [ ] Interpretability comparison: coefficients/rules/SHAP for ML; attributions/attention/counterfactuals for DL; cited evidence/audit trace/human escalation for Agent AI.
- [ ] Production recommendation with guardrails: human decision authority, calibration and drift checks, fairness monitoring, privacy controls, uncertainty/manual-review routing and model/version rollback.
- [ ] Consolidated capstone presentation/report and final submission checklist.

---

## 8. Execution Order and Quality Gates

1. **Foundation and Approach 1:** build, run, inspect and verify the entire classical baseline first. No slide/PDF result is final until it is generated from verified output.
2. **Approach 2:** only after the frozen split and baseline artifacts exist, build/rerun the DL suite against the same test set and complete its verification.
3. **Approach 3:** complete the secure, offline-first evidence workflow, then test its scenarios and optional Gemini integration boundary.
4. **Capstone verification:** regenerate all written materials from current results, open generated PPT/PDF files, test a clean install/command path, validate links and filenames, run automated tests, and record known limitations.

### Completion guardrails

- Do not claim performance before an actual run writes the result.
- Do not label synthetic data as an observed Indian insurer dataset.
- Do not expose secrets or protected personal/health information.
- Do not use demographics as direct adverse-decision features in an operational scoring surface.
- Do not convert high-risk scores into automatic rejections; use them to prioritise review.
- Do not replace verifiable engineering with large, padded documentation. Documentation must be substantive, cited, internally consistent and traceable to code/artifacts.

This file is the project source of truth. A checklist item changes from unchecked only when its corresponding artifact exists and has passed the stated verification gate.
