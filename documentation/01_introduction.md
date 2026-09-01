# Chapter 1: Introduction and Problem Statement

**Project Title:** Medical Insurance Claim Fraud Detection  
**Subtitle:** AI-Driven Claim Verification & Explainable Fraud Detection Platform  
**Institution:** Indian Institute of Information Technology (IIIT), Dharwad  
**Department:** B.Tech in Data Science and Artificial Intelligence  
**Faculty Adviser:** Prof. Ramesh Athe  
**Team Members:**  
- B Varshith — Roll Number 23BDS011  
- M Jagadeshwar — Roll Number 23BDS033  
- J Ganesh — Roll Number 23BDS024  
**Academic Year:** 2024–2025 / 2025–2026  

---

## 1.1 Project Background
Health insurance fraud is one of the most pervasive, costly, and destabilizing challenges facing modern healthcare systems worldwide. In India, the rapid growth of both government health coverage (such as the Ayushman Bharat Pradhan Mantri Jan Arogya Yojana - PM-JAY) and private health insurance (underwritten by Star Health, ICICI Lombard, HDFC ERGO, and New India Assurance) has led to an unprecedented surge in claim volumes.

However, fraudulent practices — including:
1. **Inflated Medical Billing:** Healthcare providers charging 2x to 5x standard rates for routine interventions.
2. **Phantom Hospitalizations:** Submitting fake admission records, forged discharge summaries, and fabricated nursing logs for patients who were never admitted.
3. **Upcoding of Procedures:** Billing for high-complexity surgical procedures when only minor conservative treatment was provided.
4. **Waiting Period Exploits:** Filing claims for pre-existing medical conditions shortly after policy inception by misrepresenting diagnosis timing.
5. **Organized Fraud Syndicates:** Collusion between corrupt clinics, rogue diagnostic laboratories, and claimant networks.

These fraudulent schemes impose staggering financial losses on insurance providers, directly leading to increased premiums for honest policyholders and draining public healthcare funds.

---

## 1.2 The Problem Statement
Traditional fraud detection in Indian health insurance relies on manual auditing by human claim adjusters and Third-Party Administrators (TPAs). This paradigm suffers from critical bottlenecks:
- **Scalability Failure:** Manual review cannot process hundreds of thousands of daily claims without severe settlement delays.
- **Inconsistency:** Human auditors exhibit subjective decision-making and cognitive fatigue.
- **The Black-Box Dilemma:** When modern automated algorithms or auditors flag/reject a claim, policyholders rarely receive a transparent, understandable explanation.
- **Regulatory Penalties:** The Insurance Regulatory and Development Authority of India (IRDAI) mandates strict settlement timelines and requires clear, evidence-backed justifications for claim rejections.

---

## 1.3 Project Objectives
The primary objectives of this project are:
1. **Tripartite Modeling Paradigm:** Implement, benchmark, and compare three distinct AI paradigms:
   - *Approach 1:* Traditional Machine Learning with 12+ classifiers and domain feature engineering.
   - *Approach 2:* Tabular Deep Learning with 10 neural architectures (FT-Transformer, TabNet, NODE, ResNet, VAE) trained with Focal Loss.
   - *Approach 3:* Agent AI Multi-Agent Cognitive System with LangGraph orchestration, multi-modal OCR/VLM extraction, and RAG retrieval over IRDAI guidelines.
2. **Indian Context Grounding:** Model Indian currency (INR ₹), tiered healthcare facilities (Tier 1 Metros to Tier 3 Nursing Homes), regional cost distributions across 16 Indian states, and government schemes like Ayushman Bharat.
3. **Transparent Explainability & Bilingual Reasoning:** Provide evidence-backed, layered explanations in both English and Hindi citing exact policy clauses and rupee tariff deviations.
4. **Rigorous Statistical Benchmarking:** Evaluate all models on a held-out test partition using F2-Score, AUC-ROC, MCC, McNemar's test, and Wilcoxon signed-rank tests.
5. **Interactive Full-Stack Platform:** Deliver an interactive web dashboard and FastAPI REST server bound to `0.0.0.0:8000` with live preview.
