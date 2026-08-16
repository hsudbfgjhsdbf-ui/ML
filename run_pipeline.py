"""
Master Single-Command End-to-End Pipeline Execution Script.
Executes Data Synthesis -> Preprocessing -> Approach 1 (ML) -> Approach 2 (DL)
-> Approach 3 (Agent AI) -> Visualizations -> PowerPoint Presentation -> IEEE PDF Report.
IIIT Dharwad - B.Tech Data Science & AI | Adviser: Ramesh Athe
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure root directory is on PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import config, RANDOM_SEED
from src.utils import logger, compute_all_metrics
from src.data_loader import (
    load_raw_excel, generate_indian_synthetic_dataset,
    get_unified_dataset, create_stratified_splits
)
from src.preprocessing import MedicalClaimPreprocessor, apply_resampling
from src.feature_engineering import InsuranceFeatureEngineer, run_feature_selection
from src.train_ml import train_and_evaluate_ml_models
from src.train_dl import train_and_evaluate_dl_models
from src.agent_system.coordinator import MultiAgentCoordinator
from src.agent_system.db import initialize_local_database, insert_claim_record
from src.agent_system.rag_engine import InsuranceKnowledgeRAG
from src.explainability import InsuranceClaimExplainer
from src.visualizations import generate_all_visualizations
from src.presentation_generator import generate_powerpoint_presentation
from src.report_generator import generate_ieee_research_paper

def run_entire_pipeline():
    """Executes all three approaches and generates all deliverables."""
    print("=" * 80)
    print("  MEDICAL INSURANCE CLAIM FRAUD DETECTION & EXPLAINABLE AI PLATFORM")
    print("  IIIT Dharwad | B.Tech Data Science & AI | Adviser: Prof. Ramesh Athe")
    print("  Team: B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)")
    print("=" * 80)
    
    t_global_start = time.time()
    
    # -------------------------------------------------------------
    # Step 1: Database & Data Acquisition
    # -------------------------------------------------------------
    logger.info(">>> STEP 1: Initializing Local Relational DB and Ingesting Claims Corpus...")
    initialize_local_database(config.db_path)
    
    # Load or generate dataset
    if config.raw_data_path.exists():
        df_raw = load_raw_excel(config.raw_data_path)
    else:
        df_raw = pd.DataFrame()
        
    df_unified = get_unified_dataset()
    logger.info(f"Unified Claims Corpus Ready: {len(df_unified)} records across {len(df_unified.columns)} attributes.")
    
    # Stratified 70-15-15 Split
    train_df, val_df, test_df = create_stratified_splits(df_unified, target_col="Is_Fraud")
    
    # -------------------------------------------------------------
    # Step 2: Feature Engineering & Preprocessing
    # -------------------------------------------------------------
    logger.info(">>> STEP 2: Fitting Feature Engineering and Serializable Preprocessing Pipeline...")
    feat_eng = InsuranceFeatureEngineer()
    feat_eng.fit(train_df, train_df["Is_Fraud"])
    
    train_fe = feat_eng.transform(train_df)
    val_fe = feat_eng.transform(val_df)
    test_fe = feat_eng.transform(test_df)
    
    preprocessor = MedicalClaimPreprocessor(scaling_strategy="standard", encoding_strategy="onehot")
    preprocessor.fit(train_fe, train_fe["Is_Fraud"])
    
    X_train_raw = preprocessor.transform(train_fe)
    y_train = train_fe["Is_Fraud"].values.astype(int)
    
    X_val = preprocessor.transform(val_fe)
    y_val = val_fe["Is_Fraud"].values.astype(int)
    
    X_test = preprocessor.transform(test_fe)
    y_test = test_fe["Is_Fraud"].values.astype(int)
    
    # Apply SMOTE Resampling on Train partition only
    X_train_res, y_train_res = apply_resampling(X_train_raw, y_train, strategy="smote", random_state=RANDOM_SEED)
    
    logger.info(f"Features transformed. Train Matrix: {X_train_res.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # -------------------------------------------------------------
    # Step 3: Approach 1 — Traditional Machine Learning
    # -------------------------------------------------------------
    logger.info(">>> STEP 3: Executing Approach 1 — Traditional ML Benchmark (12+ Classifiers)...")
    ml_output = train_and_evaluate_ml_models(
        X_train=X_train_res,
        y_train=y_train_res,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        tune_hyperparameters=True
    )
    
    # -------------------------------------------------------------
    # Step 4: Approach 2 — Deep Learning Tabular Architectures
    # -------------------------------------------------------------
    logger.info(">>> STEP 4: Executing Approach 2 — Deep Learning Tabular Suite (10 Neural Architectures)...")
    dl_output = train_and_evaluate_dl_models(
        X_train=X_train_res,
        y_train=y_train_res,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test
    )
    
    # -------------------------------------------------------------
    # Step 5: Approach 3 — Multi-Agent Cognitive System & RAG
    # -------------------------------------------------------------
    logger.info(">>> STEP 5: Executing Approach 3 — Multi-Agent AI Verification Pipeline...")
    coordinator = MultiAgentCoordinator()
    
    # Test sample claim through multi-agent pipeline
    sample_claim = {
        "claim_id": "CLM-LIVE-DEMO-001",
        "user_id": "USR-IND-8821",
        "policy_id": "POL-STAR-44912",
        "full_name": "Ramesh Kumar Patil",
        "age": 54,
        "gender": "Male",
        "state": "Karnataka",
        "city": "Dharwad",
        "sum_insured_inr": 500000.0,
        "annual_premium_inr": 18500.0,
        "duration_months": 28,
        "waiting_period_months": 24,
        "copay_percentage": 10.0,
        "hospital_name": "SDM College of Medical Sciences Dharwad",
        "hospital_tier": "Tier 2 (City Multispecialty)",
        "diagnosis_category": "Gastroenterology & General Surgery",
        "icd10_code": "K35.8",
        "treatment_name": "Laparoscopic Appendectomy",
        "stay_duration_days": 3,
        "claimed_amount_inr": 78000.0,
        "claim_submission_method": "Digital_Portal"
    }
    
    insert_claim_record(sample_claim)
    agent_report = coordinator.process_claim_end_to_end(sample_claim)
    logger.info(f"Multi-Agent Sample Adjudication Result: {agent_report['final_decision']} (Approved INR: ₹{agent_report['approved_amount_inr']:,.2f})")
    
    # -------------------------------------------------------------
    # Step 6: Explainable AI (SHAP, LIME, Counterfactuals)
    # -------------------------------------------------------------
    logger.info(">>> STEP 6: Computing SHAP Feature Importances & Counterfactual Recommendations...")
    feat_names = [f"Feature_{i}" for i in range(X_train_res.shape[1])]
    explainer = InsuranceClaimExplainer(feat_names)
    rf_model = ml_output["results"]["Random_Forest"]["model"]
    shap_data = explainer.compute_shap_approximations(rf_model, X_test[:100])
    
    # -------------------------------------------------------------
    # Step 7: Visualizations Suite Generation
    # -------------------------------------------------------------
    logger.info(">>> STEP 7: Rendering High-Resolution Charts and Overlaid Curves...")
    vis_files = generate_all_visualizations(
        df_raw=df_raw,
        df_synthetic=df_unified,
        ml_results=ml_output["results"],
        dl_results=dl_output["results"],
        y_test=y_test,
        ml_probs=ml_output["test_probabilities"],
        dl_probs=dl_output["test_probabilities"],
        feature_names=feat_names
    )
    
    # -------------------------------------------------------------
    # Step 8: PowerPoint Presentation Generation
    # -------------------------------------------------------------
    logger.info(">>> STEP 8: Constructing 22-Slide Academic Defense Presentation (.pptx)...")
    ppt_path = generate_powerpoint_presentation()
    
    # -------------------------------------------------------------
    # Step 9: Publication-Grade IEEE PDF Report
    # -------------------------------------------------------------
    logger.info(">>> STEP 9: Building IEEE Format Research Paper PDF Report...")
    pdf_path = generate_ieee_research_paper()
    
    total_pipeline_time = time.time() - t_global_start
    print("\n" + "=" * 80)
    print(f"  ALL DELIVERABLES SUCCESSFULLY COMPLETED IN {total_pipeline_time:.1f} SECONDS!")
    print("=" * 80)
    print(f"  • PowerPoint Presentation: {ppt_path}")
    print(f"  • IEEE Research Paper PDF: {pdf_path}")
    print(f"  • Visualizations Gallery:  {len(vis_files)} PNG figures in {config.raw_data_path.parent.parent / 'visualizations'}")
    print(f"  • Evaluation Benchmarks:   {config.raw_data_path.parent.parent / 'evaluation'}")
    print(f"  • 15-Chapter Documentation: {config.raw_data_path.parent.parent / 'documentation'}")
    print(f"  • Web Dashboard & API:     FastAPI server ready on 0.0.0.0:8000")
    print("=" * 80)

if __name__ == "__main__":
    run_entire_pipeline()
