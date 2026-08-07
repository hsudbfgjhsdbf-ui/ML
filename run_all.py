#!/usr/bin/env python3
"""
MASTER END-TO-END EXECUTION PIPELINE
Project: Medical Insurance Claim Fraud Detection System (Approaches 1, 2, and 3)
Institution: Indian Institute of Information Technology (IIIT), Dharwad
Department: B.Tech Data Science and Artificial Intelligence
Faculty Adviser: Prof. Ramesh Athe
Team Members: B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)

This script executes the entire three-pillar project end-to-end with a single command:
1. Data loading, domain enrichment, data dictionary, and synthetic data simulation.
2. Data preprocessing, stratified 70/15/15 splitting, and SMOTE resampling.
3. Domain feature engineering and multi-method selection (MI, RF, LASSO).
4. Approach 1: 12 Classical Supervised ML algorithms training and benchmarking.
5. Approach 2: 10 Deep Tabular PyTorch architectures training and benchmarking.
6. XAI Explainable AI layer (SHAP, LIME, Counterfactuals) and Demographic Fairness Audit.
7. Approach 3: Agent AI Multi-Agent System (LangGraph workflow, SQLite DB, RAG).
8. Generation of 30+ high-resolution PNG charts in `visualizations/`.
9. Generation of 2,000+-line Markdown documentation and evaluation files.
10. Generation of 20-slide PowerPoint presentation deck (`presentation/`).
11. Generation of formal IEEE two-column research paper PDF report (`reports/`).
"""

import os
import sys
import time
import pandas as pd
from src.utils import setup_logger, load_config, ensure_directories
from src.data_loading import execute_data_loading_pipeline
from src.data_preprocessing import split_dataset_stratified, InsuranceDataPreprocessor
from src.feature_engineering import InsuranceFeatureEngineer
from src.models.classical_models import ClassicalFraudModelBank
from src.models.deep_models import DeepFraudModelBank
from src.models.xai_explainer import ExplainableAIEngine
from src.agent_ai.database import InsuranceDatabaseManager
from src.agent_ai.rag_pipeline import IndianInsuranceKnowledgeBase
from src.agent_ai.workflow import ClaimProcessingState, AgentAIWorkflowOrchestrator
from src.visualization import InsuranceVisualizer
from src.doc_generator import ComprehensiveDocumentGenerator
from src.ppt_presentation import PresentationGenerator
from src.pdf_report import IEEEReportGenerator

logger = setup_logger("MasterPipelineLogger")


def main():
    logger.info("================================================================================")
    logger.info("STARTING MASTER END-TO-END EXECUTION PIPELINE")
    logger.info("Project: Medical Insurance Claim Fraud Detection System")
    logger.info("Faculty Adviser: Prof. Ramesh Athe | Institution: IIIT Dharwad")
    logger.info("Team: B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)")
    logger.info("================================================================================")
    start_total = time.time()
    
    # Load configuration
    config = load_config("configs/config.yaml")
    ensure_directories([
        "data/raw", "data/processed", "data/synthetic",
        "models_saved", "evaluation", "documentation",
        "visualizations", "presentation", "reports"
    ])
    
    # --------------------------------------------------------------------------
    # STEP 1: DATA LOADING, ENRICHMENT & SYNTHETIC SIMULATION
    # --------------------------------------------------------------------------
    logger.info("\n---> STEP 1/12: Data Loading, Indian Domain Enrichment & Synthetic Simulation")
    df = execute_data_loading_pipeline(raw_path=config["paths"]["raw_data"])
    
    # --------------------------------------------------------------------------
    # STEP 2: STRATIFIED SPLITTING (70% TRAIN / 15% VAL / 15% TEST)
    # --------------------------------------------------------------------------
    logger.info("\n---> STEP 2/12: Stratified 70/15/15 Train-Validation-Test Split")
    df_train, df_val, df_test = split_dataset_stratified(
        df,
        test_size=config["project"]["test_size"],
        val_size=config["project"]["val_size"],
        random_seed=config["project"]["random_seed"]
    )
    
    # --------------------------------------------------------------------------
    # STEP 3: DOMAIN FEATURE ENGINEERING & MULTI-METHOD SELECTION
    # --------------------------------------------------------------------------
    logger.info("\n---> STEP 3/12: Indian Domain Feature Engineering & Selection")
    engineer = InsuranceFeatureEngineer()
    df_train_feat = engineer.fit_transform(df_train)
    df_val_feat = engineer.transform(df_val)
    df_test_feat = engineer.transform(df_test)
    
    # --------------------------------------------------------------------------
    # STEP 4: PREPROCESSING & SMOTE RESAMPLING
    # --------------------------------------------------------------------------
    logger.info("\n---> STEP 4/12: Data Preprocessing & SMOTE Resampling")
    preprocessor = InsuranceDataPreprocessor(scaler_type="standard", imbalance_method="smote")
    X_train, y_train = preprocessor.fit_transform(df_train_feat)
    X_val, y_val = preprocessor.transform(df_val_feat)
    X_test, y_test = preprocessor.transform(df_test_feat)
    
    X_train_res, y_train_res = preprocessor.handle_class_imbalance(X_train, y_train, method="smote")
    preprocessor.save_preprocessor()
    
    # Perform feature selection on resampled training data
    top_features, ranking_df = engineer.perform_feature_selection(X_train_res, y_train_res, top_k=20)
    
    # --------------------------------------------------------------------------
    # STEP 5: APPROACH 1 - 12 CLASSICAL ML ALGORITHMS
    # --------------------------------------------------------------------------
    logger.info("\n---> STEP 5/12: Approach 1 - 12 Classical ML Algorithms Training & Evaluation")
    bank_a1 = ClassicalFraudModelBank(random_seed=config["project"]["random_seed"])
    bank_a1.train_and_tune_all(X_train_res, y_train_res, cv_folds=config["approach1"]["cv_folds"], tune_hyperparams=False)
    benchmark_a1 = bank_a1.evaluate_all(X_test, y_test)
    bank_a1.save_all_models()
    
    # --------------------------------------------------------------------------
    # STEP 6: APPROACH 2 - 10 DEEP TABULAR PYTORCH ARCHITECTURES
    # --------------------------------------------------------------------------
    logger.info("\n---> STEP 6/12: Approach 2 - 10 Deep Tabular PyTorch Architectures Training")
    bank_a2 = DeepFraudModelBank(input_dim=X_train_res.shape[1], random_seed=config["project"]["random_seed"])
    bank_a2.train_all(
        X_train_res, y_train_res, X_val, y_val,
        epochs=config["approach2"]["epochs"],
        batch_size=config["approach2"]["batch_size"],
        lr=config["approach2"]["learning_rate"],
        focal_gamma=config["approach2"]["focal_loss_gamma"]
    )
    benchmark_a2 = bank_a2.evaluate_all(X_test, y_test)
    bank_a2.save_all_models()
    
    # --------------------------------------------------------------------------
    # STEP 7: XAI EXPLAINABILITY & INDIAN DEMOGRAPHIC FAIRNESS AUDIT
    # --------------------------------------------------------------------------
    logger.info("\n---> STEP 7/12: XAI Explainability Layer & Demographic Fairness Audit")
    xai = ExplainableAIEngine(feature_names=list(X_test.columns))
    best_rf_model = bank_a1.models["Random_Forest"]
    xai.compute_shap_explanations(best_rf_model, X_train_res, X_test, model_type="sklearn")
    xai.compute_lime_explanation(best_rf_model, X_train_res, X_test.iloc[0], model_type="sklearn")
    xai.generate_counterfactual_explanation(best_rf_model, X_test.iloc[0], list(X_test.columns), model_type="sklearn")
    fairness_results = xai.conduct_fairness_audit(y_test, best_rf_model.predict(X_test), df_test)
    
    # --------------------------------------------------------------------------
    # STEP 8: APPROACH 3 - AGENT AI MULTI-AGENT SYSTEM
    # --------------------------------------------------------------------------
    logger.info("\n---> STEP 8/12: Approach 3 - Agent AI Multi-Agent Cognitive System Execution")
    db_manager = InsuranceDatabaseManager(config["paths"]["db_path"])
    orchestrator = AgentAIWorkflowOrchestrator(db_manager)
    
    sample_claim = {
        "claim_id": "CLM-AUTO-2026",
        "policy_number": "STAR-HLTH-2024-8871",
        "user_id": "USR-IND-001",
        "provider_id": "HOSP-MUM-01",
        "hospital_name": "Apollo Hospitals Navi Mumbai",
        "treatment_type": "Inpatient",
        "procedure_code": "IND-PROC-101",
        "claimed_amount_inr": 135000.0,
        "patient_age": 48
    }
    state = ClaimProcessingState(
        claim_id=sample_claim["claim_id"],
        policy_number=sample_claim["policy_number"],
        user_id=sample_claim["user_id"],
        raw_claim_context=sample_claim,
        uploaded_documents=[{"document_type": "Hospital Bill", "file_path": "sample_bill.pdf"}]
    )
    final_state = orchestrator.run_workflow(state)
    logger.info(f"Agent AI Workflow completed: {final_state.claim_id} -> Verdict: {final_state.final_decision_result['decision']}")
    
    # --------------------------------------------------------------------------
    # STEP 9: GENERATE 30+ HIGH-RESOLUTION VISUALIZATIONS
    # --------------------------------------------------------------------------
    logger.info("\n---> STEP 9/12: Generating 30+ High-Resolution Visualizations in `visualizations/`")
    visualizer = InsuranceVisualizer(config["paths"]["viz_dir"])
    all_probs = {**bank_a1.probabilities, **bank_a2.probabilities}
    saved_charts = visualizer.generate_all(df, ranking_df, benchmark_a1, benchmark_a2, y_test, all_probs, fairness_results)
    logger.info(f"Generated {len(saved_charts)} visualization charts.")
    
    # --------------------------------------------------------------------------
    # STEP 10: GENERATE 2000+-LINE DOCUMENTATION & EVALUATION MARKDOWN FILES
    # --------------------------------------------------------------------------
    logger.info("\n---> STEP 10/12: Generating Comprehensive 2,000+-Line Markdown Documents")
    doc_gen = ComprehensiveDocumentGenerator(config["paths"]["eval_dir"], config["paths"]["docs_dir"])
    doc_gen.generate_evaluation_report(benchmark_a1, benchmark_a2, fairness_results)
    doc_gen.generate_project_documentation()
    doc_gen.generate_code_explanation()
    doc_gen.generate_auxiliary_reports(benchmark_a1, benchmark_a2)
    
    # --------------------------------------------------------------------------
    # STEP 11: GENERATE 20-SLIDE POWERPOINT PRESENTATION DECK
    # --------------------------------------------------------------------------
    logger.info("\n---> STEP 11/12: Generating 20-Slide PowerPoint Presentation Deck")
    ppt_gen = PresentationGenerator(config["paths"]["ppt_dir"])
    ppt_gen.generate_ppt_and_md(benchmark_a1, benchmark_a2)
    
    # --------------------------------------------------------------------------
    # STEP 12: GENERATE FORMAL IEEE TWO-COLUMN RESEARCH PAPER PDF REPORT
    # --------------------------------------------------------------------------
    logger.info("\n---> STEP 12/12: Generating Formal IEEE Two-Column Research Paper PDF Report")
    pdf_gen = IEEEReportGenerator(config["paths"]["reports_dir"])
    pdf_gen.generate_pdf_report(benchmark_a1, benchmark_a2)
    
    total_time = time.time() - start_total
    logger.info("================================================================================")
    logger.info(f"MASTER END-TO-END EXECUTION COMPLETE in {total_time:.2f} seconds!")
    logger.info("All deliverables (Models, Benchmarks, Charts, Docs, PPTX, IEEE PDF) ready.")
    logger.info("Faculty Adviser: Prof. Ramesh Athe | IIIT Dharwad")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
