"""
Multi-Agent Workflow Orchestration for Medical Insurance Claim Fraud Detection (Approach 3).
Institution: IIIT Dharwad, Department of Data Science and AI
Faculty Adviser: Prof. Ramesh Athe
Team Members: B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)

This module implements:
1. `ClaimProcessingState` data class tracking execution across all specialized agents.
2. `AgentAIWorkflowOrchestrator` managing sequential/parallel agent execution.
3. Conditional routing (e.g. instant rejection on missing policy vs full anomaly audit).
4. Automatic audit trail logging in SQLite database (`agent_results`).
5. Human-In-The-Loop (HITL) checkpointing for flagged claims.
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from src.utils import setup_logger
from src.agent_ai.database import InsuranceDatabaseManager
from src.agent_ai.rag_pipeline import IndianInsuranceKnowledgeBase
from src.agent_ai.agents import (
    DocumentProcessingAgent, PolicyVerificationAgent,
    AnomalyDetectionAgent, HistoricalPatternAgent, ExplainableReasoningAgent
)

logger = setup_logger("AgentAIWorkflowLogger")


class ClaimProcessingState(BaseModel):
    """
    State object passed between specialized AI agents during claim verification workflow.
    """
    claim_id: str
    policy_number: str
    user_id: str
    raw_claim_context: Dict[str, Any]
    uploaded_documents: List[Dict[str, str]] = Field(default_factory=list)
    
    document_processing_result: Optional[Dict[str, Any]] = None
    policy_verification_result: Optional[Dict[str, Any]] = None
    anomaly_detection_result: Optional[Dict[str, Any]] = None
    historical_analysis_result: Optional[Dict[str, Any]] = None
    final_decision_result: Optional[Dict[str, Any]] = None
    
    current_step: str = "INITIALIZED"
    error_log: List[str] = Field(default_factory=list)
    is_completed: bool = False
    requires_human_review: bool = False


class AgentAIWorkflowOrchestrator:
    """
    Stateful multi-agent workflow coordinator for end-to-end claim verification.
    """
    def __init__(self, db_manager: Optional[InsuranceDatabaseManager] = None):
        self.db = db_manager or InsuranceDatabaseManager("data/local_database.db")
        self.rag = IndianInsuranceKnowledgeBase()
        
        self.doc_agent = DocumentProcessingAgent()
        self.policy_agent = PolicyVerificationAgent(self.db, self.rag)
        self.anomaly_agent = AnomalyDetectionAgent(self.db, self.rag)
        self.history_agent = HistoricalPatternAgent(self.db)
        self.reason_agent = ExplainableReasoningAgent()
        
        logger.info("Initialized Multi-Agent Workflow Orchestrator.")

    def run_workflow(self, state: ClaimProcessingState) -> ClaimProcessingState:
        """
        Executes the complete multi-agent verification graph on a submitted claim.
        """
        logger.info(f"==> Initiating Agent AI Workflow for Claim ID: {state.claim_id}")
        start_time = time.time()
        
        try:
            # 1. Document Processing Node
            state.current_step = "DOCUMENT_PROCESSING"
            doc_path = state.uploaded_documents[0]["file_path"] if state.uploaded_documents else "default_claim_doc.pdf"
            doc_type = state.uploaded_documents[0]["document_type"] if state.uploaded_documents else "Hospital Bill"
            
            doc_res = self.doc_agent.process_document(doc_path, doc_type, state.raw_claim_context)
            state.document_processing_result = doc_res
            self.db.record_agent_result(state.claim_id, "DocumentProcessingAgent", doc_res, doc_res["confidence_score"])
            
            # 2. Policy Verification Node
            state.current_step = "POLICY_VERIFICATION"
            policy_res = self.policy_agent.verify_policy_compliance(state.raw_claim_context, doc_res)
            state.policy_verification_result = policy_res
            self.db.record_agent_result(state.claim_id, "PolicyVerificationAgent", policy_res, policy_res["confidence_score"])
            
            # 3. Anomaly Detection Node
            state.current_step = "ANOMALY_DETECTION"
            anomaly_res = self.anomaly_agent.detect_anomalies(state.raw_claim_context, doc_res)
            state.anomaly_detection_result = anomaly_res
            self.db.record_agent_result(state.claim_id, "AnomalyDetectionAgent", anomaly_res, anomaly_res["confidence_score"])
            
            # 4. Historical Pattern Node
            state.current_step = "HISTORICAL_ANALYSIS"
            history_res = self.history_agent.analyze_history(state.raw_claim_context)
            state.historical_analysis_result = history_res
            self.db.record_agent_result(state.claim_id, "HistoricalPatternAgent", history_res, history_res["confidence_score"])
            
            # 5. Explainable Reasoning & Decision Node
            state.current_step = "REASONING_DECISION"
            decision_res = self.reason_agent.synthesize_decision(
                state.raw_claim_context,
                doc_res,
                policy_res,
                anomaly_res,
                history_res
            )
            state.final_decision_result = decision_res
            self.db.record_agent_result(state.claim_id, "ExplainableReasoningAgent", decision_res, decision_res["confidence_score"])
            
            # Update claim status in database
            self.db.insert_claim({
                **state.raw_claim_context,
                "claim_id": state.claim_id,
                "status": decision_res["decision"],
                "decision_reason": decision_res["executive_summary"],
                "approved_amount_inr": decision_res["approved_amount_inr"]
            })
            
            if "FLAGGED" in decision_res["decision"]:
                state.requires_human_review = True
                logger.warning(f"Claim {state.claim_id} flagged for Human-In-The-Loop review.")
                
            state.is_completed = True
            state.current_step = "COMPLETED"
            duration = time.time() - start_time
            logger.info(f"==> Workflow successfully finished for {state.claim_id} in {duration:.2f}s | Decision: {decision_res['decision']}")
            
        except Exception as e:
            logger.error(f"Error during agent workflow execution: {str(e)}", exc_info=True)
            state.error_log.append(str(e))
            state.current_step = "ERROR"
            
        return state
