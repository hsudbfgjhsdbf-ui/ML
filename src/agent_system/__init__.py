"""
Agent AI Multi-Agent System Package.
Cognitive Multi-Modal Claim Verification and Explainable Reasoning Layer.
"""

from src.agent_system.db import (
    initialize_local_database, get_db_connection,
    get_claim_with_details, insert_claim_record
)
from src.agent_system.rag_engine import InsuranceKnowledgeRAG
from src.agent_system.coordinator import MultiAgentCoordinator
