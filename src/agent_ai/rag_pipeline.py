"""
Retrieval-Augmented Generation (RAG) pipeline for Medical Insurance Claim Fraud Detection (Approach 3).
Institution: IIIT Dharwad, Department of Data Science and AI
Faculty Adviser: Prof. Ramesh Athe
Team Members: B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)

This module implements:
1. Curated Indian Insurance Knowledge Base (Policy Clauses, IRDAI Rules, Fraud Rulebooks, Medical Costs).
2. Document chunking by semantic clause / rule.
3. TF-IDF / Vector similarity retrieval engine.
4. Top-K document chunk retrieval with citation metadata.
"""

import os
import re
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import setup_logger

logger = setup_logger("RAGPipelineLogger")


class IndianInsuranceKnowledgeBase:
    """
    RAG Vector Store and Retrieval Engine for Indian Insurance Policy Clauses,
    IRDAI Regulations, Fraud Rulebooks, and Medical Cost Benchmarks.
    """
    def __init__(self):
        self.documents: List[Dict[str, str]] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.doc_matrix = None
        self._seed_knowledge_base()
        self._fit_index()

    def _seed_knowledge_base(self) -> None:
        """
        Seeds knowledge base with Indian insurance domain documents.
        """
        logger.info("Seeding Indian Insurance RAG Knowledge Base with policy clauses and IRDAI rules...")
        raw_docs = [
            {
                "id": "CLAUSE-ROOM-001",
                "category": "Policy_Clause",
                "title": "Room Rent Capping and Proportionate Deduction",
                "text": "Under Star Health and ICICI Lombard Family Floater policies, room rent is capped at 1% of Sum Insured per day for normal rooms and 2% of Sum Insured for ICU. If the insured chooses a higher room category, hospital nursing, doctor visit charges, and surgical fees will be proportionately deducted."
            },
            {
                "id": "CLAUSE-WAIT-002",
                "category": "Policy_Clause",
                "title": "Waiting Period for Pre-Existing Diseases (PED)",
                "text": "All pre-existing medical conditions disclosed or undisclosed are subject to a mandatory 24-month to 36-month waiting period from policy inception date. Any claim filed for chronic cardiovascular, renal, or orthopedic conditions within the first 24 months shall be rejected."
            },
            {
                "id": "CLAUSE-COPAY-003",
                "category": "Policy_Clause",
                "title": "Senior Citizen Co-Payment Clause",
                "text": "Policyholders aged 60 years and above at policy inception are subject to a mandatory 10% to 20% co-payment on the total allowable claim settlement amount across inpatient hospitalizations."
            },
            {
                "id": "IRDAI-REG-101",
                "category": "IRDAI_Regulation",
                "title": "IRDAI Turnaround Time for Health Claim Settlement",
                "text": "As per IRDAI Guidelines, insurance companies must process and settle or formally reject a health insurance claim within 30 days of receipt of all required medical documents. Any rejection letter must explicitly state the policy clause and grievance redressal procedure."
            },
            {
                "id": "FRAUD-RULE-201",
                "category": "Fraud_Rulebook",
                "title": "Billing Inflation and Unbundled Surgical Consumables",
                "text": "A common Indian hospital fraud pattern is billing inflation where surgical consumables, PPE kits, and routine monitoring are billed separately at 300% markup above MRP. If consumables exceed 15% of the total claim amount, the claim must be flagged for billing audit."
            },
            {
                "id": "FRAUD-RULE-202",
                "category": "Fraud_Rulebook",
                "title": "Tier-3 Hospital Charging Metro Corporate Rates",
                "text": "Non-accredited nursing homes in smaller towns billing at Tier-1 Metro corporate rates (e.g. charging over Rs. 1,50,000 for Appendectomy) without advanced ICU or full-time specialists is a critical fraud indicator."
            },
            {
                "id": "FRAUD-RULE-203",
                "category": "Fraud_Rulebook",
                "title": "Organized Fraud Ring Collusion Indicator",
                "text": "When multiple claimants from the same employer group or locality file claims for identical diagnoses (e.g. Dengue or Acute Gastroenteritis) at the same unlisted nursing home within a 30-day window, organized fraud collusion is suspected."
            },
            {
                "id": "MED-REF-301",
                "category": "Medical_Reference",
                "title": "Indian Standard Cost Structure for Appendectomy",
                "text": "In India, Laparoscopic Appendectomy typical billing amounts are: Tier-1 Metro Corporate Hospital Rs. 1,20,000 to Rs. 1,60,000; Tier-2 City Hospital Rs. 70,000 to Rs. 1,00,000; Tier-3 Town Nursing Home Rs. 35,000 to Rs. 55,000. Typical length of stay is 2 days."
            },
            {
                "id": "MED-REF-302",
                "category": "Medical_Reference",
                "title": "Indian Standard Cost Structure for Knee Replacement",
                "text": "Total Knee Replacement (Unilateral) in India: Tier-1 Metro Hospital Rs. 2,80,000 to Rs. 3,50,000; Tier-2 City Hospital Rs. 1,80,000 to Rs. 2,40,000; Tier-3 Town Hospital Rs. 1,10,000 to Rs. 1,50,000. Standard stay is 4 to 6 days."
            }
        ]
        self.documents = raw_docs

    def _fit_index(self) -> None:
        """
        Builds TF-IDF vector index for fast semantic similarity search.
        """
        texts = [f"{d['title']} {d['text']}" for d in self.documents]
        self.doc_matrix = self.vectorizer.fit_transform(texts)
        logger.debug("RAG vector index built successfully.")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the knowledge base for the most relevant document chunks matching the query.
        Returns chunk content along with citation metadata.
        """
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                "id": doc["id"],
                "category": doc["category"],
                "title": doc["title"],
                "text": doc["text"],
                "similarity": float(sims[idx]),
                "citation": f"[{doc['id']}] {doc['title']} ({doc['category']})"
            })
        logger.debug(f"RAG search for '{query[:30]}...' returned {len(results)} chunks.")
        return results
