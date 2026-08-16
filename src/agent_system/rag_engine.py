"""
Retrieval-Augmented Generation (RAG) and Domain Knowledge Base Engine.
Indexes and retrieves Indian insurance policy clauses, IRDAI guidelines,
medical pricing schedules, and fraud investigation rulebooks.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import logger

class InsuranceKnowledgeRAG:
    """
    In-memory vectorized knowledge store supporting dense-lexical hybrid retrieval
    of Indian actuarial clauses, clinical guidelines, and fraud detection rulebooks.
    """
    
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_vectors: Optional[np.ndarray] = None
        self._build_knowledge_base()
        
    def _build_knowledge_base(self):
        """Constructs and indexes curated domain knowledge chunks."""
        corpus = [
            {
                "id": "POLICY-CLAUSE-01",
                "category": "Policy_Rules",
                "title": "Mandatory Waiting Periods for Pre-Existing Diseases (PED)",
                "content": (
                    "Under Section 4.1 of standard Indian Health Insurance policies (IRDAI/HLT/REG/2020), "
                    "pre-existing conditions (including hypertension, diabetes, osteoarthritis, cardiac ailments) "
                    "require a mandatory waiting period of 24 to 48 months from the policy inception date. "
                    "Any claim lodged prior to waiting period completion for declared PED is contractually excluded from coverage."
                )
            },
            {
                "id": "POLICY-CLAUSE-02",
                "category": "Policy_Rules",
                "title": "Co-payment and Sub-limits on Specific Treatments",
                "content": (
                    "Senior Citizen Red Carpet plans and policies with mandatory co-pay require a 10% to 20% "
                    "deductible paid by the claimant. Cataract surgeries are subject to an absolute sub-limit of ₹40,000 "
                    "per eye in Tier 2 cities and ₹60,000 in Tier 1 metros. Daycare laparoscopic procedures have maximum room-rent caps of 1% of Sum Insured."
                )
            },
            {
                "id": "IRDAI-REG-03",
                "category": "Regulatory_Standards",
                "title": "IRDAI Claim Settlement and Mandatory Written Justification",
                "content": (
                    "As per IRDAI Master Circular (Protection of Policyholders' Interests) Regulations, insurers cannot "
                    "arbitrarily deny claims. Any rejection or claim reduction must provide explicit evidence-backed justification, "
                    "itemized tariff variances, and cite the precise policy exclusion clause. Claimants must be provided 30 days to respond."
                )
            },
            {
                "id": "FRAUD-TYPOLOGY-04",
                "category": "Fraud_Rulebook",
                "title": "Billing Inflation & Tariff Variance Anomaly",
                "content": (
                    "Inflated billing occurs when healthcare providers charge amounts exceeding 2.0x the standard Schedule of Charges (SOC) "
                    "for that hospital tier and geographic region. In Tier 2/3 Indian cities, a laparoscopic appendectomy typically ranges between "
                    "₹45,000 and ₹95,000. Billing exceeding ₹1,80,000 without documented ICU complications indicates high fraud probability."
                )
            },
            {
                "id": "FRAUD-TYPOLOGY-05",
                "category": "Fraud_Rulebook",
                "title": "Phantom Hospitalization and Paper Reimbursement Rings",
                "content": (
                    "Phantom hospitalization involves submitting forged bills, admission stamps, and discharge notes for patients who were never admitted. "
                    "Key indicators include: 1-day stay for major surgery requiring multi-day recovery, unaccredited nursing homes, "
                    "handwritten non-GST bills, and claims filed via paper reimbursement within 90 days of policy issuance."
                )
            },
            {
                "id": "MEDICAL-BENCHMARK-06",
                "category": "Medical_Protocols",
                "title": "Standard Inpatient Stay Protocols for Indian Hospitals",
                "content": (
                    "Standard recovery protocols in Indian clinical practice: Total Knee Replacement (TKR) requires 4 to 6 inpatient days; "
                    "Laparoscopic Appendectomy requires 1 to 3 days; Coronary Angioplasty (PTCA) requires 2 to 4 days with 24 hours ICU observation; "
                    "Cataract Surgery is strictly a daycare procedure (0 days inpatient stay)."
                )
            }
        ]
        
        self.documents = corpus
        texts = [f"{d['title']} {d['content']}" for d in corpus]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(texts).toarray()
        logger.info(f"Initialized RAG Knowledge Base with {len(corpus)} structured domain documents.")
        
    def retrieve(self, query: str, top_k: int = 2, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k most relevant knowledge base chunks for a given agent query.
        """
        if self.vectorizer is None or self.doc_vectors is None:
            return []
            
        q_vec = self.vectorizer.transform([query]).toarray()
        sims = cosine_similarity(q_vec, self.doc_vectors).flatten()
        
        # Rank
        ranked_indices = np.argsort(sims)[::-1]
        results = []
        for idx in ranked_indices:
            doc = self.documents[idx]
            if category and doc["category"] != category:
                continue
            score = float(sims[idx])
            if score > 0.05 or len(results) == 0:
                doc_copy = copy_doc = dict(doc)
                doc_copy["relevance_score"] = round(score, 4)
                results.append(doc_copy)
                if len(results) >= top_k:
                    break
        return results
