"""
05_agentic_rag_reasoning.py — Grounded agentic reasoning workflow using policy docs, fraud rules, claim history.

Agents / Modules:
- Document Verification Agent
- Policy Rule Matching Agent
- Claim Consistency Agent
- Historical Pattern / Anomaly Agent
- Evidence Retrieval Agent
- Decision Synthesis Agent
- Explanation Generation Agent

RAG pipeline retrieves from policy docs, coverage rules, exclusion clauses, fraud rules, guidelines, historical summaries.

Grounded: every explanation must be backed by retrieved evidence, extracted fields, model outputs, or explicit rules.
No hidden chain-of-thought; returns auditable reasoning summaries.

Optional LLM API with deterministic fallback.
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict
import math

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config
from common.logging_utils import get_logger
from common.artifacts import save_json
from common.result_formatting import risk_category_from_prob

logger = get_logger("05_agentic_rag")

# Simple TF-IDF fallback if sentence-transformers not available
class TFIDFVectorizerFallback:
    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.built = False

    def fit(self, docs: List[str]):
        # Build vocab and idf
        df = defaultdict(int)
        total = len(docs)
        tokenized = []
        for doc in docs:
            tokens = self.tokenize(doc)
            tokenized.append(tokens)
            for t in set(tokens):
                df[t]+=1
        # vocab
        self.vocab = {t:i for i,t in enumerate(sorted(df.keys()))}
        # idf
        for t, freq in df.items():
            self.idf[t] = math.log((total+1)/(freq+1))+1
        self.built=True
        self.tokenized_docs = tokenized
        # precompute doc vectors
        self.doc_vectors = [self.vectorize_tokens(toks) for toks in tokenized]

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def vectorize_tokens(self, tokens: List[str]) -> Dict[int,float]:
        tf = defaultdict(int)
        for t in tokens:
            tf[t]+=1
        vec = {}
        for t, cnt in tf.items():
            if t in self.vocab:
                idx = self.vocab[t]
                vec[idx] = (cnt/len(tokens))*self.idf.get(t,1.0)
        return vec

    def vectorize(self, text: str) -> Dict[int,float]:
        return self.vectorize_tokens(self.tokenize(text))

    @staticmethod
    def cosine_sim(vec1: Dict[int,float], vec2: Dict[int,float]) -> float:
        # sparse cosine
        if not vec1 or not vec2:
            return 0.0
        # dot
        dot = sum(vec1.get(k,0.0)*v for k,v in vec2.items())
        norm1 = math.sqrt(sum(v*v for v in vec1.values()))
        norm2 = math.sqrt(sum(v*v for v in vec2.values()))
        if norm1==0 or norm2==0:
            return 0.0
        return dot/(norm1*norm2)

    def search(self, query: str, top_k: int=5) -> List[Tuple[int,float]]:
        q_vec = self.vectorize(query)
        scores = [(i, self.cosine_sim(q_vec, doc_vec)) for i, doc_vec in enumerate(self.doc_vectors)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

class KnowledgeBase:
    def __init__(self, kb_dir: Path, config):
        self.kb_dir = Path(kb_dir)
        self.config = config
        self.chunks = []  # list of dict {id, text, source, metadata}
        self.vectorizer = TFIDFVectorizerFallback()
        self.use_sentence_transformers = False
        self.embedding_model = None

        # Try to use sentence-transformers if available and configured
        emb_provider = os.getenv("EMBEDDING_PROVIDER", config.get("rag",{}).get("embedding_model","tfidf_fallback"))
        if "sentence" in emb_provider.lower():
            try:
                from sentence_transformers import SentenceTransformer
                model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                self.embedding_model = SentenceTransformer(model_name)
                self.use_sentence_transformers = True
                logger.info(f"Using sentence-transformers {model_name}")
            except Exception as e:
                logger.info(f"Sentence-transformers not available {e}, using TFIDF fallback")
                self.use_sentence_transformers = False

        self.load_documents()

    def load_documents(self):
        """Ingest documents from kb_dir."""
        if not self.kb_dir.exists():
            logger.warning(f"KB dir {self.kb_dir} not found, creating sample")
            self.kb_dir.mkdir(parents=True, exist_ok=True)
        # Read all txt files
        docs = list(self.kb_dir.glob("*.txt")) + list(self.kb_dir.glob("*.md"))
        all_texts = []
        for doc in docs:
            try:
                text = doc.read_text(encoding="utf-8", errors="ignore")
                # Chunking
                chunk_size = self.config.get("rag",{}).get("chunk_size",500)
                overlap = self.config.get("rag",{}).get("chunk_overlap",50)
                chunks = self.chunk_text(text, chunk_size, overlap)
                for i, chunk in enumerate(chunks):
                    self.chunks.append({
                        "id": f"{doc.name}_chunk_{i}",
                        "text": chunk,
                        "source": str(doc),
                        "source_name": doc.name,
                        "chunk_index": i
                    })
                    all_texts.append(chunk)
            except Exception as e:
                logger.warning(f"Failed to read {doc}: {e}")

        if not self.chunks:
            # Create minimal fallback knowledge
            fallback = [
                {"id":"fallback_policy","text":"Policy must be active at claim date. Pre-authorization required for >$10000","source":"fallback"},
                {"id":"fallback_fraud","text":"High claim amount vs peers, duplicate billing, upcoding are fraud indicators","source":"fallback"}
            ]
            self.chunks = fallback
            all_texts = [c["text"] for c in fallback]

        # Build vector index
        if self.use_sentence_transformers:
            try:
                corpus = [c["text"] for c in self.chunks]
                self.embeddings = self.embedding_model.encode(corpus, convert_to_numpy=True)
            except Exception as e:
                logger.warning(f"Embedding failed {e}, fallback TFIDF")
                self.use_sentence_transformers=False
                self.vectorizer.fit(all_texts)
        else:
            self.vectorizer.fit(all_texts)

        logger.info(f"Knowledge base loaded {len(self.chunks)} chunks")

    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        words = text.split()
        chunks = []
        start=0
        while start < len(words):
            end = start+chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end-overlap
            if start<0:
                start=0
            if start>=len(words):
                break
        return chunks if chunks else [text]

    def similarity_search(self, query: str, top_k: int=5) -> List[Dict]:
        if self.use_sentence_transformers:
            try:
                q_emb = self.embedding_model.encode([query], convert_to_numpy=True)[0]
                # cosine sim
                import numpy as np
                scores = np.dot(self.embeddings, q_emb) / (np.linalg.norm(self.embeddings, axis=1)*np.linalg.norm(q_emb)+1e-9)
                top_indices = np.argsort(scores)[::-1][:top_k]
                results=[]
                for idx in top_indices:
                    c = self.chunks[idx]
                    results.append({**c, "score": float(scores[idx])})
                return results
            except Exception as e:
                logger.warning(f"ST search failed {e}, fallback")
        # TFIDF fallback
        top = self.vectorizer.search(query, top_k=top_k)
        results=[]
        for idx, score in top:
            c = self.chunks[idx]
            results.append({**c, "score": float(score)})
        return results

# Agents

class EvidenceRetrievalAgent:
    def __init__(self, kb: KnowledgeBase, config):
        self.kb = kb
        self.config = config

    def retrieve(self, query: str, top_k: int=None) -> List[Dict]:
        k = top_k or self.config.get("rag",{}).get("top_k",5)
        return self.kb.similarity_search(query, top_k=k)

class DocumentVerificationAgent:
    def check(self, claim: Dict, doc_validation_result: Dict) -> Dict:
        """Verify documents."""
        missing = doc_validation_result.get("missing_documents",[])
        duplicates = doc_validation_result.get("duplicates",[])
        overall_status = doc_validation_result.get("overall_validation_status","UNKNOWN")
        doc_results = doc_validation_result.get("document_results",[])

        evidence = []
        risk_signals = []
        if missing:
            risk_signals.append(f"Missing required documents: {', '.join(missing)}")
            evidence.append({"type":"missing_docs","details":missing, "source":"document_intelligence"})
        if duplicates:
            risk_signals.append(f"Duplicate documents detected: {len(duplicates)}")
            evidence.append({"type":"duplicate_docs","details":duplicates, "source":"document_intelligence"})
        if overall_status=="FAILED":
            risk_signals.append("Document validation FAILED")
            for dr in doc_results:
                if dr["validation"]["errors"]:
                    evidence.append({"type":"doc_error","doc":dr["document_type"],"errors":dr["validation"]["errors"],"source":dr["document_path"]})
        elif overall_status=="NEEDS_REVIEW":
            risk_signals.append("Document validation needs review")

        confidence = 0.9 if overall_status=="PASSED" else 0.4 if overall_status=="FAILED" else 0.6
        return {
            "agent": "DocumentVerificationAgent",
            "status": overall_status,
            "confidence": confidence,
            "risk_signals": risk_signals,
            "evidence": evidence,
            "positive_evidence": [] if missing else ["All required documents present"] if overall_status=="PASSED" else []
        }

class PolicyRuleMatchingAgent:
    def __init__(self, retriever: EvidenceRetrievalAgent):
        self.retriever = retriever

    def check(self, claim: Dict) -> Dict:
        # Build query from claim
        query = f"policy rules coverage for claim type {claim.get('ClaimType')} amount {claim.get('ClaimAmount')} procedure {claim.get('ProcedureCode')} diagnosis {claim.get('DiagnosisCode')}"
        retrieved = self.retriever.retrieve(query)

        violations = []
        matches = []
        # Simple rule checks deterministic
        try:
            amount = float(claim.get("ClaimAmount",0))
            if amount > 10000:
                # Check if pre-auth required per retrieved rule
                # Look for $10000 mention in retrieved
                text_combined = " ".join([r["text"] for r in retrieved]).lower()
                if "pre-authorization" in text_combined or "preauthorization" in text_combined or "10000" in text_combined:
                    # Assume need pre-auth - we don't have field, so flag for review if amount high
                    violations.append("High amount >$10000 requires pre-authorization check (policy rule)")
        except Exception:
            pass

        # Check ClaimType
        ctype = claim.get("ClaimType","")
        # If Emergency, need ER documentation - if we don't have, flag
        # This is heuristic

        evidence = [{"text": r["text"][:500], "source": r["source"], "score": r["score"]} for r in retrieved]

        confidence = 0.7
        status = "FAILED" if violations else "PASSED"
        return {
            "agent": "PolicyRuleMatchingAgent",
            "status": status,
            "confidence": confidence,
            "risk_signals": violations,
            "evidence": evidence,
            "positive_evidence": matches,
            "retrieved": retrieved
        }

class ClaimConsistencyAgent:
    def check(self, claim: Dict, doc_fields: Dict=None) -> Dict:
        risk_signals=[]
        evidence=[]
        # Check age vs diagnosis? Simplified
        try:
            age = int(claim.get("PatientAge",0))
            diag = str(claim.get("DiagnosisCode",""))
            # Example: if pediatric code but age > 20 flag? We don't have mapping, use heuristic if Cluster maybe
            if age < 0 or age > 120:
                risk_signals.append(f"Invalid patient age {age}")
        except:
            pass

        # Check amount vs income? High amount vs low income might be okay but flag
        try:
            amt = float(claim.get("ClaimAmount",0))
            inc = float(claim.get("PatientIncome",0))
            if inc>0 and amt>inc*0.5:
                risk_signals.append(f"Claim amount {amt} high relative to income {inc}")
                evidence.append({"type":"amount_vs_income","amount":amt,"income":inc})
        except:
            pass

        # Check provider specialty vs procedure? Simplified
        specialty = claim.get("ProviderSpecialty","")
        procedure = claim.get("ProcedureCode","")
        # No real mapping, just log

        # Date consistency: ClaimDate vs today? Should not be future
        try:
            import pandas as pd
            cd = pd.to_datetime(claim.get("ClaimDate"))
            if pd.notna(cd) and cd > pd.Timestamp.now():
                risk_signals.append("Claim date in future")
        except:
            pass

        # Compare doc fields if available
        if doc_fields:
            # bill_total vs claim amount
            bill_total = doc_fields.get("bill_total")
            if bill_total and claim.get("ClaimAmount"):
                try:
                    bt = float(bill_total)
                    ca = float(claim.get("ClaimAmount"))
                    if abs(bt-ca)>5:
                        risk_signals.append(f"Bill total {bt} != claimed {ca}")
                        evidence.append({"type":"bill_mismatch","bill_total":bt,"claimed":ca})
                except:
                    pass
            # diagnosis consistency
            doc_diag = doc_fields.get("diagnosis_code")
            if doc_diag and claim.get("DiagnosisCode") and doc_diag!=claim.get("DiagnosisCode"):
                risk_signals.append(f"Diagnosis mismatch doc {doc_diag} vs claim {claim.get('DiagnosisCode')}")

        status = "FAILED" if any("mismatch" in r.lower() or "invalid" in r.lower() for r in risk_signals) else "PASSED" if not risk_signals else "NEEDS_REVIEW"
        return {
            "agent": "ClaimConsistencyAgent",
            "status": status,
            "confidence": 0.75,
            "risk_signals": risk_signals,
            "evidence": evidence,
            "positive_evidence": [] if risk_signals else ["Claim fields internally consistent"]
        }

class HistoricalPatternAgent:
    def __init__(self, config, df_history: 'pd.DataFrame'=None):
        self.config = config
        self.df = df_history

    def check(self, claim: Dict, anomaly_score: float=None, model_prob: float=None) -> Dict:
        risk_signals=[]
        evidence=[]
        # If we have history dataframe, compare claim amount vs peer group
        if self.df is not None:
            try:
                # Group by ProviderSpecialty or DiagnosisCode
                specialty = claim.get("ProviderSpecialty")
                if specialty:
                    peer = self.df[self.df["ProviderSpecialty"]==specialty]
                    if len(peer)>10:
                        mean_amt = peer["ClaimAmount"].mean()
                        std_amt = peer["ClaimAmount"].std()
                        amt = float(claim.get("ClaimAmount",0))
                        if std_amt>0 and amt > mean_amt + 3*std_amt:
                            risk_signals.append(f"Claim amount {amt:.2f} > 3 std above peer mean {mean_amt:.2f} for specialty {specialty}")
                            evidence.append({"type":"peer_comparison","peer_mean":mean_amt,"peer_std":std_amt,"amount":amt})
            except Exception as e:
                pass

        if anomaly_score is not None:
            if anomaly_score > 0.8:  # assuming normalized 0-1, high = anomalous
                risk_signals.append(f"High anomaly score {anomaly_score:.3f} indicates deviation from normal pattern")
                evidence.append({"type":"anomaly_score","score":anomaly_score})
            elif anomaly_score > 0.5:
                risk_signals.append(f"Moderate anomaly score {anomaly_score:.3f}")

        if model_prob is not None:
            if model_prob > 0.7:
                risk_signals.append(f"Supervised model fraud probability high {model_prob:.3f}")
                evidence.append({"type":"model_prob","prob":model_prob})
            elif model_prob>0.3:
                risk_signals.append(f"Model fraud probability moderate {model_prob:.3f}")

        status = "FAILED" if any("high" in r.lower() for r in risk_signals) and len(risk_signals)>=2 else "NEEDS_REVIEW" if risk_signals else "PASSED"
        return {
            "agent": "HistoricalPatternAnomalyAgent",
            "status": status,
            "confidence": 0.7,
            "risk_signals": risk_signals,
            "evidence": evidence,
            "positive_evidence": [] if risk_signals else ["No strong historical anomaly"]
        }

class DecisionSynthesisAgent:
    def synthesize(self, results: List[Dict], model_prob: float, anomaly_score: float, config) -> Dict:
        # Aggregate risk
        all_risks = []
        all_evidence = []
        positive = []
        for r in results:
            all_risks.extend(r.get("risk_signals",[]))
            all_evidence.extend(r.get("evidence",[]))
            positive.extend(r.get("positive_evidence",[]))

        # Weights from config
        thr = config.get("hybrid",{}).get("decision_thresholds",{})
        approve_max = thr.get("approve_max_prob",0.3)
        reject_min = thr.get("reject_min_prob",0.7)

        # Determine risk category
        prob = model_prob if model_prob is not None else 0.5
        risk_cat = risk_category_from_prob(prob)

        # Adjust risk category based on number of risk signals
        if len(all_risks)>=3:
            risk_cat = "HIGH"
        elif len(all_risks)>=1 and risk_cat=="LOW":
            risk_cat="MEDIUM"

        # Decision
        # Conservative: if doc FAILED or many signals, escalate
        doc_failed = any(r["agent"]=="DocumentVerificationAgent" and r["status"]=="FAILED" for r in results)
        if doc_failed:
            if prob>=reject_min:
                decision="REJECT_OR_ESCALATE"
            else:
                decision="FLAG_FOR_MANUAL_REVIEW"
        elif prob < approve_max and not all_risks:
            decision="APPROVE"
        elif prob < reject_min:
            decision="FLAG_FOR_MANUAL_REVIEW"
        else:
            decision="REJECT_OR_ESCALATE"

        # Human review recommendation
        human_review = decision!="APPROVE" or len(all_risks)>0

        return {
            "risk_category": risk_cat,
            "recommended_decision": decision,
            "human_review_required": human_review,
            "aggregated_risk_signals": all_risks,
            "aggregated_evidence": all_evidence,
            "positive_evidence": positive,
            "fraud_probability": prob,
            "anomaly_score": anomaly_score
        }

class ExplanationGenerationAgent:
    def generate(self, synthesis: Dict, results: List[Dict]) -> str:
        prob = synthesis.get("fraud_probability",0.0)
        risk_cat = synthesis.get("risk_category","UNKNOWN")
        decision = synthesis.get("recommended_decision","FLAG_FOR_MANUAL_REVIEW")
        risks = synthesis.get("aggregated_risk_signals",[])
        positive = synthesis.get("positive_evidence",[])
        evidence_refs = synthesis.get("aggregated_evidence",[])

        parts=[]
        parts.append(f"Fraud risk probability estimated at {prob:.2f} ({risk_cat} risk).")
        if risks:
            parts.append(f"Key risk signals: {'; '.join(risks[:5])}.")
        if positive:
            parts.append(f"Positive evidence: {'; '.join(positive[:3])}.")
        # Evidence citations
        if evidence_refs:
            srcs = [e.get("source") or e.get("type") or "rule" for e in evidence_refs[:3]]
            parts.append(f"Evidence sources: {', '.join(srcs)}.")
        parts.append(f"Recommended operational decision: {decision}.")

        if decision=="REJECT_OR_ESCALATE":
            parts.append("This claim requires escalation and thorough human review before any rejection; high-impact decisions must not be automated solely on model output.")
        elif decision=="FLAG_FOR_MANUAL_REVIEW":
            parts.append("Flagged for manual review due to moderate risk or validation issues; reviewer should verify documents and policy coverage.")
        else:
            parts.append("No strong fraud indicators detected; may approve subject to routine audit and document checks.")

        parts.append("Disclaimer: This result is decision support and not a final legal or insurance determination. Human reviewer must remain involved.")

        return " ".join(parts)

# RAG pipeline wrapper
class AgenticRAGWorkflow:
    def __init__(self, config, kb_dir: Path, history_df: 'pd.DataFrame'=None):
        self.config = config
        self.kb = KnowledgeBase(kb_dir, config)
        self.retriever = EvidenceRetrievalAgent(self.kb, config)
        self.doc_agent = DocumentVerificationAgent()
        self.policy_agent = PolicyRuleMatchingAgent(self.retriever)
        self.consistency_agent = ClaimConsistencyAgent()
        self.history_agent = HistoricalPatternAgent(config, history_df)
        self.synthesis_agent = DecisionSynthesisAgent()
        self.explanation_agent = ExplanationGenerationAgent()

        # LLM optional
        self.llm_enabled = os.getenv("LLM_ENABLED", str(config.get("rag",{}).get("llm_enabled","false"))).lower() in ["true","1"]
        self.llm_provider = os.getenv("LLM_PROVIDER","none")

    def run(self, claim: Dict, doc_validation: Dict, doc_fields: Dict, model_prob: float, anomaly_score: float) -> Dict:
        # Agents
        doc_res = self.doc_agent.check(claim, doc_validation)
        policy_res = self.policy_agent.check(claim)
        consistency_res = self.consistency_agent.check(claim, doc_fields)
        history_res = self.history_agent.check(claim, anomaly_score, model_prob)

        all_results = [doc_res, policy_res, consistency_res, history_res]

        synthesis = self.synthesis_agent.synthesize(all_results, model_prob, anomaly_score, self.config)
        explanation = self.explanation_agent.generate(synthesis, all_results)

        # Evidence retrieval for audit
        # Retrieve additional evidence for each risk signal?
        # We'll include retrieved docs from policy agent already

        # Optional LLM refinement (deterministic fallback if no API)
        llm_output = None
        if self.llm_enabled and os.getenv("LLM_API_KEY"):
            # Placeholder - do not call external by default
            enable_external = os.getenv("ENABLE_EXTERNAL_API_CALLS","false").lower()=="true"
            if enable_external:
                llm_output = {"note":"LLM refinement would happen here, but keeping deterministic for privacy"}
            else:
                llm_output = {"note":"LLM disabled for privacy - deterministic fallback used"}

        final = {
            "claim_id": claim.get("ClaimID","unknown"),
            "model_version": "agentic_rag_v1.0",
            "fraud_probability": model_prob,
            "anomaly_score": anomaly_score,
            "agent_results": all_results,
            "synthesis": synthesis,
            "explanation": explanation,
            "evidence_references": synthesis.get("aggregated_evidence",[]),
            "retrieved_policy_docs": policy_res.get("retrieved",[]),
            "llm": llm_output,
            "timestamp": datetime.utcnow().isoformat(),
            "disclaimer": "Decision-support prototype. Not final legal/insurance determination. Human review mandatory.",
            "auditable_summary": {
                "observed_evidence": [e for e in synthesis.get("aggregated_evidence",[])[:5]],
                "applied_rule": "Policy rule: pre-auth >$10000, duplicate check, bill total matching, peer amount comparison",
                "risk_signal": synthesis.get("aggregated_risk_signals",[]),
                "model_result": f"fraud_prob={model_prob} risk={synthesis.get('risk_category')}",
                "recommended_action": synthesis.get("recommended_decision"),
                "source_reference": [r.get("source") for r in policy_res.get("retrieved",[])[:3]]
            }
        }
        return final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--claim_id", type=str, default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else PROJECT_ROOT/"config.yaml")
    kb_dir = PROJECT_ROOT / config.get("rag",{}).get("knowledge_base_dir","data/sample/knowledge_base")

    # Load sample history for pattern agent
    try:
        import pandas as pd
        history_path = PROJECT_ROOT / config.get("dataset",{}).get("processed_path","data/processed/claims_processed.csv")
        if history_path.exists():
            df_hist = pd.read_csv(history_path)
        else:
            df_hist = None
    except Exception:
        df_hist = None

    workflow = AgenticRAGWorkflow(config, kb_dir, df_hist)

    # Load sample claim
    try:
        import pandas as pd
        sample_csv = PROJECT_ROOT/"data/sample/sample_100.csv"
        if sample_csv.exists():
            df = pd.read_csv(sample_csv)
            if args.claim_id:
                claim = df[df["ClaimID"]==args.claim_id].iloc[0].to_dict() if (df["ClaimID"]==args.claim_id).any() else df.iloc[0].to_dict()
            else:
                claim = df.iloc[0].to_dict()
        else:
            claim = {"ClaimID":"test123","ClaimAmount":5000,"ClaimType":"Inpatient","ProcedureCode":"iO013","DiagnosisCode":"Ta150","ProviderSpecialty":"Orthopedics","PatientAge":45,"PatientIncome":50000,"ClaimDate":"2024-07-08"}
    except Exception as e:
        claim = {"ClaimID":"test123","ClaimAmount":5000,"ClaimType":"Inpatient","ProcedureCode":"iO013","DiagnosisCode":"Ta150","ProviderSpecialty":"Orthopedics","PatientAge":45,"PatientIncome":50000,"ClaimDate":"2024-07-08"}

    # Mock doc validation
    doc_validation = {
        "missing_documents":[],
        "duplicates":[],
        "overall_validation_status":"PASSED",
        "document_results":[]
    }
    doc_fields = {"bill_total": claim.get("ClaimAmount"), "diagnosis_code": claim.get("DiagnosisCode")}

    # Mock model prob and anomaly from previous steps if available, else random moderate
    model_prob = 0.65
    anomaly_score = 0.7

    result = workflow.run(claim, doc_validation, doc_fields, model_prob, anomaly_score)

    eval_dir = PROJECT_ROOT / config.get("paths",{}).get("evaluation_dir","evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_json(result, eval_dir/"agentic_rag_sample_output.json")

    doc_dir = PROJECT_ROOT / config.get("paths",{}).get("documentation_dir","documentation")
    doc_dir.mkdir(parents=True, exist_ok=True)
    with open(doc_dir/"agentic_rag.md","w") as f:
        f.write("# Agentic RAG Reasoning\n\n")
        f.write("## Agents\n- Document Verification\n- Policy Rule Matching\n- Claim Consistency\n- Historical Pattern/Anomaly\n- Evidence Retrieval\n- Decision Synthesis\n- Explanation Generation\n\n")
        f.write("## RAG\n")
        f.write(f"KB dir {kb_dir} chunks {len(workflow.kb.chunks)}\n")
        f.write("Chunk size 500 overlap 50 TFIDF fallback or sentence-transformers if configured.\n\n")
        f.write("## Grounding\nEvery explanation grounded in retrieved evidence, extracted fields, model outputs, rules.\n\n")
        f.write("## LLM\nOptional, controlled via LLM_ENABLED env. Deterministic fallback used by default.\n\n")
        f.write("## Sample Output\n")
        f.write(json.dumps(result, indent=2)[:5000])

    logger.info("Agentic RAG completed")
    print(json.dumps(result, indent=2)[:5000])

if __name__ == "__main__":
    main()
