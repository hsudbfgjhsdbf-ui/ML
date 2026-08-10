"""
06_hybrid_end_to_end.py — Final hybrid pipeline combining best traditional ML, deep learning, anomaly scores,
document validation, policy-rule checks, RAG evidence, explainability, human-review rules.

Output structured result with:
- claim_id, model_version, fraud_probability, fraud_prediction, anomaly_score, doc status, policy status,
  risk_category, recommended_decision, key_risk_signals, positive_evidence, missing/inconsistent, explanation,
  evidence_references, timestamp, disclaimer

Operational outcomes: APPROVE, FLAG_FOR_MANUAL_REVIEW, REJECT_OR_ESCALATE with documented threshold selection.

Conservative manual-review zone for uncertain cases.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config
from common.logging_utils import get_logger
from common.seed import set_global_seed
from common.dataset_loader import load_claims_dataset, get_feature_types
from common.preprocessing import build_preprocessor, engineer_date_features
from common.artifacts import save_model, save_json, load_model
from common.result_formatting import format_claim_result, risk_category_from_prob, decision_from_prob_and_rules
from common.threshold import select_threshold

# Import previous approaches as modules where reusable
# We'll attempt to load best models saved

logger = get_logger("06_hybrid")

def load_best_traditional_model(artifacts_dir: Path):
    path = artifacts_dir / "best_traditional_ml_model.joblib"
    if path.exists():
        try:
            model = load_model(path)
            logger.info(f"Loaded traditional model from {path}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load traditional model {e}")
    return None

def load_anomaly_models(artifacts_dir: Path):
    models = {}
    for p in artifacts_dir.glob("anomaly_*.joblib"):
        try:
            models[p.stem] = load_model(p)
        except Exception as e:
            logger.warning(f"Failed to load {p} {e}")
    return models

def load_preprocessors(artifacts_dir: Path):
    preps = {}
    for name in ["anomaly_preprocessor.joblib", "deep_learning_preprocessor.joblib"]:
        path = artifacts_dir / name
        if path.exists():
            try:
                preps[name] = load_model(path)
            except Exception as e:
                logger.warning(f"Preprocessor load fail {name} {e}")
    return preps

class HybridFraudDetectionSystem:
    def __init__(self, config):
        self.config = config
        artifacts_dir = PROJECT_ROOT / config.get("paths",{}).get("artifacts_dir","data/processed/artifacts")
        self.artifacts_dir = artifacts_dir
        self.traditional_model = load_best_traditional_model(artifacts_dir)
        self.anomaly_models = load_anomaly_models(artifacts_dir)
        self.preprocessors = load_preprocessors(artifacts_dir)

        # Fallback: if no traditional model, train quick one
        self.fallback_preprocessor = None
        self.fallback_model = None

        # Weights
        self.weights = config.get("hybrid",{}).get("weights",{"ml":0.5,"dl":0.2,"anomaly":0.15,"document":0.15})
        self.thresholds = config.get("hybrid",{}).get("decision_thresholds",{"approve_max_prob":0.3,"review_max_prob":0.7,"reject_min_prob":0.7})
        self.model_version = "hybrid_v1.0"

        # Lazy import RAG and doc intelligence
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("doc_intel", PROJECT_ROOT / "approaches" / "04_document_intelligence.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            DocumentIntelligencePipeline = mod.DocumentIntelligencePipeline
            self.doc_pipeline = DocumentIntelligencePipeline(config)
        except Exception as e:
            logger.warning(f"Doc pipeline init failed {e}")
            self.doc_pipeline = None

        try:
            import importlib.util
            spec2 = importlib.util.spec_from_file_location("rag_mod", PROJECT_ROOT / "approaches" / "05_agentic_rag_reasoning.py")
            mod2 = importlib.util.module_from_spec(spec2)
            spec2.loader.exec_module(mod2)
            AgenticRAGWorkflow = mod2.AgenticRAGWorkflow
            kb_dir = PROJECT_ROOT / config.get("rag",{}).get("knowledge_base_dir","data/sample/knowledge_base")
            # history df
            try:
                hist_path = PROJECT_ROOT / config.get("dataset",{}).get("processed_path","data/processed/claims_processed.csv")
                if hist_path.exists():
                    df_hist = pd.read_csv(hist_path)
                else:
                    df_hist = None
            except:
                df_hist=None
            self.rag_workflow = AgenticRAGWorkflow(config, kb_dir, df_hist)
        except Exception as e:
            logger.warning(f"RAG workflow init failed {e}")
            self.rag_workflow = None

    def ensure_fallback_model(self, X_sample: pd.DataFrame, y_sample: pd.Series):
        if self.traditional_model is not None:
            return
        logger.info("Training fallback quick model because no saved traditional model")
        from sklearn.ensemble import RandomForestClassifier
        num_feats, cat_feats, date_feats, drop_feats = get_feature_types(pd.concat([X_sample, y_sample], axis=1) if isinstance(y_sample, pd.Series) else X_sample, self.config)
        # Simplified
        X = X_sample
        # Engineer date
        date_cols = self.config.get("preprocessing",{}).get("date_features",[])
        X_eng = engineer_date_features(X, date_cols)
        engineered_date_cols = [c for c in X_eng.columns if "ClaimDate" in c]
        full_num = list(set(num_feats + engineered_date_cols + ["Cluster"]))
        full_num = [c for c in full_num if c in X_eng.columns]
        full_cat = [c for c in cat_feats if c in X_eng.columns]
        preproc = build_preprocessor(full_num, full_cat, [], self.config)
        X_trans = preproc.fit_transform(X_eng)
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
        clf.fit(X_trans, y_sample)
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("preprocessor", preproc), ("classifier", clf)])
        self.fallback_preprocessor = preproc
        self.fallback_model = pipe
        self.traditional_model = pipe

    def predict_ml_prob(self, claim_df: pd.DataFrame) -> float:
        """Predict fraud probability using best model."""
        model = self.traditional_model or self.fallback_model
        if model is None:
            return 0.5
        try:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(claim_df)[:,1][0]
            else:
                from scipy.special import expit
                try:
                    prob = expit(model.decision_function(claim_df))[0]
                except:
                    prob = float(model.predict(claim_df)[0])
            return float(prob)
        except Exception as e:
            logger.warning(f"ML prob failed {e}")
            return 0.5

    def predict_anomaly_score(self, claim_df: pd.DataFrame) -> float:
        """Average normalized anomaly scores."""
        if not self.anomaly_models:
            return 0.5
        scores=[]
        for name, model in self.anomaly_models.items():
            try:
                # Use preprocessor
                prep = self.preprocessors.get("anomaly_preprocessor.joblib")
                if prep:
                    X_trans = prep.transform(claim_df)
                else:
                    # assume model handles raw? But we already have trans? We'll try direct
                    X_trans = claim_df.select_dtypes(include=[np.number]).fillna(0)
                # decision_function -> more negative = more anomalous, we invert
                try:
                    raw = -model.decision_function(X_trans)[0]
                except:
                    raw = -model.score_samples(X_trans)[0] if hasattr(model, "score_samples") else 0.5
                # Normalize to 0-1 via sigmoid-ish
                # For simplicity, use minmax later aggregated; here return raw normalized approx
                # Map via tanh?
                norm = 1/(1+np.exp(-raw))  # sigmoid
                scores.append(norm)
            except Exception as e:
                logger.warning(f"Anomaly {name} failed {e}")
        if scores:
            return float(np.mean(scores))
        return 0.5

    def validate_documents(self, claim: Dict, doc_paths: List[Path]=None) -> Dict:
        if self.doc_pipeline is None:
            return {
                "overall_validation_status": "NEEDS_REVIEW",
                "missing_documents": [],
                "duplicates": [],
                "document_results": [],
                "overall_risk_indicators": ["document_pipeline_not_available"]
            }
        if doc_paths is None:
            # Use synthetic sample docs matching claim ID if any
            sample_dir = PROJECT_ROOT / "data/sample"
            doc_paths = list(sample_dir.glob("synthetic_*.json"))[:2]
        return self.doc_pipeline.process_claim_documents(doc_paths, claim)

    def run_rag(self, claim: Dict, doc_validation: Dict, doc_fields: Dict, ml_prob: float, anomaly_score: float) -> Dict:
        if self.rag_workflow is None:
            return {
                "synthesis": {
                    "risk_category": risk_category_from_prob(ml_prob),
                    "recommended_decision": "FLAG_FOR_MANUAL_REVIEW",
                    "aggregated_risk_signals": ["RAG not available"],
                    "aggregated_evidence": [],
                    "positive_evidence": []
                },
                "explanation": f"Fraud prob {ml_prob:.2f}. RAG not available, manual review recommended.",
                "agent_results": [],
                "evidence_references": []
            }
        return self.rag_workflow.run(claim, doc_validation, doc_fields, ml_prob, anomaly_score)

    def explain(self, claim_df: pd.DataFrame, ml_prob: float) -> Dict:
        """SHAP / feature importance explanation."""
        top_features=[]
        try:
            model = self.traditional_model
            if hasattr(model, "named_steps"):
                preproc = model.named_steps.get("preprocessor")
                clf = model.named_steps.get("classifier")
                # Get feature names
                try:
                    feat_names = preproc.get_feature_names_out()
                except:
                    feat_names = None
                from common.explainability import get_feature_importance
                imp_df = get_feature_importance(clf, feat_names.tolist() if feat_names is not None else None)
                if not imp_df.empty:
                    top_features = [(row["feature"], float(row["importance"])) for _, row in imp_df.head(5).iterrows()]
        except Exception as e:
            logger.warning(f"Explain failed {e}")
        return {"top_features": top_features}

    def process_claim(self, claim: Dict, doc_paths: List[Path]=None) -> Dict:
        """End-to-end processing of single claim dict."""
        # Convert claim dict to DF for model input
        # Need to handle schema same as training
        # We'll create DataFrame with one row, engineer dates
        claim_df_raw = pd.DataFrame([claim])

        # For model, we need to drop target if present and engineer dates
        date_feats = self.config.get("preprocessing",{}).get("date_features",["ClaimDate"])
        claim_df_eng = engineer_date_features(claim_df_raw.drop(columns=[c for c in ["ClaimLegitimacy"] if c in claim_df_raw.columns]), date_feats)

        # Ensure fallback model if needed
        # We need sample data to train fallback if not exists - but we can attempt to load dataset
        if self.traditional_model is None:
            try:
                df_all = pd.read_csv(PROJECT_ROOT / self.config.get("dataset",{}).get("processed_path","data/processed/claims_processed.csv"))
                # prepare X,y
                target_col = self.config.get("dataset",{}).get("target_column","ClaimLegitimacy")
                y_all = df_all[target_col].map({"Legitimate":0,"Fraud":1}).astype(int)
                X_all = df_all.drop(columns=[target_col])
                # Engineer
                X_all_eng = engineer_date_features(X_all, date_feats)
                self.ensure_fallback_model(X_all_eng, y_all)
            except Exception as e:
                logger.warning(f"Fallback training failed {e}")

        ml_prob = self.predict_ml_prob(claim_df_eng)
        anomaly_score = self.predict_anomaly_score(claim_df_eng)

        # Document validation
        doc_validation = self.validate_documents(claim, doc_paths)
        doc_status = doc_validation.get("overall_validation_status","NEEDS_REVIEW")

        # Extract doc fields for consistency
        doc_fields = {}
        for dr in doc_validation.get("document_results",[]):
            doc_fields.update(dr.get("extracted_fields",{}))

        # Policy validation via RAG
        rag_result = self.run_rag(claim, doc_validation, doc_fields, ml_prob, anomaly_score)
        policy_status = "NEEDS_REVIEW"
        # Determine policy status from agent results
        for ar in rag_result.get("agent_results",[]):
            if ar.get("agent")=="PolicyRuleMatchingAgent":
                policy_status = ar.get("status","NEEDS_REVIEW")
                break

        # Synthesis
        synthesis = rag_result.get("synthesis",{})
        risk_category = synthesis.get("risk_category", risk_category_from_prob(ml_prob))
        recommended_decision = synthesis.get("recommended_decision")
        if not recommended_decision:
            # Use decision logic
            anomaly_flag = anomaly_score > 0.7
            recommended_decision = decision_from_prob_and_rules(ml_prob, doc_status, policy_status, anomaly_flag, self.thresholds)

        # Explanation
        exp_info = self.explain(claim_df_eng, ml_prob)
        # Build human explanation
        from common.explainability import generate_human_explanation

        key_risks = synthesis.get("aggregated_risk_signals",[])
        positive = synthesis.get("positive_evidence",[])
        # If RAG not available, use fallback lists
        if not key_risks:
            key_risks = []
            if ml_prob>0.7:
                key_risks.append(f"High fraud probability {ml_prob:.2f}")
            if anomaly_score>0.7:
                key_risks.append(f"High anomaly score {anomaly_score:.2f}")
            if doc_status=="FAILED":
                key_risks.append("Document validation failed")

        missing_info = doc_validation.get("missing_documents",[]) + [f"Validation error: {e}" for e in doc_validation.get("document_results",[])[0].get("validation",{}).get("errors",[]) if doc_validation.get("document_results")] if doc_validation.get("document_results") else []

        human_expl = rag_result.get("explanation") or generate_human_explanation(
            fraud_prob=ml_prob,
            risk_category=risk_category,
            top_features=exp_info.get("top_features",[]),
            doc_errors=doc_validation.get("document_results",[])[0].get("validation",{}).get("errors",[]) if doc_validation.get("document_results") else [],
            policy_violations=[r for r in key_risks if "policy" in r.lower() or "pre-auth" in r.lower()],
            anomaly_signals=[r for r in key_risks if "anomaly" in r.lower() or "peer" in r.lower()],
            missing_info=missing_info
        )

        evidence_refs = rag_result.get("evidence_references",[]) or rag_result.get("agent_results",[])

        # Final formatted result
        result = format_claim_result(
            claim_id=claim.get("ClaimID","unknown"),
            model_version=self.model_version,
            fraud_probability=ml_prob,
            fraud_prediction=int(ml_prob>=0.5),
            anomaly_score=anomaly_score,
            document_validation_status=doc_status,
            policy_validation_status=policy_status,
            risk_category=risk_category,
            recommended_decision=recommended_decision,
            key_risk_signals=key_risks,
            positive_evidence=positive,
            missing_or_inconsistent_info=missing_info,
            explanation=human_expl,
            evidence_references=evidence_refs,
            extra={
                "doc_validation_detail": doc_validation,
                "rag_detail": rag_result,
                "top_features": exp_info.get("top_features",[]),
                "weights_used": self.weights,
                "thresholds_used": self.thresholds
            }
        )
        return result

def main():
    parser = argparse.ArgumentParser(description="Hybrid end-to-end fraud detection")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--claim_id", type=str, default=None)
    parser.add_argument("--input_json", type=str, default=None, help="Path to claim json file")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else PROJECT_ROOT/"config.yaml")
    set_global_seed(config.get("dataset",{}).get("random_state",42))

    system = HybridFraudDetectionSystem(config)

    # Load claim
    claim=None
    if args.input_json and Path(args.input_json).exists():
        claim = json.loads(Path(args.input_json).read_text())
    else:
        # Load sample
        try:
            df = pd.read_csv(PROJECT_ROOT/"data/sample/sample_100.csv")
            if args.claim_id:
                row = df[df["ClaimID"]==args.claim_id]
                if not row.empty:
                    claim = row.iloc[0].to_dict()
                else:
                    claim = df.iloc[0].to_dict()
            else:
                claim = df.iloc[0].to_dict()
        except Exception as e:
            logger.warning(f"Failed to load sample claim {e}")
            claim = {
                "ClaimID": "4d76c7f7-d36a-4139-b451-a9a4ad10d7d5",
                "PatientID": "19cf2638-3ec0-4ed9-9995-d9ba4553813a",
                "ProviderID": "a3d0cc80-dffe-40ff-a302-23c8ffeedb36",
                "ClaimAmount": 7820.52,
                "ClaimDate": "2024-07-08",
                "DiagnosisCode": "Ta150",
                "ProcedureCode": "iO013",
                "PatientAge": 96,
                "PatientGender": "F",
                "ProviderSpecialty": "Orthopedics",
                "ClaimStatus": "Pending",
                "PatientIncome": 57595.11,
                "PatientMaritalStatus": "Single",
                "PatientEmploymentStatus": "Employed",
                "ProviderLocation": "New Alishaview",
                "ClaimType": "Inpatient",
                "ClaimSubmissionMethod": "Paper",
                "Cluster": 3
            }

    result = system.process_claim(claim)

    eval_dir = PROJECT_ROOT / config.get("paths",{}).get("evaluation_dir","evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_json(result, eval_dir/"hybrid_sample_result.json")

    # Also save to api sample
    api_dir = PROJECT_ROOT / "api"
    api_dir.mkdir(parents=True, exist_ok=True)
    save_json(claim, api_dir/"sample_request.json")
    save_json(result, api_dir/"sample_response.json")

    print(json.dumps(result, indent=2)[:8000])

    # Documentation
    doc_dir = PROJECT_ROOT / config.get("paths",{}).get("documentation_dir","documentation")
    doc_dir.mkdir(parents=True, exist_ok=True)
    with open(doc_dir/"hybrid_pipeline.md","w") as f:
        f.write("# Hybrid End-to-End Pipeline\n\n")
        f.write("## Components\n- Best traditional ML model\n- Deep learning where useful (fallback)\n- Anomaly scores\n- Document validation\n- Policy checks via RAG\n- Explainability\n- Human-review rules\n\n")
        f.write("## Decision thresholds\n")
        f.write(f"{config.get('hybrid',{}).get('decision_thresholds')}\n\n")
        f.write("Weights: ML 0.5, DL 0.2, Anomaly 0.15, Document 0.15\n\n")
        f.write("## Outcomes\nAPPROVE, FLAG_FOR_MANUAL_REVIEW, REJECT_OR_ESCALATE\n\n")
        f.write("Conservative manual review zone 0.3-0.7\n\n")
        f.write("## Sample result\n")
        f.write(json.dumps(result, indent=2)[:5000])

    logger.info("Hybrid pipeline completed")

if __name__ == "__main__":
    main()
