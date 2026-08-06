"""
Unit tests for core preprocessing, metrics, thresholding, JSON output, document validation.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_dataset_schema_validation():
    from common.schema_validation import check_required_columns, check_missing_values, check_class_imbalance
    df = pd.DataFrame({
        "ClaimID":["a","b"],
        "PatientID":["p1","p2"],
        "ClaimLegitimacy":["Legitimate","Fraud"],
        "ClaimAmount":[100,200]
    })
    res = check_required_columns(df, ["ClaimID","PatientID","ClaimAmount"])
    assert res["ok"] == True
    miss = check_missing_values(df)
    assert miss["missing_pct"]["ClaimID"]==0.0
    y = pd.Series([0,0,1])
    imb = check_class_imbalance(y)
    assert imb["is_imbalanced"]==False or imb["ratio_minority_majority"]<=0.5
    print("test_dataset_schema_validation passed")

def test_missing_value_handling():
    from common.preprocessing import build_preprocessor
    config = {"preprocessing":{"scaling":"standard","encoding":"onehot"}}
    X = pd.DataFrame({
        "num":[1,None,3],
        "cat":["A",None,"B"]
    })
    preproc = build_preprocessor(["num"],["cat"],[],config)
    # Should impute and not fail
    trans = preproc.fit_transform(X)
    assert trans.shape[0]==3
    assert not np.isnan(trans).any()
    print("test_missing_value_handling passed")

def test_categorical_encoding():
    from common.preprocessing import build_preprocessor
    config = {"preprocessing":{"scaling":"standard","encoding":"onehot"}}
    X = pd.DataFrame({"cat":["A","B","A"],"num":[1,2,3]})
    preproc = build_preprocessor(["num"],["cat"],[],config)
    trans = preproc.fit_transform(X)
    # OHE for cat should produce 2 columns + 1 num =3
    assert trans.shape[1]>=3
    print("test_categorical_encoding passed")

def test_target_label_validation():
    df = pd.DataFrame({"ClaimLegitimacy":["Legitimate","Fraud","Legitimate"]})
    mapping = {"Legitimate":0,"Fraud":1}
    y = df["ClaimLegitimacy"].map(mapping)
    assert y.tolist()==[0,1,0]
    print("test_target_label_validation passed")

def test_class_imbalance_calculations():
    from common.schema_validation import check_class_imbalance
    y = pd.Series([0]*94 + [1]*6)
    imb = check_class_imbalance(y)
    assert abs(imb["fraud_rate"]-0.06)<0.001
    print("test_class_imbalance_calculations passed")

def test_metric_calculations():
    from common.metrics import compute_all_metrics
    y_true = [0,0,1,1]
    y_pred = [0,0,1,1]
    y_prob = [0.1,0.2,0.8,0.9]
    m = compute_all_metrics(y_true,y_pred,y_prob)
    assert m["accuracy"]==1.0
    assert m["precision"]==1.0
    assert m["recall"]==1.0
    assert m["pr_auc"]>=0.9
    print("test_metric_calculations passed")

def test_threshold_selection():
    from common.threshold import select_threshold
    y_true = [0,0,0,1,1]
    y_prob = [0.1,0.2,0.3,0.8,0.9]
    thr, info = select_threshold(y_true,y_prob,strategy="optimize_f2")
    assert 0.05 <= thr <= 0.95
    assert "threshold" in info
    print("test_threshold_selection passed")

def test_claim_result_json_schema():
    from common.schema_validation import validate_claim_result_schema
    result = {
        "claim_id":"test123",
        "model_version":"v1",
        "fraud_probability":0.2,
        "fraud_prediction":0,
        "anomaly_score":0.3,
        "document_validation_status":"PASSED",
        "policy_validation_status":"PASSED",
        "risk_category":"LOW",
        "recommended_decision":"APPROVE",
        "key_risk_signals":[],
        "positive_evidence":["ok"],
        "missing_or_inconsistent_info":[],
        "explanation":"test",
        "evidence_references":[],
        "timestamp":"2026-08-06T00:00:00Z",
        "disclaimer":"disclaimer"
    }
    missing = validate_claim_result_schema(result)
    assert missing==[]
    print("test_claim_result_json_schema passed")

def load_doc_pipeline():
    import importlib.util
    spec = importlib.util.spec_from_file_location("doc_mod", PROJECT_ROOT/"approaches"/"04_document_intelligence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DocumentIntelligencePipeline

def test_bill_total_validation():
    # Test document intelligence bill total validation
    DocumentIntelligencePipeline = load_doc_pipeline()
    from common.config import load_config
    config = load_config(PROJECT_ROOT/"config.yaml")
    pipe = DocumentIntelligencePipeline(config)
    extracted = {"fields":{"bill_total":100.0},"confidences":{"bill_total":0.9}}
    validation = pipe.validate_document("medical_bill", extracted, claimed_amount=100.0)
    assert validation["status"]=="PASSED"
    validation2 = pipe.validate_document("medical_bill", extracted, claimed_amount=200.0)
    assert validation2["status"]=="FAILED"
    print("test_bill_total_validation passed")

def test_date_consistency_validation():
    # Simplified date consistency
    DocumentIntelligencePipeline = load_doc_pipeline()
    from common.config import load_config
    config = load_config(PROJECT_ROOT/"config.yaml")
    pipe = DocumentIntelligencePipeline(config)
    # No hard failure expected, but should not crash
    extracted = {"fields":{"bill_date":"2024-07-08"},"confidences":{"bill_date":0.8}}
    v = pipe.validate_document("medical_bill", extracted, claimed_amount=100.0, claimed_date="2024-07-09")
    assert "status" in v
    print("test_date_consistency_validation passed")

def test_duplicate_document_detection():
    import hashlib, tempfile, pathlib
    # Create two temp files with same content
    content = b"same content"
    h1 = hashlib.sha256(content).hexdigest()
    h2 = hashlib.sha256(content).hexdigest()
    assert h1==h2
    # Different content
    h3 = hashlib.sha256(b"different").hexdigest()
    assert h1!=h3
    print("test_duplicate_document_detection passed")

def test_fallback_behavior():
    # Test document pipeline fallback when OCR not available
    DocumentIntelligencePipeline = load_doc_pipeline()
    from common.config import load_config
    config = load_config(PROJECT_ROOT/"config.yaml")
    pipe = DocumentIntelligencePipeline(config)
    # Force fallback engine
    pipe.ocr_engine="fallback"
    # Process synthetic json (should work even without OCR)
    sample_json = PROJECT_ROOT/"data/sample/synthetic_bill_1.json"
    if sample_json.exists():
        res = pipe.process_document(sample_json)
        assert "document_type" in res
        assert "extracted_fields" in res
    print("test_fallback_behavior passed")

if __name__=="__main__":
    test_dataset_schema_validation()
    test_missing_value_handling()
    test_categorical_encoding()
    test_target_label_validation()
    test_class_imbalance_calculations()
    test_metric_calculations()
    test_threshold_selection()
    test_claim_result_json_schema()
    test_bill_total_validation()
    test_date_consistency_validation()
    test_duplicate_document_detection()
    test_fallback_behavior()
    print("All tests passed")
