# Experiment Log

Auto-generated log of experiments. See also documentation/experiment_tracking.md for detailed.

## 2026-08-06 10:53 Run 01_traditional_ml quick
- Models: Dummy, Logistic, DecisionTree, RandomForest, HistGradientBoosting
- Best: DecisionTree PR-AUC 1.0 val thr 0.95
- Test: accuracy 0.9989 precision 0.9818 recall 1.0 F1 0.9908 PR 0.9818 ROC 0.9994
- Artifacts: best_traditional_ml_model.joblib
- Notes: near-perfect suggests synthetic separable income+amount

## 2026-08-06 10:58 Run 02_deep_learning
- Framework sklearn_mlp_fallback torch not available
- Hidden [128,64,32] dropout 0.3 lr 0.001 batch 64 epochs 100 early stopping 10
- SMOTE 2925->5498
- Val PR-AUC 0.8729 Test 0.9026 thr 0.05 low due to underconfidence
- Observation DL does NOT outperform trees on tabular

## 2026-08-06 10:58 Run 03_anomaly_detection
- Train only legit 3384
- Contamination 0.06
- IsolationForest PR 0.07 ROC 0.487 Prec@10 0.2 Rec@200 0.185
- LOF PR 0.1468 ROC 0.737 Prec@10 0.2 Rec@200 0.518 best
- OneClassSVM PR 0.1228 ROC 0.679
- Ensemble PR 0.129 ROC 0.656
- EllipticEnvelope failed memory 7.19 GiB for 9822 features
- Autoencoder torch missing
- Anomaly != fraud prob

## 2026-08-06 11:03 Run 04_document_intelligence
- OCR fallback VLM disabled
- Processed 4 synthetic docs 2 bills prescription discharge
- Bill total mismatch detected HIGH risk FAILED
- Missing docs flagged
- Output evaluation/document_intelligence_sample_output.json

## 2026-08-06 11:03 Run 05_agentic_rag
- KB 5 chunks TFIDF fallback
- Retrieved policy_rules coverage claim_guidelines fraud_indicators exclusion
- Sample claim prob 0.65 moderate
- All agents PASSED initially

## 2026-08-06 11:03 Run 06_hybrid
- Loaded best traditional DecisionTree
- Anomaly models IsolationForest LOF OneClassSVM
- Preprocessor anomaly
- Doc pipeline fallback RAG TFIDF
- Sample claim 6eea92b2 amount 1703 legit predicted prob 0.0 anomaly 0.404 doc FAILED HIGH risk FLAG_FOR_MANUAL_REVIEW

## Visualization
- visualization_generator.py generated images/ class_distribution missing numerical fraud comparison correlation fraud rate by specialty/type/status/gender model comparison runtime feature importance confusion threshold performance anomaly distribution architecture document flow entity diagram

## Presentation
- presentation/generate_presentation.py generated 20 slides pptx 580K with actual eval numbers or Pending

## Report
- report/generate_report.py generated PDF 414K with sections cover abstract problem motivation objectives dataset dictionary quality relationships feature engineering methodologies architecture implementation training evaluation benchmark error explainability security fairness limitations future conclusion references appendix

## Next Steps
- Run full traditional ML (all models) when time permits: python approaches/01_traditional_ml.py (without --quick)
- Install optional deps: pip install -r requirements_optional.txt for torch xgboost lightgbm catboost OCR LLM embeddings
- Run tests: python tests/test_basic.py

