# System Architecture

## High-Level
```
[Claim Input: claimant, policy, incident, docs]
-> [Document Intelligence: OCR (Tesseract/EasyOCR/PaddleOCR/fallback) + VLM optional + Validation]
-> [Preprocessing: Scaling, Encoding, Date Engineering]
-> [ML Layer: Traditional ML + Deep Learning MLP + Anomaly Detection]
-> [Policy & RAG: Policy Rules, Exclusion, Fraud Indicators, Guidelines retrieval]
-> [Agentic Reasoning: 7 Agents]
-> [Hybrid Synthesis: Weighted prod + rules]
-> [Explainability: SHAP, Importance, Evidence Citations]
-> [Output: APPROVE / FLAG_FOR_MANUAL_REVIEW / REJECT_OR_ESCALATE + Explanation + Disclaimer]
```

## Components

### Data
- Raw Excel 4500 rows
- Processed CSV
- Sample 100
- Synthetic bills, prescriptions, discharge
- Knowledge base: policy_rules.txt, etc in data/sample/knowledge_base

### Common Utilities
- config.py, logging_utils.py, seed.py, dataset_loader.py, schema_validation.py, preprocessing.py, metrics.py, threshold.py, explainability.py, artifacts.py, serialization.py, result_formatting.py

### Approaches (6 files)
See individual docs.

### Evaluation
- metrics_summary.csv, model_comparison.csv, per_class, confusion_matrices/, threshold_analysis.csv, calibration_results.csv, runtime_comparison.csv, data_quality_report.md, experiment_log.md, evaluation_protocol.md, model_selection.md, run_metadata.json

### Images
- Generated via visualization_generator.py from actual data/eval files

### Relations
- schema.md, ER diagram mmd/png, feature_relationships.csv, correlation_analysis.md, data_lineage.md

### Deployment
- requirements.txt, config.yaml, .env.example, run_pipeline.py, run_all_experiments.py, Makefile
- api/: sample_request.json, sample_response.json, README.md, optional FastAPI service (not required)
- Next.js integration via documented JSON contract

### Diagram Files
- `images/architecture_diagram.png`
- `images/document_validation_flow.png`
- `relations/entity_relationship_diagram.png`

## Next.js Integration
- Frontend sends claim JSON + document paths/hashes to backend
- Backend runs hybrid pipeline (common utilities reused)
- Returns structured result JSON (see api_contract.md)
- Frontend displays risk category, explanation, evidence refs, document validation status, human review required flag

## Security
- ANONYMIZE_PII true
- No hard-coded API keys
- ENABLE_EXTERNAL_API_CALLS false by default
- Docs synthetic, IDs hashed
- Encryption and access control future work

## Reproducibility
- Python 3.10+, seed 42, config.yaml, requirements.txt
