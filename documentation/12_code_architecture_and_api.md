# Chapter 12: Code Architecture, APIs and System Design

**Institution:** IIIT Dharwad — Department of Data Science & AI  
**Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

---

## 12.1 Codebase Organization
The codebase is structured into modular, production-ready components:
- `src/config.py`: Global configuration, paths, seeds, hospital tiers, and diagnostic categories.
- `src/utils.py`: Logging engine, metrics computation (F2, MCC, Brier score), INR cost matrix, McNemar test, and Wilcoxon signed-rank tests.
- `src/data_loader.py`: Raw dataset loading, statistical validation, and Indian synthetic claims generator.
- `src/preprocessing.py`: Imputation, outlier handling, categorical encodings, feature scaling, and class imbalance resamplers.
- `src/feature_engineering.py`: Domain ratios, treatment cost deviation Z-scores, claim velocity, and automated feature selection.
- `src/models_ml.py` & `src/train_ml.py`: 12+ traditional ML models, grid search tuning, and threshold optimization.
- `src/models_dl.py` & `src/train_dl.py`: 10 tabular deep learning architectures in PyTorch with Focal Loss and MC Dropout.
- `src/explainability.py`: SHAP, LIME, attention masks, and counterfactual recommendations.
- `src/agent_system/`: Multi-agent orchestration, SQLite local DB, RAG engine, OCR/VLM agent, policy agent, anomaly agent, and reasoning agent.
- `src/api.py` & `src/web_app.py`: FastAPI REST API and responsive web dashboard bound to `0.0.0.0:8000`.
- `run_pipeline.py`: Unified master script executing the complete pipeline end-to-end.
