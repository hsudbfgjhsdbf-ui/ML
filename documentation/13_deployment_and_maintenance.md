# Chapter 13: Deployment, Production Monitoring & Technical Debt

**Institution:** IIIT Dharwad — Department of Data Science & AI  
**Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

---

## 13.1 Production Architecture & Scalability
- **Real-Time Claim Screening:** FastAPI async endpoints provide ultra-low latency inference (0.8 ms for LightGBM, 4.8 ms for FT-Transformer).
- **Batch Processing:** Support for bulk batch claim auditing capable of screening 100,000 claims/hour.
- **Model Versioning & Serialization:** All model artifacts and preprocessing pipelines are persisted with metadata in `saved_models/` (`.joblib` and `.pt`).
- **Drift Detection:** Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI) monitoring incoming claim amount distributions to detect seasonal drift and evolving fraud patterns.
