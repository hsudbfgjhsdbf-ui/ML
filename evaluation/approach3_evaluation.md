# APPROACH 3: AGENT AI / MULTI-AGENT SYSTEM EVALUATION REPORT
**Institution:** IIIT Dharwad | B.Tech Data Science & AI  
**Faculty Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

## 1. Cognitive Multi-Agent System Verification Summary
The Agent AI system integrates five specialized LangGraph agents (DocumentProcessing, PolicyVerification, AnomalyDetection, HistoricalPattern, ExplainableReasoning) with a local SQLite database and TF-IDF/Vector RAG pipeline.

### 1.1 Key Performance Advantages over Numerical Classifiers
1. **Direct Document Reasoning:** Vision Language Models extract structured JSON directly from hospital bills and prescriptions.
2. **Policy Clause Attribution:** RAG cites exact policy clauses (e.g., `[CLAUSE-ROOM-001] Room Rent Capping`) when checking limits.
3. **Transparent Explanations:** Produces human-readable natural language decisions with full legal grounding.