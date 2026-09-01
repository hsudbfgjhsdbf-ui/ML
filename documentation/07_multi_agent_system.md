# Chapter 7: Multi-Agent Cognitive Verification System and RAG

**Institution:** IIIT Dharwad — Department of Data Science & AI  
**Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

---

## 7.1 Multi-Agent Cognitive Architecture (Approach 3)
Approach 3 introduces an agentic AI verification ecosystem organized as a stateful graph (LangGraph pattern):

1. **Coordinator Agent:**
   - Manages workflow state, routes tasks dynamically, handles API retries with exponential backoff, and compiles the final audit report.
2. **Document Processing & OCR/VLM Agent:**
   - Multi-modal extraction from scanned bills, prescriptions, and discharge summaries into structured JSON schemas with field-level confidence scores.
3. **Policy Verification & RAG Agent:**
   - Grounded knowledge retrieval over IRDAI circulars, waiting period clauses, and co-payment schedules to confirm contractual validity.
4. **Clinical Anomaly Detection Agent:**
   - Evaluates claimed billing amounts against standard Schedule of Charges (SOC) for that specific hospital tier and diagnosis.
5. **Historical Pattern Agent:**
   - Evaluates longitudinal claim frequency, prior rejection ratios, and provider collusion risks.
6. **Reasoning & Decision Agent:**
   - Synthesizes findings from all specialized agents, resolves conflicting evidence, and produces structured verdicts (Approved, Flagged for Manual Review, Rejected) with layered bilingual natural language justifications (English & Hindi).

---

## 7.2 Communication Protocol & Audit Traceability
Every agent interaction produces a standardized message payload recorded in the SQLite relational `agent_results` table:
```json
{
  "agent_name": "PolicyVerificationAgent",
  "status": "COMPLIANT",
  "confidence": 0.95,
  "verification_checks": [
    {
      "check_name": "Sum Insured Coverage Check",
      "status": "PASS",
      "clause_ref": "Clause 2.1"
    }
  ]
}
```
This architecture guarantees full regulatory compliance under IRDAI guidelines while maintaining complete transparency for claimants.
