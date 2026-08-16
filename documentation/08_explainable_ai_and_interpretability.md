# Chapter 8: Explainable AI (XAI) and Interpretability

**Institution:** IIIT Dharwad — Department of Data Science & AI  
**Adviser:** Prof. Ramesh Athe  
**Team:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  

---

## 8.1 Multi-Layered Explainability Suite
To eliminate black-box opacity in claim adjudication, our platform implements four complementary interpretability techniques:

1. **SHAP (SHapley Additive exPlanations):**
   - TreeExplainer for tree ensembles and DeepExplainer for neural networks.
   - Computes local and global Shapley values guaranteeing efficiency, symmetry, and additivity:
     $$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$
   - Highlights that Treatment Cost Deviation, Claim-to-Premium Ratio, and Hospital Rejection History are the primary global drivers of fraud predictions.

2. **LIME (Local Interpretable Model-agnostic Explanations):**
   - Fits an interpretable sparse linear surrogate $g \in G$ locally in the perturbed neighborhood of a specific claim submission.

3. **TabNet Sequential Attention Maps:**
   - Visualizes the step-level feature selection masks across decision steps, explicitly showing how the neural network gathers information.

4. **Actionable Counterfactual Recommendations:**
   - Instead of merely rejecting a claim, the system identifies the minimal change in billing or documentation required for approval (e.g., revising billed amounts to match standard IRDAI schedule tariffs).
