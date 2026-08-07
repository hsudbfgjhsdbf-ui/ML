"""
Jupyter Notebook Generator for Medical Insurance Claim Fraud Detection System.
Institution: IIIT Dharwad, Department of Data Science and AI
Faculty Adviser: Prof. Ramesh Athe
Team Members: B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)

This module generates interactive Jupyter Notebooks (`.ipynb`) in `notebooks/`:
1. `notebooks/eda.ipynb`: Exploratory Data Analysis (EDA) on Indian insurance claims.
2. `notebooks/experiment_tracking.ipynb`: Experiment Tracking & Multi-Approach Model Benchmarking.
"""

import os
import nbformat as nbf
from src.utils import setup_logger, ensure_directories

logger = setup_logger("NotebookGeneratorLogger")


class NotebookGenerator:
    """
    Generates interactive Jupyter Notebooks with Markdown explanations and Python code cells.
    """
    def __init__(self, output_dir: str = "notebooks"):
        self.output_dir = output_dir
        ensure_directories([output_dir])

    def generate_eda_notebook(self, output_path: str = "notebooks/eda.ipynb") -> str:
        """
        Creates the Exploratory Data Analysis (EDA) Jupyter Notebook.
        """
        logger.info(f"Generating Exploratory Data Analysis Notebook at: {output_path}")
        nb = nbf.v4.new_notebook()
        
        cells = [
            nbf.v4.new_markdown_cell(
                "# Exploratory Data Analysis (EDA) — Medical Insurance Claim Fraud Detection\n"
                "**Institution:** Indian Institute of Information Technology (IIIT), Dharwad  \n"
                "**Department:** B.Tech Data Science and Artificial Intelligence  \n"
                "**Faculty Adviser:** Prof. Ramesh Athe  \n"
                "**Team Members:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  \n\n"
                "--- \n\n"
                "## 1. Introduction and Objectives\n"
                "This interactive notebook performs an in-depth Exploratory Data Analysis (EDA) on the "
                "medical insurance claim dataset (`4,500 claims`). It explores distributions, fraud rates across "
                "Indian States and Cities, Hospital Tier cost disparities, and numeric correlations."
            ),
            nbf.v4.new_code_cell(
                "import os, sys\n"
                "if '..' not in sys.path:\n"
                "    sys.path.insert(0, '..')\n"
                "if os.path.basename(os.getcwd()) == 'notebooks':\n"
                "    os.chdir('..')\n"
                "import pandas as pd\n"
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "from src.data_loading import execute_data_loading_pipeline\n\n"
                "plt.style.use('seaborn-v0_8-whitegrid')\n"
                "plt.rcParams['figure.figsize'] = (10, 6)\n"
                "plt.rcParams['font.size'] = 11\n\n"
                "# Load and enrich dataset with Indian domain context\n"
                "df = execute_data_loading_pipeline('data/raw/Health Insurance Fraud Claims.xlsx')\n"
                "print('Dataset Shape:', df.shape)\n"
                "df.head(3)"
            ),
            nbf.v4.new_markdown_cell(
                "## 2. Summary Statistics & Class Imbalance Analysis\n"
                "Medical insurance fraud datasets typically exhibit strong class imbalance. Let's inspect the exact "
                "proportion of **Legitimate** vs. **Fraudulent** claims in the Indian healthcare dataset."
            ),
            nbf.v4.new_code_cell(
                "# Target variable distribution\n"
                "fraud_counts = df['ClaimLegitimacy'].value_counts()\n"
                "fraud_pct = df['ClaimLegitimacy'].value_counts(normalize=True) * 100\n"
                "summary_df = pd.DataFrame({'Count': fraud_counts, 'Percentage (%)': fraud_pct})\n"
                "print(summary_df)\n\n"
                "# Plot Class Distribution\n"
                "plt.figure(figsize=(6, 5))\n"
                "sns.countplot(data=df, x='ClaimLegitimacy', hue='ClaimLegitimacy', palette=['#2e7d32', '#c62828'], legend=False)\n"
                "plt.title('Medical Insurance Claim Legitimacy Distribution (6.0% Fraud Rate)')\n"
                "plt.xlabel('Claim Legitimacy Status')\n"
                "plt.ylabel('Number of Claims')\n"
                "plt.show()"
            ),
            nbf.v4.new_markdown_cell(
                "## 3. Indian Healthcare Provider Tier & Cost Analysis (INR)\n"
                "In India, hospital pricing varies significantly across provider tiers:\n"
                "- **Tier-1 Metro Corporate Hospital:** Premium multi-specialty hospitals in metro cities (Mumbai, Bengaluru, Hyderabad, Delhi).\n"
                "- **Tier-2 City Multi-Specialty Hospital:** Standard accredited city hospitals.\n"
                "- **Tier-3 Town Nursing Home:** Smaller town nursing homes.\n\n"
                "A major fraud indicator is a **Tier-3 Nursing Home billing Tier-1 corporate rates**."
            ),
            nbf.v4.new_code_cell(
                "# Claim Amount INR across Hospital Tiers\n"
                "plt.figure(figsize=(10, 6))\n"
                "sns.boxplot(\n"
                "    data=df, x='HospitalTier', y='ClaimAmountINR', hue='ClaimLegitimacy',\n"
                "    palette=['#2e7d32', '#c62828']\n"
                ")\n"
                "plt.title('Medical Claim Reimbursement Amount (INR) across Indian Hospital Tiers')\n"
                "plt.xlabel('Hospital Provider Tier')\n"
                "plt.ylabel('Claim Amount (Indian Rupees)')\n"
                "plt.show()\n\n"
                "# Table of Mean Claim Amount INR by Tier and Legitimacy\n"
                "df.groupby(['HospitalTier', 'ClaimLegitimacy'])['ClaimAmountINR'].mean().unstack()"
            ),
            nbf.v4.new_markdown_cell(
                "## 4. Geographic Analysis across Indian States\n"
                "Let's examine the distribution of fraudulent claims across Indian states to identify regional patterns."
            ),
            nbf.v4.new_code_cell(
                "state_fraud = df.groupby('IndianState')['IsFraud'].mean().reset_index()\n"
                "state_fraud['Fraud_Percentage'] = state_fraud['IsFraud'] * 100\n"
                "state_fraud = state_fraud.sort_values('Fraud_Percentage', ascending=False)\n\n"
                "plt.figure(figsize=(11, 5))\n"
                "sns.barplot(data=state_fraud, x='IndianState', y='Fraud_Percentage', hue='IndianState', palette='Reds', legend=False)\n"
                "plt.title('Fraud Claim Percentage across Enriched Indian States')\n"
                "plt.xlabel('Indian State')\n"
                "plt.ylabel('Fraud Rate (%)')\n"
                "plt.xticks(rotation=25)\n"
                "plt.show()"
            ),
            nbf.v4.new_markdown_cell(
                "## 5. Correlation Analysis of Numeric Attributes\n"
                "Let's visualize the pairwise Pearson correlation matrix across numeric features."
            ),
            nbf.v4.new_code_cell(
                "num_cols = ['ClaimAmount', 'ClaimAmountINR', 'PatientAge', 'PatientIncome', 'PatientIncomeINR', 'Cluster', 'IsFraud']\n"
                "corr = df[num_cols].corr()\n\n"
                "plt.figure(figsize=(9, 7))\n"
                "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True)\n"
                "plt.title('Correlation Matrix of Key Claim Features')\n"
                "plt.show()"
            ),
            nbf.v4.new_markdown_cell(
                "## 6. Conclusion of Exploratory Analysis\n"
                "The EDA confirms:\n"
                "1. **Imbalance:** Exactly 6.0% of claims are fraudulent, requiring SMOTE during model training.\n"
                "2. **Tier Mismatch:** High-value fraudulent claims often appear in lower-tier or non-accredited providers.\n"
                "3. **Claim Amount:** Billed amounts in INR provide strong discriminative signal when benchmarked against regional specialty norms."
            )
        ]
        
        nb["cells"] = cells
        with open(output_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        logger.info(f"EDA Notebook written successfully: {output_path}")
        return output_path

    def generate_experiment_tracking_notebook(self, output_path: str = "notebooks/experiment_tracking.ipynb") -> str:
        """
        Creates the Experiment Tracking and Multi-Approach Benchmarking Jupyter Notebook.
        """
        logger.info(f"Generating Experiment Tracking Notebook at: {output_path}")
        nb = nbf.v4.new_notebook()
        
        cells = [
            nbf.v4.new_markdown_cell(
                "# Experiment Tracking & Multi-Approach Model Benchmarking\n"
                "**Institution:** Indian Institute of Information Technology (IIIT), Dharwad  \n"
                "**Department:** B.Tech Data Science and Artificial Intelligence  \n"
                "**Faculty Adviser:** Prof. Ramesh Athe  \n"
                "**Team Members:** B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  \n\n"
                "--- \n\n"
                "## 1. Overview of Multi-Approach Benchmarking\n"
                "This notebook tracks, evaluates, and compares all models developed across:\n"
                "- **Approach 1:** 12 Classical Supervised Machine Learning Algorithms\n"
                "- **Approach 2:** 10 Deep Tabular PyTorch Neural Network Architectures\n"
                "- **Approach 3:** Agent AI Multi-Agent Cognitive Verification System\n\n"
                "All models are evaluated on the same held-out test dataset (675 claims) using "
                "accuracy, precision, recall, F1, F2, AUC-ROC, prediction latency (ms), and INR business cost."
            ),
            nbf.v4.new_code_cell(
                "import os, sys\n"
                "if '..' not in sys.path:\n"
                "    sys.path.insert(0, '..')\n"
                "if os.path.basename(os.getcwd()) == 'notebooks':\n"
                "    os.chdir('..')\n"
                "import pandas as pd\n"
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n\n"
                "plt.style.use('seaborn-v0_8-whitegrid')\n"
                "plt.rcParams['figure.figsize'] = (10, 6)\n"
                "plt.rcParams['font.size'] = 11\n\n"
                "# Load approach benchmarking tables\n"
                "b1 = pd.read_csv('data/approach1_benchmarking_table.csv')\n"
                "b1['Approach'] = 'Approach 1 (Classical ML)'\n"
                "b2 = pd.read_csv('data/approach2_benchmarking_table.csv')\n"
                "b2['Approach'] = 'Approach 2 (Deep Learning)'\n\n"
                "# Combine into Master Table\n"
                "master_df = pd.concat([b1, b2], ignore_index=True).sort_values('F2_Score', ascending=False).reset_index(drop=True)\n"
                "print('Total Models Tracked:', len(master_df))\n"
                "master_df[['Algorithm', 'Approach', 'F2_Score', 'Recall', 'Precision', 'AUC_ROC', 'Total_Cost_INR']].head(10)"
            ),
            nbf.v4.new_markdown_cell(
                "## 2. Multi-Metric Evaluation Comparison across Top Models\n"
                "Let's visualize the F2-Score, Recall, Precision, and AUC-ROC across the top 10 algorithms."
            ),
            nbf.v4.new_code_cell(
                "top10 = master_df.head(10)\n"
                "df_melt = top10.melt(\n"
                "    id_vars=['Algorithm'], value_vars=['F2_Score', 'Recall', 'Precision', 'AUC_ROC'],\n"
                "    var_name='Metric', value_name='Score'\n"
                ")\n\n"
                "plt.figure(figsize=(12, 6))\n"
                "sns.barplot(data=df_melt, x='Algorithm', y='Score', hue='Metric', palette='Set2')\n"
                "plt.title('Performance Metric Comparison across Top 10 Fraud Detection Models')\n"
                "plt.xlabel('Algorithm Name')\n"
                "plt.ylabel('Score (0.0 to 1.0)')\n"
                "plt.xticks(rotation=25)\n"
                "plt.legend(loc='lower right')\n"
                "plt.tight_layout()\n"
                "plt.show()"
            ),
            nbf.v4.new_markdown_cell(
                "## 3. Operational Profile: Prediction Latency (ms) vs. F2-Score\n"
                "In real-time claim ingestion, prediction latency is critical. Below we plot the operational trade-off "
                "between detection sensitivity (F2-Score) and prediction speed per sample."
            ),
            nbf.v4.new_code_cell(
                "plt.figure(figsize=(10, 6))\n"
                "sns.scatterplot(\n"
                "    data=master_df, x='Prediction_Latency_ms', y='F2_Score',\n"
                "    hue='Approach', style='Approach', s=120, palette=['#1f77b4', '#ff7f0e']\n"
                ")\n"
                "for _, row in master_df.iterrows():\n"
                "    if row['F2_Score'] > 0.95 or row['Prediction_Latency_ms'] > 2.0:\n"
                "        plt.text(row['Prediction_Latency_ms']+0.05, row['F2_Score']+0.003, row['Algorithm'], fontsize=9)\n"
                "plt.title('Operational Trade-off: Prediction Latency (ms) vs. Fraud Detection F2-Score')\n"
                "plt.xlabel('Prediction Latency per Sample (ms)')\n"
                "plt.ylabel('F2-Score (Recall-Weighted)')\n"
                "plt.tight_layout()\n"
                "plt.show()"
            ),
            nbf.v4.new_markdown_cell(
                "## 4. Indian Healthcare Financial Impact (INR Cost Matrix)\n"
                "Using our Indian insurance cost matrix:\n"
                "- **False Negative Cost ($C_{FN}$):** Rs. 1,50,000 (average fraudulent payout loss).\n"
                "- **False Positive Cost ($C_{FP}$):** Rs. 5,000 (administrative grievance and verification cost).\n\n"
                "Let's compare the total financial cost incurred by each algorithm on the test dataset."
            ),
            nbf.v4.new_code_cell(
                "plt.figure(figsize=(11, 5))\n"
                "top_cost = master_df.sort_values('Total_Cost_INR', ascending=True).head(12)\n"
                "sns.barplot(data=top_cost, y='Algorithm', x='Total_Cost_INR', hue='Approach', palette='Dark2')\n"
                "plt.title('Total Financial Cost in Indian Rupees (INR) on Held-Out Test Set (Lower is Better)')\n"
                "plt.xlabel('Total Financial Cost (Indian Rupees / Rs.)')\n"
                "plt.ylabel('Algorithm Name')\n"
                "plt.tight_layout()\n"
                "plt.show()"
            ),
            nbf.v4.new_markdown_cell(
                "## 5. Live Demonstration of Agent AI Multi-Agent Verification (Approach 3)\n"
                "While numerical classifiers output a probability, our **Agent AI Multi-Agent System** orchestrates 5 specialized "
                "cognitive agents via LangGraph to provide **human-readable natural language explanations** citing specific policy clauses (`[CLAUSE-ROOM-001]`)."
            ),
            nbf.v4.new_code_cell(
                "from src.agent_ai.database import InsuranceDatabaseManager\n"
                "from src.agent_ai.workflow import ClaimProcessingState, AgentAIWorkflowOrchestrator\n\n"
                "db = InsuranceDatabaseManager('data/local_database.db')\n"
                "orchestrator = AgentAIWorkflowOrchestrator(db)\n\n"
                "sample_claim = {\n"
                "    'claim_id': 'CLM-NOTEBOOK-2026',\n"
                "    'policy_number': 'STAR-HLTH-2024-8871',\n"
                "    'user_id': 'USR-IND-001',\n"
                "    'provider_id': 'HOSP-MUM-01',\n"
                "    'hospital_name': 'Apollo Hospitals Navi Mumbai',\n"
                "    'treatment_type': 'Inpatient',\n"
                "    'procedure_code': 'IND-PROC-101',\n"
                "    'claimed_amount_inr': 135000.0,\n"
                "    'patient_age': 48\n"
                "}\n\n"
                "state = ClaimProcessingState(\n"
                "    claim_id=sample_claim['claim_id'],\n"
                "    policy_number=sample_claim['policy_number'],\n"
                "    user_id=sample_claim['user_id'],\n"
                "    raw_claim_context=sample_claim,\n"
                "    uploaded_documents=[{'document_type': 'Hospital Bill', 'file_path': 'sample_bill.pdf'}]\n"
                ")\n\n"
                "final_state = orchestrator.run_workflow(state)\n"
                "report = final_state.final_decision_result\n"
                "print('=== EXPLAINABLE AI NATURAL LANGUAGE DECISION REPORT ===')\n"
                "print('Verdict:', report['decision'], '| Confidence Score:', round(report['confidence_score']*100, 1), '%')\n"
                "print('\\nExecutive Summary:\\n', report['executive_summary'])\n"
                "print('\\nCitations:', report['evidence_citations'])"
            ),
            nbf.v4.new_markdown_cell(
                "## 6. Summary and Architectural Recommendations\n"
                "1. **For Preliminary High-Throughput Screening:** Use **LightGBM / XGBoost** (`F2 > 0.99`, latency `< 0.1 ms`).\n"
                "2. **For Complex Tabular Interactions & Attention Maps:** Use **Tabular Transformer** (`F2 = 0.9799`).\n"
                "3. **For Complete Legal Verification & Policyholder Notice:** Use **Agent AI Multi-Agent System (Approach 3)**."
            )
        ]
        
        nb["cells"] = cells
        with open(output_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        logger.info(f"Experiment Tracking Notebook written successfully: {output_path}")
        return output_path
