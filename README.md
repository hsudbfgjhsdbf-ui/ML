# Medical Insurance Claim Fraud Detection Project
**Indian Institute of Information Technology (IIIT Dharwad)**  
**Department of Data Science and Artificial Intelligence**  
**Faculty Adviser**: Prof. Ramesh Athe  
**Team Members**: B Varshith, M Jagadeshwar, J Ganesh  

---

## Project Overview
This project delivers a complete, end-to-end multi-approach framework for detecting medical insurance claim fraud within the Indian healthcare ecosystem:
1. **Approach 1**: Traditional Machine Learning (12 classification algorithms with rigorous feature engineering and evaluation).
2. **Approach 2**: Deep Learning (10 advanced neural network architectures capturing hierarchical feature interactions and attention).
3. **Approach 3**: Agent AI Multi-Agent System (Cognitive AI agents with Gemini API, LangChain, LangGraph, RAG, OCR document processing, and Explainable AI).

---

## Directory Structure
```
.
├── Health Insurance Fraud Claims.xlsx  # Raw dataset
├── README.md                           # Project documentation & setup
├── config/                             # Configuration files
├── data/                               # Raw, processed, and synthetic data
├── src/                                # Source code modules
│   ├── data_preprocessing.py           # Preprocessing & feature engineering
│   ├── train_ml.py                     # Approach 1: Traditional ML training
│   ├── train_dl.py                     # Approach 2: Deep Learning training
│   ├── app.py                          # Approach 3: FastAPI backend & Web UI
│   ├── agents/                         # Multi-agent workflows
│   └── database/                       # SQLAlchemy models
├── evaluation/                         # Benchmarking CSVs and Markdown reports
├── models/                             # Serialized ML and DL models
├── visualizations/                     # Generated charts and plots
├── docs/                               # Comprehensive project documentation
├── presentation/                       # 20-slide presentation outline
└── reports/                            # IEEE research paper PDF/markdown reports
```

---

## Setup & Installation

1. **Clone and Setup Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt  # (or install dependencies via pip)
   ```

2. **Run Data Preprocessing**:
   ```bash
   python3 src/data_preprocessing.py
   ```

3. **Train Traditional ML Models (Approach 1)**:
   ```bash
   python3 src/train_ml.py
   ```

4. **Train Deep Learning Architectures (Approach 2)**:
   ```bash
   python3 src/train_dl.py
   ```

5. **Start Agent AI Web Application / Live Preview (Approach 3)**:
   ```bash
   python3 src/app.py
   ```
   Access the interactive web application at `http://0.0.0.0:8000`.

---

## Deliverables & Evaluation
- **Traditional ML Benchmarks**: `evaluation/traditional_ml_benchmark.md`
- **Deep Learning Benchmarks**: `evaluation/deep_learning_benchmark.md`
- **IEEE Research Reports**: `reports/ieee_report_ml.md`, `reports/ieee_report_dl.md`
- **Project Documentation**: `docs/project_documentation.md`
- **Presentation Outline**: `presentation/presentation_outline.md`
