# Source manuscript — Approach 1 project report

This Markdown source documents the content generated into
`approach_1_project_report.pdf`. The PDF builder is
`src/reporting/documents.py`; it reads the current evaluation artifacts rather
than this file for metric values.

## Title page

**Medical Insurance Claim Fraud Detection — Approach 1: Traditional Machine Learning**  
B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)  
IIIT Dharwad, Department of Data Science and AI  
Faculty Adviser: Prof. Ramesh Athe

## Abstract

This report presents a reproducible traditional machine-learning baseline for
medical insurance claim fraud screening. The repository-provided workbook is
validated, profiled, transformed without target leakage, and evaluated with a
20-model classical model zoo. Validation F2 selects the operating threshold and
the model winner; the locked test set is evaluated only after selection. The
artifact chain contains figures, curve data, calibration, fairness slices,
permutation importance, serialized state, a 20-slide deck, and a compact IEEE-
inspired manuscript. The result is a decision-support baseline, not an
automatic claim-denial system.

## Chapter map

1. Introduction and problem statement
2. Indian context and responsible use
3. Dataset and data quality
4. Exploratory data analysis
5. Preprocessing and feature engineering
6. Algorithms and tuning
7. Evaluation and thresholding
8. Explainability and fairness
9. Discussion
10. Conclusion and future work
11. Figures and artifact references
12. References and acknowledgments
13. Appendices: model inventory, search budgets, metrics, threshold, fairness,
   feature dictionary, data relations, reproducibility, model cards,
   error-analysis worksheet, Indian-context gap analysis, deployment boundary,
   future approaches, examiner questions, and artifact navigation

## Evidence sources

- `evaluation/leaderboard.csv`
- `evaluation/metrics/`
- `evaluation/curves/`
- `evaluation/calibration/`
- `evaluation/fairness/`
- `evaluation/explainability/`
- `documentation/complete_technical_manual.md`
- `data/dataset_card.md`

## Number integrity rule

The PDF is regenerated from the pipeline context. If data or code changes,
regenerate the PDF and do not hand edit a metric in this source document.
