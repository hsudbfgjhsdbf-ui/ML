# Code walkthrough

**Purpose:** map source modules to the data flow.  
**Run:** `run_20260807_151423`.  
**Last updated:** 07-08-2026 15:14:23 UTC.

| File | Responsibility | Key public objects |
| --- | --- | --- |
| `src/data/loading.py` | Excel ingestion, schema gate, profile | `load_claims`, `validate_schema` |
| `src/features/engineering.py` | Date, financial, category features and exclusions | `engineer_features` |
| `src/features/preprocessing.py` | Split, imputation, encoding, scaling | `stratified_three_way_split`, `fit_transform_matrices` |
| `src/models/zoo.py` | Declarative model suite and search spaces | `ModelSpec`, `build_model_specs` |
| `src/evaluation/metrics.py` | Probabilities, metrics, threshold, bootstrap, fairness | `compute_metrics`, `select_threshold` |
| `src/evaluation/plots.py` | EDA, curves, comparison, confusion, calibration | `generate_eda_plots`, `generate_model_curves` |
| `src/evaluation/statistics.py` | McNemar and Wilcoxon helpers | `mcnemar_p_value`, `wilcoxon_p_value` |
| `src/reporting/documents.py` | Markdown, PPTX, project PDF, IEEE PDF | `build_all_documents` |
| `src/pipeline.py` | Ordered orchestration and artifact gates | `run_pipeline` |
| `scripts/run_pipeline.py` | User-facing command-line entry point | `main` |

## Data flow

`xlsx -> load_claims -> engineer_features -> stratified_three_way_split ->
fit_transform_matrices -> model zoo -> validation threshold -> frozen test ->
metrics/curves/plots -> documentation/PPT/PDF`.

The model matrix does not contain raw identifiers. The same transformed column
names are saved in `evaluation/feature_registry.csv`, used for permutation
importance, and referenced by the report. Exceptions are raised at schema,
label, non-finite matrix, and artifact-writing boundaries.
