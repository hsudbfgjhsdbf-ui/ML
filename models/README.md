# Serialized model inventory

Runtime model binaries are written to `artifacts/models/` and are included in
the release because the approach specification requires saved model state. This
directory documents the contract expected by later inference code.

- `artifacts/models/preprocessor.joblib` — train-fitted feature transformer.
- `artifacts/models/<model_key>.joblib` — fitted comparative estimator.
- `artifacts/models/best_model.joblib` — winner bundle containing transformer,
  model, feature names, threshold, target semantics, and run identifier.
- `artifacts/models/calibrator.joblib` — validation-fitted probability calibrator.

Load the winner only with a trusted, compatible environment. Never load an
untrusted pickle/joblib file. The companion metric JSON and model card live
under `evaluation/metrics/` and `evaluation/model_cards/`.
