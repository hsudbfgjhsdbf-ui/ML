"""Dataset sourcing guard: public source retrieval is deliberately not assumed."""
raise SystemExit('This release uses the deterministic synthetic fallback. Run: python run_pipeline.py --config config/default.yaml')
