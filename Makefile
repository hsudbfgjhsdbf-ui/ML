.PHONY: install self-test dry-run run verify clean

install:
	python -m pip install -r requirements.txt

self-test:
	python scripts/run_pipeline.py --self-test

dry-run:
	python scripts/run_pipeline.py --config config/default.yaml --dry-run

run:
	python scripts/run_pipeline.py --config config/default.yaml

verify:
	python scripts/verify_artifacts.py

clean:
	find workspace -mindepth 1 -maxdepth 1 -exec rm -rf {} +
