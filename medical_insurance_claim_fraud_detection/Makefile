.PHONY: all install test traditional deep anomaly doc rag hybrid viz presentation report clean

all: traditional deep anomaly doc rag hybrid viz presentation report test

install:
	pip install -r requirements.txt

install-optional:
	pip install -r requirements_optional.txt

traditional:
	python approaches/01_traditional_ml.py --quick --data_path /home/user/ML/Health\ Insurance\ Fraud\ Claims.xlsx

traditional-full:
	python approaches/01_traditional_ml.py --data_path /home/user/ML/Health\ Insurance\ Fraud\ Claims.xlsx

deep:
	python approaches/02_deep_learning.py --data_path /home/user/ML/Health\ Insurance\ Fraud\ Claims.xlsx

anomaly:
	python approaches/03_anomaly_detection.py --data_path /home/user/ML/Health\ Insurance\ Fraud\ Claims.xlsx

doc:
	python approaches/04_document_intelligence.py

rag:
	python approaches/05_agentic_rag_reasoning.py

hybrid:
	python approaches/06_hybrid_end_to_end.py

viz:
	python visualization_generator.py

presentation:
	python presentation/generate_presentation.py

report:
	python report/generate_report.py

test:
	python tests/test_basic.py

pipeline:
	python run_pipeline.py

all-experiments:
	python run_all_experiments.py

clean:
	rm -rf data/processed/artifacts/*.joblib
	rm -rf data/processed/artifacts/*.pt
	rm -rf evaluation/*.csv evaluation/*.json evaluation/confusion_matrices/*.csv
	rm -rf images/*.png visualizations/*.png
	rm -rf presentation/*.pptx report/*.pdf
	@echo "Cleaned. Keep sample and raw data."

help:
	@echo "Targets: all install test traditional deep anomaly doc rag hybrid viz presentation report pipeline all-experiments clean"
