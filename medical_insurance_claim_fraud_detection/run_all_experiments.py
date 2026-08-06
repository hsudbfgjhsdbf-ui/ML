"""Run all experiments benchmarking all compatible approaches."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

def run(cmd, desc):
    print(f"\n=== {desc}: {' '.join(cmd)} ===")
    ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
    print(f"Finished {desc} with exit {ret.returncode}")
    return ret.returncode

def main():
    data_path = "/home/user/ML/Health Insurance Fraud Claims.xlsx"
    # Check if exists else use config default
    if not Path(data_path).exists():
        data_path = None

    # Traditional ML full (no --quick) — may be heavy but try
    cmd = [sys.executable, "approaches/01_traditional_ml.py"]
    if data_path:
        cmd.extend(["--data_path", data_path])
    run(cmd, "Traditional ML Full")

    cmd = [sys.executable, "approaches/02_deep_learning.py"]
    if data_path:
        cmd.extend(["--data_path", data_path])
    run(cmd, "Deep Learning")

    cmd = [sys.executable, "approaches/03_anomaly_detection.py"]
    if data_path:
        cmd.extend(["--data_path", data_path])
    run(cmd, "Anomaly Detection")

    run([sys.executable, "approaches/04_document_intelligence.py"], "Document Intelligence")
    run([sys.executable, "approaches/05_agentic_rag_reasoning.py"], "Agentic RAG")
    run([sys.executable, "approaches/06_hybrid_end_to_end.py"], "Hybrid")

    run([sys.executable, "visualization_generator.py"], "Visualizations")
    run([sys.executable, "presentation/generate_presentation.py"], "Presentation")
    run([sys.executable, "report/generate_report.py"], "Report")
    run([sys.executable, "tests/test_basic.py"], "Tests")

    print("\nAll experiments completed. Check evaluation/, images/, presentation/, report/")

if __name__=="__main__":
    main()
