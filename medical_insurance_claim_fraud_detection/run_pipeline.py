"""Unified pipeline runner for medical insurance fraud detection."""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config
from common.logging_utils import get_logger
from common.seed import set_global_seed

logger = get_logger("run_pipeline")

def main():
    parser = argparse.ArgumentParser(description="Run end-to-end fraud detection pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data_path", type=str, default=None, help="Override raw data path")
    parser.add_argument("--skip_train", action="store_true", help="Skip training, only run hybrid inference")
    parser.add_argument("--claim_id", type=str, default=None, help="Specific claim ID to test")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    set_global_seed(config.get("dataset",{}).get("random_state",42))

    logger.info("Starting pipeline...")

    # 1. Traditional ML (quick for pipeline)
    if not args.skip_train:
        logger.info("Step 1: Traditional ML")
        import subprocess
        cmd = [sys.executable, "approaches/01_traditional_ml.py", "--quick"]
        if args.data_path:
            cmd.extend(["--data_path", args.data_path])
        subprocess.run(cmd, check=False)

        logger.info("Step 2: Deep Learning")
        cmd = [sys.executable, "approaches/02_deep_learning.py"]
        if args.data_path:
            cmd.extend(["--data_path", args.data_path])
        subprocess.run(cmd, check=False)

        logger.info("Step 3: Anomaly Detection")
        cmd = [sys.executable, "approaches/03_anomaly_detection.py"]
        if args.data_path:
            cmd.extend(["--data_path", args.data_path])
        subprocess.run(cmd, check=False)

    logger.info("Step 4: Document Intelligence")
    import subprocess
    subprocess.run([sys.executable, "approaches/04_document_intelligence.py"], check=False)

    logger.info("Step 5: Agentic RAG")
    subprocess.run([sys.executable, "approaches/05_agentic_rag_reasoning.py"], check=False)

    logger.info("Step 6: Hybrid End-to-End")
    hybrid_cmd = [sys.executable, "approaches/06_hybrid_end_to_end.py"]
    if args.claim_id:
        hybrid_cmd.extend(["--claim_id", args.claim_id])
    subprocess.run(hybrid_cmd, check=False)

    logger.info("Step 7: Visualizations")
    subprocess.run([sys.executable, "visualization_generator.py"], check=False)

    logger.info("Step 8: Presentation")
    subprocess.run([sys.executable, "presentation/generate_presentation.py"], check=False)

    logger.info("Step 9: Report")
    subprocess.run([sys.executable, "report/generate_report.py"], check=False)

    logger.info("Pipeline completed. See evaluation/, images/, presentation/, report/")

if __name__=="__main__":
    main()
