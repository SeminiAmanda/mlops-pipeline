import subprocess
import sys
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_step(name, script):
    logger.info(f"Running: {name}")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        logger.error(f"FAILED: {name}")
        logger.error(result.stderr)
        sys.exit(1)
    logger.info(f"DONE: {name}")
    return result.stdout

def main(force_retrain=False):
    logger.info("=== MLOps Pipeline Starting ===")

    # Step 1: Check drift
    logger.info("Step 1: Drift detection")
    result = subprocess.run(
        [sys.executable, "src/monitor.py", "normal"],
        capture_output=True, text=True
    )
    
    # Parse last JSON block from output
    output_lines = result.stdout.strip().split('\n')
    json_output = '\n'.join(output_lines)
    
    try:
        summary = json.loads(json_output)
    except json.JSONDecodeError:
        # Find the JSON part
        for i, line in enumerate(output_lines):
            if line.strip() == '{':
                json_str = '\n'.join(output_lines[i:])
                summary = json.loads(json_str)
                break

    logger.info(f"Drift result: {summary}")

    # Step 2: Retrain if needed
    if summary.get("drift_detected") or force_retrain:
        logger.info("Step 2: Retraining triggered")
        run_step("Data ingestion", "src/ingest.py")
        run_step("Feature engineering", "src/features.py")
        run_step("Model training", "src/train.py")
        logger.info("=== Retraining Complete ===")
    else:
        logger.info("Step 2: No retraining needed — model is healthy")

    logger.info("=== Pipeline Complete ===")
    return summary

if __name__ == "__main__":
    force = "--force" in sys.argv
    main(force_retrain=force)