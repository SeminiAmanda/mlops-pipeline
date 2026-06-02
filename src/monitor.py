import pandas as pd
import numpy as np
import pickle
import json
import os
import logging
from datetime import datetime
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently import Report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES_PATH = "data/processed/features.pkl"
REPORTS_DIR = "reports"

def load_artifacts():
    with open(FEATURES_PATH, "rb") as f:
        data = pickle.load(f)
    return data["X"], data["y"], data["encoder"], data["vectorizer"]

def get_reference_df(X, y, encoder, n_samples=200):
    np.random.seed(42)
    indices = np.random.choice(len(X), n_samples, replace=False)
    # Use only first 50 features to keep reports readable
    df = pd.DataFrame(X[indices, :50], columns=[f"feature_{i}" for i in range(50)])
    return df

def simulate_production_df(vectorizer, scenario="normal"):
    if scenario == "normal":
        texts = [
            "The government announced new tax policies affecting businesses",
            "Arsenal beat Chelsea in a thrilling premier league match",
            "New smartphone released with improved camera and battery life",
            "Stock markets rallied after positive economic data released",
            "Scientists discover new treatment for common diseases",
            "The prime minister held talks with foreign leaders today",
            "Olympic athletes prepare for upcoming championship games",
            "Tech giant releases major software update for all users",
        ]
    else:  # drift
        texts = [
            "Quantum entanglement experiments reveal subatomic particle behavior",
            "Molecular gastronomy techniques transform restaurant menus globally",
            "Archaeological excavation uncovers ancient civilisation artefacts",
            "Astrophysicists detect gravitational waves from neutron star merger",
            "Biochemical synthesis pathway identified in cellular metabolism study",
            "Geological survey reveals tectonic plate movement in Pacific region",
            "Anthropological research documents indigenous cultural practices",
            "Neurological study maps brain activity during complex decision making",
        ]

    X_prod = vectorizer.transform(texts).toarray()
    df = pd.DataFrame(X_prod[:, :50], columns=[f"feature_{i}" for i in range(50)])
    return df

def run_drift_detection(scenario="normal"):
    logger.info(f"Running drift detection — scenario: {scenario}")

    X, y, encoder, vectorizer = load_artifacts()

    reference_df = get_reference_df(X, y, encoder)
    current_df = simulate_production_df(vectorizer, scenario)

    # Build Evidently datasets
    definition = DataDefinition()
    reference = Dataset.from_pandas(reference_df, data_definition=definition)
    current = Dataset.from_pandas(current_df, data_definition=definition)

    # Run report
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference, current)

    # Save HTML report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{REPORTS_DIR}/drift_report_{scenario}_{timestamp}.html"
    snapshot.save_html(report_path)
    logger.info(f"Saved drift report: {report_path}")

    # Extract result
    result_dict = snapshot.dict()
    metrics = result_dict.get("metrics", [])

    # Find drift metric
    drift_detected = False
    drift_share = 0.0
    for m in metrics:
        if "dataset_drift" in str(m):
            drift_detected = m.get("dataset_drift", False)
            drift_share = m.get("share_of_drifted_columns", 0.0)
            break

    summary = {
        "timestamp": timestamp,
        "scenario": scenario,
        "drift_detected": drift_detected,
        "drift_share": round(drift_share, 4),
        "action": "RETRAIN" if drift_detected else "OK",
        "report": report_path
    }

    summary_path = f"{REPORTS_DIR}/drift_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Drift detected: {drift_detected} | Action: {summary['action']}")
    return summary

if __name__ == "__main__":
    import sys
    scenario = sys.argv[1] if len(sys.argv) > 1 else "normal"
    result = run_drift_detection(scenario)
    print(json.dumps(result, indent=2))