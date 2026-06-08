# BBC News MLOps Pipeline

An end-to-end MLOps system that trains a news article classifier, serves it as a REST API, monitors it for data drift in production, and automatically retrains itself when degradation is detected — with zero manual intervention.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green) ![MLflow](https://img.shields.io/badge/MLflow-tracking-orange) ![Docker](https://img.shields.io/badge/Docker-containerised-blue) ![CI/CD](https://img.shields.io/badge/GitHub_Actions-automated-brightgreen)

---

## What This Project Does

Most ML portfolios stop at "trained a model and got 92% accuracy." This project goes further — it builds the infrastructure that keeps a model accurate over time in production. That is what ML engineering teams actually spend their time on.

The system:
- Ingests and validates raw BBC News articles across 5 categories
- Trains a text classifier with full experiment tracking
- Serves predictions via a REST API with confidence scores
- Monitors incoming traffic for data drift using statistical tests
- Automatically retrains the model via GitHub Actions when drift is detected

---

## Architecture

```
Data ingestion → Feature engineering → Training + MLflow
                                              ↓
                                      Model registry
                                              ↓
                              FastAPI serving (Docker)
                                              ↓
                              Evidently drift monitoring
                                              ↓
                        GitHub Actions retraining trigger
                                              ↓
                              (loops back to training)
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| ML Framework | scikit-learn, PyTorch |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| API Serving | FastAPI + Uvicorn |
| Containerisation | Docker |
| Drift Detection | Evidently AI |
| CI/CD | GitHub Actions |
| Dataset | BBC News (2,225 articles, 5 categories) |

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 98.12% |
| Macro F1 | 0.98 |
| Inference time | ~1.5ms |
| Categories | business, entertainment, politics, sport, tech |

---

## Project Structure

```
mlops-pipeline/
├── src/
│   ├── ingest.py          # Data loading, validation, cleaning
│   ├── features.py        # TF-IDF vectorisation, label encoding
│   ├── train.py           # Model training + MLflow logging
│   ├── predict.py         # Inference module
│   ├── api.py             # FastAPI application
│   ├── monitor.py         # Evidently drift detection
│   └── pipeline.py        # End-to-end pipeline orchestrator
├── data/
│   ├── raw/               # Raw BBC News articles (DVC tracked)
│   └── processed/         # Features, cleaned CSV, model pickle
├── reports/               # Evidently HTML drift reports
├── .github/workflows/
│   └── retrain.yml        # Automated retraining workflow
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## How to Run Locally

**1 — Clone and set up environment:**

```bash
git clone https://github.com/yourusername/mlops-pipeline.git
cd mlops-pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2 — Download data:**

```bash
kaggle datasets download -d bbose71/bbc-full-text-document-classification \
  -p data/raw/ --unzip
```

**3 — Run the full pipeline:**

```bash
python src/ingest.py       # Load and validate data
python src/features.py     # Build TF-IDF features
python src/train.py        # Train model + log to MLflow
```

**4 — View experiment tracking:**

```bash
mlflow ui
# Open http://localhost:5000
```

**5 — Start the API:**

```bash
uvicorn src.api:app --reload --port 8000
# Open http://localhost:8000/docs
```

**6 — Run drift detection:**

```bash
python src/monitor.py normal   # No drift — action: OK
python src/monitor.py drift    # Drift detected — action: RETRAIN
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health root |
| `/health` | GET | Model health check |
| `/predict` | POST | Classify a news article |
| `/docs` | GET | Interactive Swagger UI |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Arsenal beat Chelsea in a thrilling premier league match"}'
```

**Example response:**

```json
{
  "category": "sport",
  "confidence": 0.97,
  "all_probabilities": {
    "business": 0.01,
    "entertainment": 0.01,
    "politics": 0.01,
    "sport": 0.97,
    "tech": 0.01
  },
  "inference_time_ms": 1.41
}
```

---

## Docker

```bash
# Build
docker build -t bbc-classifier:v2 .

# Run
docker run -p 8000:8000 bbc-classifier:v2

# Test
curl http://localhost:8000/health
```

---

## Automated Retraining (GitHub Actions)

The `.github/workflows/retrain.yml` workflow:

1. Triggers on schedule (daily), push to `data/` or `src/`, or manual dispatch
2. Runs drift detection on incoming production data
3. If drift is detected → reruns the full pipeline automatically
4. Saves the retrained model and drift report as workflow artifacts
5. If no drift → logs "model is healthy" and skips retraining

To trigger manually: **Actions → Automated Retraining Pipeline → Run workflow → select scenario**

---

## Drift Detection

Evidently AI computes statistical tests (chi-squared, correlation) between the training data distribution and incoming production requests. When the share of drifted features exceeds the threshold, the system flags a `RETRAIN` action.

```bash
# Simulate normal traffic (no drift)
python src/monitor.py normal
# → {"drift_detected": false, "action": "OK"}

# Simulate drifted traffic
python src/monitor.py drift
# → {"drift_detected": true, "action": "RETRAIN"}
```

HTML drift reports are saved to `reports/` with full feature-level breakdown.

---

## Bullet Points

```
• Built end-to-end MLOps pipeline for BBC News classification —
  PyTorch, scikit-learn, MLflow experiment tracking, DVC data versioning

• Deployed model as REST API (FastAPI + Docker) with confidence scoring
  and 1.5ms inference time, achieving 98.12% classification accuracy

• Implemented production drift monitoring with Evidently AI and
  automated retraining via GitHub Actions CI/CD — zero manual intervention
```

---

## What I Learned

- How MLflow tracks experiments, parameters, metrics and model artifacts across runs
- Why data versioning with DVC matters — reproducibility is not optional in production
- How Docker isolates the serving environment from the training environment
- How statistical drift detection works and why models degrade over time
- How to build a CI/CD loop that keeps an ML model healthy without human intervention

---

## Author

**Amanda Sewwandi**
University of Sri Jayewardenepura
[text](https://github.com/SeminiAmanda/mlops-pipeline)