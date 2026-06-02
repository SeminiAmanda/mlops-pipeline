import pickle
import numpy as np
import logging

logger = logging.getLogger(__name__)

FEATURES_PATH = "data/processed/features.pkl"
MODEL_PATH = "data/processed/model.pkl"

with open(FEATURES_PATH, "rb") as f:
    data = pickle.load(f)
vectorizer = data["vectorizer"]
encoder = data["encoder"]

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

logger.info("Model and vectorizer loaded successfully")

def predict(text: str) -> dict:
    X = vectorizer.transform([text]).toarray()
    pred_idx = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    category = encoder.inverse_transform([pred_idx])[0]
    confidence = float(np.max(pred_proba))
    class_probs = {
        cls: round(float(prob), 4)
        for cls, prob in zip(encoder.classes_, pred_proba)
    }
    return {
        "category": category,
        "confidence": round(confidence, 4),
        "all_probabilities": class_probs
    }