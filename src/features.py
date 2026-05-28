import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLEAN_PATH = "data/processed/clean.csv"
FEATURES_PATH = "data/processed/features.pkl"

def build_features(path: str = CLEAN_PATH):
    df = pd.read_csv(path)
    logger.info(f"Building features for {len(df)} samples")

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),   # unigrams + bigrams
        stop_words="english"
    )
    X = vectorizer.fit_transform(df["Text"]).toarray()

    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Category"])

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Classes: {list(encoder.classes_)}")

    # Save everything
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump({
            "X": X,
            "y": y,
            "vectorizer": vectorizer,
            "encoder": encoder,
            "classes": list(encoder.classes_)
        }, f)
    logger.info(f"Saved features to {FEATURES_PATH}")

if __name__ == "__main__":
    build_features()