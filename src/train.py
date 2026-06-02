import pickle
import os
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES_PATH = "data/processed/features.pkl"

def train():
    # Load features
    with open(FEATURES_PATH, "rb") as f:
        data = pickle.load(f)

    X, y = data["X"], data["y"]
    classes = data["classes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # MLflow experiment
    mlflow.set_experiment("bbc-news-classification")

    with mlflow.start_run():
        C = 1.0
        max_iter = 1000

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("num_classes", len(classes))

        logger.info("Training model...")
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=classes)

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"\n{report}")

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")

        # Save model as pickle for Docker serving
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/model.pkl", "wb") as f:
            pickle.dump(model, f)
        logger.info("Saved model to data/processed/model.pkl")

    return acc

if __name__ == "__main__":
    train()