import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_PATH = "data/raw/bbc"
PROCESSED_PATH = "data/processed/clean.csv"

def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    logger.info(f"Loading data from folder: {path}")
    records = []

    for category in os.listdir(path):
        category_path = os.path.join(path, category)
        if not os.path.isdir(category_path):
            continue  

        for filename in os.listdir(category_path):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(category_path, filename)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            records.append({"Text": text, "Category": category})

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} articles across {df['Category'].nunique()} categories")
    return df

def validate(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Running validation checks...")

    before = len(df)
    df = df.dropna(subset=["Text", "Category"])
    logger.info(f"Dropped {before - len(df)} null rows")

    before = len(df)
    df = df.drop_duplicates(subset=["Text"])
    logger.info(f"Dropped {before - len(df)} duplicate rows")

    logger.info(f"Class distribution:\n{df['Category'].value_counts()}")
    return df

def save(df: pd.DataFrame, path: str = PROCESSED_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved cleaned data to {path}")

if __name__ == "__main__":
    df = load_data()
    df = validate(df)
    save(df)