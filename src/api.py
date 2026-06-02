from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import predict
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BBC News Classifier API",
    description="MLOps pipeline — classifies news into 5 categories",
    version="1.0.0"
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    category: str
    confidence: float
    all_probabilities: dict
    inference_time_ms: float

@app.get("/health")
def health():
    return {"status": "healthy", "model": "bbc-news-classifier-v1"}

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    start = time.time()
    result = predict(request.text)
    elapsed = round((time.time() - start) * 1000, 2)

    logger.info(f"Predicted: {result['category']} ({result['confidence']}) in {elapsed}ms")

    return {**result, "inference_time_ms": elapsed}

@app.get("/")
def root():
    return {"message": "BBC News Classifier API is running. Go to /docs for Swagger UI."}