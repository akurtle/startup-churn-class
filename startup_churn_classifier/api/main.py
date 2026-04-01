from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from startup_churn_classifier.config import FEATURE_COLUMNS
from startup_churn_classifier.inference import StartupChurnPredictor
from startup_churn_classifier.api.schemas import StartupFeatures


predictor: StartupChurnPredictor | None = None


def load_predictor() -> None:
    global predictor
    predictor = StartupChurnPredictor()


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_predictor()
    yield


app = FastAPI(title="Startup Churn Classifier", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/features")
def features() -> dict[str, list[str]]:
    return {"features": FEATURE_COLUMNS}


@app.post("/predict")
def predict(features: StartupFeatures) -> dict[str, object]:
    if predictor is None:
        raise RuntimeError("Predictor failed to initialize.")

    result = predictor.predict(features.to_inference_payload())
    return {
        "selected_model": result.selected_model,
        "churn_probability": round(result.churn_probability, 4),
        "predicted_label": result.predicted_label,
    }
