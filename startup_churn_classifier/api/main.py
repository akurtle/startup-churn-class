from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from startup_churn_classifier.config import FEATURE_COLUMNS
from startup_churn_classifier.inference import StartupChurnPredictor


class StartupFeatures(BaseModel):
    company_age_months: Any = Field(default=None)
    monthly_burn_usd: Any = Field(default=None)
    runway_months: Any = Field(default=None)
    team_size: Any = Field(default=None)
    founder_exits: Any = Field(default=None)
    customer_growth_pct: Any = Field(default=None)
    support_tickets_last_30_days: Any = Field(default=None)
    annual_revenue_usd: Any = Field(default=None)
    market_segment: Any = Field(default=None)
    growth_stage: Any = Field(default=None)
    remote_friendly: Any = Field(default=None)
    investor_tier: Any = Field(default=None)


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

    result = predictor.predict(features.model_dump())
    return {
        "selected_model": result.selected_model,
        "churn_probability": round(result.churn_probability, 4),
        "predicted_label": result.predicted_label,
    }
