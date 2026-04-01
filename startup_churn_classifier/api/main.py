from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import Response

from startup_churn_classifier.config import FEATURE_COLUMNS
from startup_churn_classifier.api.logging import configure_structured_logging, log_event
from startup_churn_classifier.api.metrics import api_metrics
from startup_churn_classifier.inference import StartupChurnPredictor
from startup_churn_classifier.api.schemas import StartupFeatures


predictor: StartupChurnPredictor | None = None


def load_predictor() -> None:
    global predictor
    predictor = StartupChurnPredictor()


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_structured_logging()
    load_predictor()
    yield


app = FastAPI(title="Startup Churn Classifier", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def add_request_context(request: Request, call_next) -> Response:
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    start = perf_counter()

    log_event(
        "request_started",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client_host=request.client.host if request.client else None,
    )

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((perf_counter() - start) * 1000, 2)
        api_metrics.record_request(path=request.url.path, status_code=500, duration_ms=duration_ms)
        log_event(
            "request_failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
        )
        raise

    duration_ms = round((perf_counter() - start) * 1000, 2)
    api_metrics.record_request(
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    log_event(
        "request_completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/features")
def features() -> dict[str, list[str]]:
    return {"features": FEATURE_COLUMNS}


@app.get("/metrics")
def metrics() -> dict[str, object]:
    return api_metrics.snapshot()


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
