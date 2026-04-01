from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from startup_churn_classifier import __version__
from startup_churn_classifier.config import ARTIFACTS_DIR
from startup_churn_classifier.config import FEATURE_COLUMNS
from startup_churn_classifier.api.logging import configure_structured_logging, log_event
from startup_churn_classifier.api.metrics import api_metrics
from startup_churn_classifier.inference import StartupChurnPredictor
from startup_churn_classifier.api.schemas import StartupFeatures


predictor: StartupChurnPredictor | None = None
STATIC_DIR = Path(__file__).resolve().parent / "static"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def load_predictor() -> None:
    global predictor
    predictor = StartupChurnPredictor()


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_structured_logging()
    load_predictor()
    yield


app = FastAPI(title="Startup Churn Classifier", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/features")
def features() -> dict[str, list[str]]:
    return {"features": FEATURE_COLUMNS}


@app.get("/dashboard/summary")
def dashboard_summary() -> dict[str, object]:
    if predictor is None:
        raise RuntimeError("Predictor failed to initialize.")

    artifact_metrics: dict[str, object] = {}
    if METRICS_PATH.exists():
        artifact_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    return {
        "app": {
            "name": app.title,
            "version": __version__,
        },
        "selected_model": predictor.selected_model,
        "model_family": predictor.model_family,
        "threshold": predictor.threshold,
        "artifact_metrics": artifact_metrics,
        "model_metadata": predictor.metadata,
        "features": FEATURE_COLUMNS,
    }


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
