from fastapi.testclient import TestClient

from startup_churn_classifier.api.main import app, load_predictor
from startup_churn_classifier.training import run_training_pipeline


def test_dashboard_root_serves_html() -> None:
    run_training_pipeline()
    load_predictor()
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Startup Churn Control Room" in response.text


def test_dashboard_summary_exposes_model_snapshot() -> None:
    run_training_pipeline()
    load_predictor()
    client = TestClient(app)

    response = client.get("/dashboard/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_model"]
    assert payload["model_metadata"]["metrics"]["roc_auc"] > 0
    assert "results" in payload["artifact_metrics"]
