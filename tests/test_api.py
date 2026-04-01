import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from startup_churn_classifier.api.logging import LOGGER_NAME
from startup_churn_classifier.api.main import app, load_predictor
from startup_churn_classifier.api.metrics import api_metrics
from startup_churn_classifier.training import run_training_pipeline


@pytest.fixture(autouse=True)
def reset_api_metrics() -> None:
    api_metrics.reset()


def test_predict_endpoint(caplog) -> None:
    summary = run_training_pipeline()
    load_predictor()
    client = TestClient(app)
    caplog.set_level("INFO", logger=LOGGER_NAME)

    response = client.post(
        "/predict",
        json={
            "company_age_months": "24",
            "monthly_burn_usd": "$120,000",
            "runway_months": "10",
            "team_size": 18,
            "founder_exits": 0,
            "customer_growth_pct": "12%",
            "support_tickets_last_30_days": 14,
            "annual_revenue_usd": "$1,900,000",
            "market_segment": "SaaS",
            "growth_stage": "Series-A",
            "remote_friendly": "yes",
            "investor_tier": "tier-2-vc",
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    payload = response.json()
    assert 0 <= payload["churn_probability"] <= 1
    assert payload["predicted_label"] in {0, 1}
    run_path = Path(summary["experiment_tracking"]["run_path"])
    record = json.loads(run_path.read_text(encoding="utf-8"))
    assert record["selected_model"] == summary["selected_model"]
    assert record["artifact_version"] == summary["experiment_tracking"]["artifact_version"]
    completed_logs = [
        json.loads(entry.message)
        for entry in caplog.records
        if entry.name == LOGGER_NAME and "request_completed" in entry.message
    ]
    assert completed_logs
    assert completed_logs[-1]["request_id"] == response.headers["X-Request-ID"]
    assert completed_logs[-1]["status_code"] == 200


def test_predict_rejects_invalid_numeric_payload() -> None:
    run_training_pipeline()
    load_predictor()
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={
            "company_age_months": "24",
            "monthly_burn_usd": "abc",
            "runway_months": "10",
            "team_size": 18,
            "founder_exits": 0,
            "customer_growth_pct": "12%",
            "support_tickets_last_30_days": 14,
            "annual_revenue_usd": "$1,900,000",
            "market_segment": "SaaS",
            "growth_stage": "Series-A",
            "remote_friendly": "yes",
            "investor_tier": "tier-2-vc",
        },
    )

    assert response.status_code == 422
    assert response.headers["X-Request-ID"]
    payload = response.json()
    assert payload["detail"][0]["loc"][-1] == "monthly_burn_usd"
    assert "valid number" in payload["detail"][0]["msg"]


def test_predict_rejects_unknown_fields() -> None:
    run_training_pipeline()
    load_predictor()
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={
            "company_age_months": "24",
            "monthly_burn_usd": "$120,000",
            "runway_months": "10",
            "team_size": 18,
            "founder_exits": 0,
            "customer_growth_pct": "12%",
            "support_tickets_last_30_days": 14,
            "annual_revenue_usd": "$1,900,000",
            "market_segment": "SaaS",
            "growth_stage": "Series-A",
            "remote_friendly": "yes",
            "investor_tier": "tier-2-vc",
            "extra_field": "should fail",
        },
    )

    assert response.status_code == 422
    assert response.headers["X-Request-ID"]
    payload = response.json()
    assert payload["detail"][0]["loc"][-1] == "extra_field"


def test_metrics_reports_prediction_volume_and_error_rate() -> None:
    run_training_pipeline()
    load_predictor()
    client = TestClient(app)

    valid_payload = {
        "company_age_months": "24",
        "monthly_burn_usd": "$120,000",
        "runway_months": "10",
        "team_size": 18,
        "founder_exits": 0,
        "customer_growth_pct": "12%",
        "support_tickets_last_30_days": 14,
        "annual_revenue_usd": "$1,900,000",
        "market_segment": "SaaS",
        "growth_stage": "Series-A",
        "remote_friendly": "yes",
        "investor_tier": "tier-2-vc",
    }
    invalid_payload = dict(valid_payload)
    invalid_payload["monthly_burn_usd"] = "bad-number"

    success_response = client.post("/predict", json=valid_payload)
    error_response = client.post("/predict", json=invalid_payload)
    metrics_response = client.get("/metrics")

    assert success_response.status_code == 200
    assert error_response.status_code == 422
    assert metrics_response.status_code == 200

    payload = metrics_response.json()
    assert payload["requests_total"] == 2
    assert payload["request_errors_total"] == 1
    assert payload["request_error_rate"] == 0.5
    assert payload["predictions_total"] == 2
    assert payload["prediction_errors_total"] == 1
    assert payload["prediction_error_rate"] == 0.5
    assert payload["status_counts"]["200"] == 1
    assert payload["status_counts"]["422"] == 1
    assert payload["path_counts"]["/predict"] == 2
    assert payload["last_updated_utc"]
