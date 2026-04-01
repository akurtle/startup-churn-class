from fastapi.testclient import TestClient

from startup_churn_classifier.api.main import app, load_predictor
from startup_churn_classifier.training import run_training_pipeline


def test_predict_endpoint() -> None:
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
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert 0 <= payload["churn_probability"] <= 1
    assert payload["predicted_label"] in {0, 1}
