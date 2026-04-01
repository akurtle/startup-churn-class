import math

import pandas as pd

from startup_churn_classifier.preprocessing import clean_startup_frame


def test_clean_startup_frame_adds_engineered_features() -> None:
    frame = pd.DataFrame(
        [
            {
                "company_age_months": "24",
                "monthly_burn_usd": "$100,000",
                "runway_months": "5",
                "team_size": 20,
                "founder_exits": 1,
                "customer_growth_pct": "12%",
                "support_tickets_last_30_days": 14,
                "annual_revenue_usd": "$2,400,000",
                "market_segment": "SaaS",
                "growth_stage": "Series-A",
                "remote_friendly": "yes",
                "investor_tier": "tier-2-vc",
            }
        ]
    )

    cleaned = clean_startup_frame(frame)

    assert math.isclose(cleaned.loc[0, "burn_to_revenue_ratio"], 0.5)
    assert math.isclose(cleaned.loc[0, "revenue_per_employee"], 120000.0)
    assert cleaned.loc[0, "runway_bucket"] == "critical"
