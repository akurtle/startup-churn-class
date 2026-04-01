from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from startup_churn_classifier.config import RANDOM_STATE


def _mess_up_numeric(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    currency: bool = False,
    percent: bool = False,
    allow_unknown: bool = True,
) -> list[object]:
    messy: list[object] = []
    missing_tokens = [None, np.nan, "", "unknown", "n/a"]

    for value in values:
        roll = rng.random()
        if allow_unknown and roll < 0.08:
            messy.append(rng.choice(missing_tokens))
            continue

        rounded = round(float(value), 2)
        if currency:
            options = [
                f"${rounded:,.0f}",
                f" USD {rounded:,.2f} ",
                f"{rounded:,.0f}",
            ]
            messy.append(rng.choice(options))
        elif percent:
            options = [f"{rounded:.1f}%", f" {rounded:.0f} % ", rounded]
            messy.append(rng.choice(options))
        else:
            options = [rounded, str(int(round(rounded))), f" {rounded} "]
            messy.append(rng.choice(options))

    return messy


def _mess_up_boolean(values: np.ndarray, rng: np.random.Generator) -> list[object]:
    true_tokens = ["yes", "Yes", "TRUE", "1", "remote"]
    false_tokens = ["no", "No", "FALSE", "0", "onsite"]
    missing_tokens = [None, "", "unknown"]
    messy: list[object] = []

    for value in values:
        roll = rng.random()
        if roll < 0.07:
            messy.append(rng.choice(missing_tokens))
        elif value:
            messy.append(rng.choice(true_tokens))
        else:
            messy.append(rng.choice(false_tokens))

    return messy


def _mess_up_category(values: np.ndarray, rng: np.random.Generator) -> list[object]:
    messy: list[object] = []
    missing_tokens = [None, "", "unknown"]

    for value in values:
        roll = rng.random()
        if roll < 0.06:
            messy.append(rng.choice(missing_tokens))
            continue

        text = str(value)
        options = [text, text.upper(), text.title(), f" {text.lower()} "]
        messy.append(rng.choice(options))

    return messy


def generate_synthetic_dataset(path: Path, rows: int = 1200) -> pd.DataFrame:
    """Create a messy but learnable churn dataset."""
    rng = np.random.default_rng(RANDOM_STATE)

    company_age_months = rng.integers(3, 120, size=rows)
    monthly_burn_usd = rng.lognormal(mean=11.2, sigma=0.55, size=rows)
    runway_months = np.clip(rng.normal(loc=13, scale=5, size=rows), 1, 36)
    team_size = np.clip(rng.normal(loc=26, scale=14, size=rows).round(), 2, 120)
    founder_exits = rng.poisson(lam=0.55, size=rows)
    customer_growth_pct = np.clip(rng.normal(loc=10, scale=18, size=rows), -35, 90)
    support_tickets_last_30_days = rng.poisson(lam=18, size=rows)
    annual_revenue_usd = np.clip(
        monthly_burn_usd * rng.normal(loc=16, scale=4, size=rows),
        30_000,
        None,
    )

    market_segment = rng.choice(
        ["fintech", "saas", "healthtech", "ai", "ecommerce"], size=rows
    )
    growth_stage = rng.choice(
        ["pre-seed", "seed", "series-a", "series-b"],
        size=rows,
        p=[0.27, 0.37, 0.24, 0.12],
    )
    remote_friendly = rng.choice([True, False], size=rows, p=[0.58, 0.42])
    investor_tier = rng.choice(
        ["none", "angel", "tier-2-vc", "tier-1-vc"],
        size=rows,
        p=[0.18, 0.34, 0.31, 0.17],
    )

    stage_risk = {
        "pre-seed": 1.15,
        "seed": 0.75,
        "series-a": 0.15,
        "series-b": -0.2,
    }
    investor_risk = {
        "none": 0.95,
        "angel": 0.4,
        "tier-2-vc": -0.1,
        "tier-1-vc": -0.35,
    }
    segment_risk = {
        "fintech": 0.1,
        "saas": -0.2,
        "healthtech": 0.05,
        "ai": 0.3,
        "ecommerce": 0.25,
    }

    burn_ratio = monthly_burn_usd / (annual_revenue_usd / 12)
    logits = (
        -1.2
        + 0.12 * burn_ratio
        - 0.08 * runway_months
        - 0.02 * company_age_months
        - 0.015 * team_size
        + 0.07 * support_tickets_last_30_days
        - 0.03 * customer_growth_pct
        + 0.35 * founder_exits
        + np.vectorize(stage_risk.get)(growth_stage)
        + np.vectorize(investor_risk.get)(investor_tier)
        + np.vectorize(segment_risk.get)(market_segment)
        - 0.18 * remote_friendly.astype(float)
    )
    probabilities = 1 / (1 + np.exp(-logits))
    churned = rng.binomial(1, probabilities)

    frame = pd.DataFrame(
        {
            "company_age_months": _mess_up_numeric(
                company_age_months, rng=rng, allow_unknown=True
            ),
            "monthly_burn_usd": _mess_up_numeric(
                monthly_burn_usd, rng=rng, currency=True
            ),
            "runway_months": _mess_up_numeric(runway_months, rng=rng),
            "team_size": _mess_up_numeric(team_size, rng=rng),
            "founder_exits": _mess_up_numeric(founder_exits, rng=rng, allow_unknown=False),
            "customer_growth_pct": _mess_up_numeric(
                customer_growth_pct, rng=rng, percent=True
            ),
            "support_tickets_last_30_days": _mess_up_numeric(
                support_tickets_last_30_days, rng=rng
            ),
            "annual_revenue_usd": _mess_up_numeric(
                annual_revenue_usd, rng=rng, currency=True
            ),
            "market_segment": _mess_up_category(market_segment, rng),
            "growth_stage": _mess_up_category(growth_stage, rng),
            "remote_friendly": _mess_up_boolean(remote_friendly, rng),
            "investor_tier": _mess_up_category(investor_tier, rng),
            "churned": churned,
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame
