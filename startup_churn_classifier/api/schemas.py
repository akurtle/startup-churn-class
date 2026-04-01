from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


NULL_TOKENS = {"", "na", "n/a", "nan", "none", "null", "unknown", "missing"}
TRUE_TOKENS = {"true", "1", "yes", "y", "remote"}
FALSE_TOKENS = {"false", "0", "no", "n", "onsite"}

MARKET_SEGMENTS = {"fintech", "saas", "healthtech", "ai", "ecommerce"}
GROWTH_STAGES = {"pre-seed", "seed", "series-a", "series-b"}
INVESTOR_TIERS = {"none", "angel", "tier-2-vc", "tier-1-vc"}


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if text in NULL_TOKENS:
        return None
    return text


def _normalize_numeric(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric.")

    if isinstance(value, (int, float)):
        numeric_value = float(value)
    else:
        text = _normalize_text(value)
        if text is None:
            return None
        text = text.replace(",", "")
        text = re.sub(r"usd", "", text)
        text = text.replace("$", "")
        text = text.replace("%", "")
        text = text.strip()
        try:
            numeric_value = float(text)
        except ValueError as error:
            raise ValueError(f"{field_name} must be a valid number.") from error

    if numeric_value != numeric_value or numeric_value in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} must be a finite number.")

    return numeric_value


def _normalize_non_negative_numeric(value: Any, *, field_name: str) -> float | None:
    numeric_value = _normalize_numeric(value, field_name=field_name)
    if numeric_value is not None and numeric_value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0.")
    return numeric_value


def _normalize_integer(value: Any, *, field_name: str, minimum: int = 0) -> int | None:
    numeric_value = _normalize_non_negative_numeric(value, field_name=field_name)
    if numeric_value is None:
        return None
    if not float(numeric_value).is_integer():
        raise ValueError(f"{field_name} must be a whole number.")

    integer_value = int(numeric_value)
    if integer_value < minimum:
        raise ValueError(f"{field_name} must be greater than or equal to {minimum}.")
    return integer_value


def _normalize_choice(value: Any, *, field_name: str, allowed_values: set[str]) -> str | None:
    normalized = _normalize_text(value)
    if normalized is None:
        return None

    canonical = normalized.replace("_", "-").replace(" ", "-")
    if canonical not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(f"{field_name} must be one of: {allowed}.")
    return canonical


def _normalize_remote_friendly(value: Any) -> str | None:
    normalized = _normalize_text(value)
    if normalized is None:
        return None
    if normalized in TRUE_TOKENS:
        return "yes"
    if normalized in FALSE_TOKENS:
        return "no"
    raise ValueError("remote_friendly must be one of: yes, no, true, false, 1, 0, remote, onsite.")


class StartupFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    company_age_months: float | None = Field(default=None)
    monthly_burn_usd: float | None = Field(default=None)
    runway_months: float | None = Field(default=None)
    team_size: int | None = Field(default=None)
    founder_exits: int | None = Field(default=None)
    customer_growth_pct: float | None = Field(default=None)
    support_tickets_last_30_days: int | None = Field(default=None)
    annual_revenue_usd: float | None = Field(default=None)
    market_segment: str | None = Field(default=None)
    growth_stage: str | None = Field(default=None)
    remote_friendly: str | None = Field(default=None)
    investor_tier: str | None = Field(default=None)

    @field_validator("company_age_months", mode="before")
    @classmethod
    def validate_company_age_months(cls, value: Any) -> float | None:
        return _normalize_non_negative_numeric(value, field_name="company_age_months")

    @field_validator("monthly_burn_usd", mode="before")
    @classmethod
    def validate_monthly_burn_usd(cls, value: Any) -> float | None:
        return _normalize_non_negative_numeric(value, field_name="monthly_burn_usd")

    @field_validator("runway_months", mode="before")
    @classmethod
    def validate_runway_months(cls, value: Any) -> float | None:
        return _normalize_non_negative_numeric(value, field_name="runway_months")

    @field_validator("team_size", mode="before")
    @classmethod
    def validate_team_size(cls, value: Any) -> int | None:
        return _normalize_integer(value, field_name="team_size", minimum=1)

    @field_validator("founder_exits", mode="before")
    @classmethod
    def validate_founder_exits(cls, value: Any) -> int | None:
        return _normalize_integer(value, field_name="founder_exits")

    @field_validator("customer_growth_pct", mode="before")
    @classmethod
    def validate_customer_growth_pct(cls, value: Any) -> float | None:
        return _normalize_numeric(value, field_name="customer_growth_pct")

    @field_validator("support_tickets_last_30_days", mode="before")
    @classmethod
    def validate_support_tickets(cls, value: Any) -> int | None:
        return _normalize_integer(value, field_name="support_tickets_last_30_days")

    @field_validator("annual_revenue_usd", mode="before")
    @classmethod
    def validate_annual_revenue_usd(cls, value: Any) -> float | None:
        return _normalize_non_negative_numeric(value, field_name="annual_revenue_usd")

    @field_validator("market_segment", mode="before")
    @classmethod
    def validate_market_segment(cls, value: Any) -> str | None:
        return _normalize_choice(
            value,
            field_name="market_segment",
            allowed_values=MARKET_SEGMENTS,
        )

    @field_validator("growth_stage", mode="before")
    @classmethod
    def validate_growth_stage(cls, value: Any) -> str | None:
        return _normalize_choice(
            value,
            field_name="growth_stage",
            allowed_values=GROWTH_STAGES,
        )

    @field_validator("remote_friendly", mode="before")
    @classmethod
    def validate_remote_friendly(cls, value: Any) -> str | None:
        return _normalize_remote_friendly(value)

    @field_validator("investor_tier", mode="before")
    @classmethod
    def validate_investor_tier(cls, value: Any) -> str | None:
        return _normalize_choice(
            value,
            field_name="investor_tier",
            allowed_values=INVESTOR_TIERS,
        )

    def to_inference_payload(self) -> dict[str, Any]:
        return self.model_dump()
