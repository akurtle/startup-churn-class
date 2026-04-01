from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "startup_churn.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"

TARGET_COLUMN = "churned"
RANDOM_STATE = 42

RAW_NUMERIC_FEATURES = [
    "company_age_months",
    "monthly_burn_usd",
    "runway_months",
    "team_size",
    "founder_exits",
    "customer_growth_pct",
    "support_tickets_last_30_days",
    "annual_revenue_usd",
]

RAW_CATEGORICAL_FEATURES = [
    "market_segment",
    "growth_stage",
    "remote_friendly",
    "investor_tier",
]

ENGINEERED_NUMERIC_FEATURES = [
    "burn_to_revenue_ratio",
    "revenue_per_employee",
]

ENGINEERED_CATEGORICAL_FEATURES = [
    "runway_bucket",
]

INPUT_FEATURE_COLUMNS = RAW_NUMERIC_FEATURES + RAW_CATEGORICAL_FEATURES
NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
CATEGORICAL_FEATURES = RAW_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES
FEATURE_COLUMNS = INPUT_FEATURE_COLUMNS
MODEL_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
