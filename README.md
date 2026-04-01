# Startup Churn Classifier

This project packages an end-to-end binary classification workflow for startup churn prediction. It handles messy tabular input, benchmarks multiple model families, persists the selected model, and serves predictions through FastAPI in a Docker-friendly shape.

## Stack

- Python
- Pandas and NumPy for cleaning, coercion, and imputation-ready feature preparation
- Scikit-learn for preprocessing, Logistic Regression, and Random Forest baselines
- PyTorch for a feed-forward MLP benchmark
- FastAPI for inference
- Docker for packaging

## What the project does

- Generates a messy startup dataset if `data/raw/startup_churn.csv` is missing.
- Cleans inconsistent currencies, percentages, booleans, mixed casing, and missing values.
- Trains three models and compares them on precision, recall, and ROC AUC.
- Applies a small complexity penalty so the final selection reflects resource-sensitive deployment trade-offs.
- Saves the selected model into `artifacts/` and exposes it through an HTTP API.

## Project structure

```text
startup_churn_classifier/
  api/
  models/
data/
  raw/
artifacts/
tests/
train.py
Dockerfile
```

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
uvicorn startup_churn_classifier.api.main:app --reload
```

## API usage

`POST /predict`

```json
{
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
  "investor_tier": "tier-2-vc"
}
```

Example response:

```json
{
  "selected_model": "random_forest",
  "churn_probability": 0.2143,
  "predicted_label": 0
}
```

## Tests

```bash
pytest
```

## Docker

```bash
docker build -t startup-churn-classifier .
docker run -p 8000:8000 startup-churn-classifier
```

## CI

GitHub Actions runs `python train.py`, `pytest`, and a Docker image build on pushes to `main` and on pull requests. The workflow lives in `.github/workflows/ci.yml`.
