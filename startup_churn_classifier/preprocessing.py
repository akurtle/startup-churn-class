from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from startup_churn_classifier.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


NULL_TOKENS = {"", "na", "n/a", "nan", "none", "null", "unknown", "missing"}
TRUE_TOKENS = {"true", "1", "yes", "y", "remote"}
FALSE_TOKENS = {"false", "0", "no", "n", "onsite"}


def _standardize_columns(columns: Iterable[str]) -> list[str]:
    return [str(column).strip().lower().replace(" ", "_") for column in columns]


def _parse_numeric(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().lower()
    if text in NULL_TOKENS:
        return np.nan

    text = text.replace(",", "")
    text = re.sub(r"usd", "", text)
    text = text.replace("$", "")
    text = text.replace("%", "")
    text = text.strip()

    if not text:
        return np.nan

    try:
        return float(text)
    except ValueError:
        return np.nan


def _parse_boolean(value: object) -> str | float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    text = str(value).strip().lower()
    if text in NULL_TOKENS:
        return np.nan
    if text in TRUE_TOKENS:
        return "yes"
    if text in FALSE_TOKENS:
        return "no"
    return text


def _parse_category(value: object) -> str | float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    text = str(value).strip().lower()
    if text in NULL_TOKENS:
        return np.nan
    return text


def clean_startup_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = _standardize_columns(cleaned.columns)

    missing_features = set(FEATURE_COLUMNS) - set(cleaned.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {sorted(missing_features)}")

    for feature in NUMERIC_FEATURES:
        cleaned[feature] = cleaned[feature].map(_parse_numeric).astype(float)

    for feature in CATEGORICAL_FEATURES:
        parser = _parse_boolean if feature == "remote_friendly" else _parse_category
        cleaned[feature] = cleaned[feature].map(parser)

    return cleaned


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
