from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from startup_churn_classifier.config import (
    ARTIFACTS_DIR,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from startup_churn_classifier.data import generate_synthetic_dataset
from startup_churn_classifier.experiment_tracking import log_experiment_run
from startup_churn_classifier.models.pytorch_mlp import predict_probabilities, train_mlp
from startup_churn_classifier.models.pytorch_mlp import MLPTrainingConfig
from startup_churn_classifier.preprocessing import build_preprocessor, clean_startup_frame


@dataclass
class ModelResult:
    name: str
    precision: float
    recall: float
    roc_auc: float
    complexity_penalty: float

    @property
    def selection_score(self) -> float:
        return self.roc_auc + 0.35 * self.recall + 0.15 * self.precision - self.complexity_penalty


LOGISTIC_REGRESSION_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 1200,
    "random_state": RANDOM_STATE,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 250,
    "min_samples_leaf": 3,
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
}

DEFAULT_MLP_CONFIG = MLPTrainingConfig()


def ensure_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return generate_synthetic_dataset(path)


def _evaluate(name: str, y_true: np.ndarray, probabilities: np.ndarray) -> ModelResult:
    predictions = (probabilities >= 0.5).astype(int)
    penalties = {
        "logistic_regression": 0.03,
        "random_forest": 0.08,
        "pytorch_mlp": 0.12,
    }
    return ModelResult(
        name=name,
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        complexity_penalty=penalties[name],
    )


def _save_sklearn_artifacts(model_name: str, pipeline: Pipeline, result: ModelResult) -> None:
    joblib.dump(pipeline, ARTIFACTS_DIR / "model.joblib")
    metadata = {
        "selected_model": model_name,
        "model_family": "sklearn",
        "threshold": 0.5,
        "metrics": {
            "precision": result.precision,
            "recall": result.recall,
            "roc_auc": result.roc_auc,
            "selection_score": result.selection_score,
        },
        "features": FEATURE_COLUMNS,
    }
    (ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))


def _save_torch_artifacts(preprocessor, model, result: ModelResult) -> None:
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")
    torch.save(model.state_dict(), ARTIFACTS_DIR / "model.pt")
    metadata = {
        "selected_model": "pytorch_mlp",
        "model_family": "torch",
        "threshold": 0.5,
        "metrics": {
            "precision": result.precision,
            "recall": result.recall,
            "roc_auc": result.roc_auc,
            "selection_score": result.selection_score,
        },
        "features": FEATURE_COLUMNS,
    }
    (ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))


def run_training_pipeline() -> dict[str, object]:
    raw_frame = ensure_dataset()
    cleaned_frame = clean_startup_frame(raw_frame)

    dataset = cleaned_frame[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    dataset[TARGET_COLUMN] = pd.to_numeric(dataset[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)

    train_frame, test_frame = train_test_split(
        dataset,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=dataset[TARGET_COLUMN],
    )

    X_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame[TARGET_COLUMN].to_numpy()
    X_test = test_frame[FEATURE_COLUMNS]
    y_test = test_frame[TARGET_COLUMN].to_numpy()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[ModelResult] = []
    trained_models: dict[str, object] = {}

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
            ),
        ]
    )
    logistic_pipeline.fit(X_train, y_train)
    logistic_probabilities = logistic_pipeline.predict_proba(X_test)[:, 1]
    logistic_result = _evaluate("logistic_regression", y_test, logistic_probabilities)
    results.append(logistic_result)
    trained_models["logistic_regression"] = logistic_pipeline

    forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(**RANDOM_FOREST_PARAMS),
            ),
        ]
    )
    forest_pipeline.fit(X_train, y_train)
    forest_probabilities = forest_pipeline.predict_proba(X_test)[:, 1]
    forest_result = _evaluate("random_forest", y_test, forest_probabilities)
    results.append(forest_result)
    trained_models["random_forest"] = forest_pipeline

    mlp_preprocessor = build_preprocessor()
    X_train_transformed = mlp_preprocessor.fit_transform(X_train)
    X_test_transformed = mlp_preprocessor.transform(X_test)
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
        X_test_transformed = X_test_transformed.toarray()

    mlp_model = train_mlp(
        np.asarray(X_train_transformed, dtype=np.float32),
        y_train.astype(np.float32),
        seed=RANDOM_STATE,
        config=DEFAULT_MLP_CONFIG,
    )
    mlp_probabilities = predict_probabilities(
        mlp_model,
        np.asarray(X_test_transformed, dtype=np.float32),
    )
    mlp_result = _evaluate("pytorch_mlp", y_test, mlp_probabilities)
    results.append(mlp_result)
    trained_models["pytorch_mlp"] = (mlp_preprocessor, mlp_model)

    results.sort(key=lambda result: result.selection_score, reverse=True)
    best = results[0]

    if best.name == "pytorch_mlp":
        preprocessor, model = trained_models["pytorch_mlp"]
        _save_torch_artifacts(preprocessor, model, best)
    else:
        _save_sklearn_artifacts(best.name, trained_models[best.name], best)

    summary = {
        "dataset_path": str(RAW_DATA_PATH),
        "rows": int(len(dataset)),
        "positive_rate": float(dataset[TARGET_COLUMN].mean()),
        "selected_model": best.name,
        "train_test_split": {
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
        },
        "results": {
            result.name: {
                "precision": float(round(result.precision, 4)),
                "recall": float(round(result.recall, 4)),
                "roc_auc": float(round(result.roc_auc, 4)),
                "selection_score": float(round(result.selection_score, 4)),
            }
            for result in results
        },
    }
    tracking = log_experiment_run(
        summary=summary,
        hyperparameters={
            "train_test_split": summary["train_test_split"],
            "logistic_regression": LOGISTIC_REGRESSION_PARAMS,
            "random_forest": RANDOM_FOREST_PARAMS,
            "pytorch_mlp": asdict(DEFAULT_MLP_CONFIG),
        },
    )
    summary["experiment_tracking"] = tracking
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary
