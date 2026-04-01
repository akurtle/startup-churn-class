from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import torch

from startup_churn_classifier.config import ARTIFACTS_DIR, FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS
from startup_churn_classifier.models.pytorch_mlp import StartupMLP, predict_probabilities
from startup_churn_classifier.preprocessing import clean_startup_frame


@dataclass
class PredictionResult:
    churn_probability: float
    predicted_label: int
    selected_model: str


class StartupChurnPredictor:
    def __init__(self) -> None:
        metadata_path = ARTIFACTS_DIR / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                "Artifacts not found. Run `python train.py` before starting the API."
            )

        self.metadata = json.loads(metadata_path.read_text())
        self.threshold = float(self.metadata.get("threshold", 0.5))
        self.model_family = self.metadata["model_family"]
        self.selected_model = self.metadata["selected_model"]

        if self.model_family == "sklearn":
            self.pipeline = joblib.load(ARTIFACTS_DIR / "model.joblib")
            self.preprocessor = None
            self.model = None
        else:
            self.preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
            transformed_width = self.preprocessor.transform(
                pd.DataFrame([{feature: np.nan for feature in MODEL_FEATURE_COLUMNS}])
            )
            if hasattr(transformed_width, "toarray"):
                transformed_width = transformed_width.toarray()
            input_dim = int(np.asarray(transformed_width).shape[1])
            self.model = StartupMLP(input_dim=input_dim)
            state_dict = torch.load(ARTIFACTS_DIR / "model.pt", map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.pipeline = None

    def predict(self, payload: dict[str, object]) -> PredictionResult:
        frame = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
        cleaned = clean_startup_frame(frame)[MODEL_FEATURE_COLUMNS]

        if self.model_family == "sklearn":
            probability = float(self.pipeline.predict_proba(cleaned)[:, 1][0])
        else:
            transformed = self.preprocessor.transform(cleaned)
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            probability = float(
                predict_probabilities(self.model, np.asarray(transformed, dtype=np.float32))[0]
            )

        return PredictionResult(
            churn_probability=probability,
            predicted_label=int(probability >= self.threshold),
            selected_model=self.selected_model,
        )
