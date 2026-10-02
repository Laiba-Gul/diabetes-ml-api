"""Loading the saved pipeline and running predictions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from diabetes_api.features import CLASS_LABELS, FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested but no model is available."""


@dataclass
class Prediction:
    predicted_class: int
    label: str
    probability: float
    model_version: str


@dataclass
class ModelService:
    pipeline: Pipeline
    metadata: dict[str, Any] = field(default_factory=dict)
    threshold: float = 0.5

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    @classmethod
    def load(cls, model_path: Path, metadata_path: Path, threshold: float = 0.5) -> ModelService:
        if not model_path.exists():
            raise ModelNotLoadedError(f"Model file not found: {model_path}")
        pipeline = joblib.load(model_path)
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        expected = metadata.get("features")
        if expected and list(expected) != FEATURE_COLUMNS:
            raise ModelNotLoadedError("Model feature schema does not match the API schema")
        service = cls(pipeline=pipeline, metadata=metadata, threshold=threshold)
        logger.info("model_loaded", extra={"model_version": service.version})
        return service

    def predict(self, features: dict[str, float]) -> Prediction:
        frame = pd.DataFrame([[features[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        probability = float(self.pipeline.predict_proba(frame)[0, 1])
        predicted_class = int(probability >= self.threshold)
        return Prediction(
            predicted_class=predicted_class,
            label=CLASS_LABELS[predicted_class],
            probability=round(probability, 4),
            model_version=self.version,
        )
