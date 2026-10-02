"""Tests for the training pipeline, saved artifact and prediction service."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from diabetes_api.features import FEATURE_COLUMNS
from diabetes_api.model import ModelNotLoadedError, ModelService
from diabetes_api.pipeline import ZeroToNaN, build_pipeline
from tests.conftest import VALID_PAYLOAD
from training.train import load_dataset


def test_zero_to_nan_only_touches_configured_columns() -> None:
    df = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0.5, 30]], columns=FEATURE_COLUMNS)
    out = ZeroToNaN(columns=["glucose", "bmi"]).fit(df).transform(df)
    assert np.isnan(out.loc[0, "glucose"]) and np.isnan(out.loc[0, "bmi"])
    assert out.loc[0, "pregnancies"] == 0  # zero pregnancies is a valid value
    assert df.loc[0, "glucose"] == 0  # input frame is not mutated


def test_build_pipeline_rejects_unknown_model() -> None:
    with pytest.raises(ValueError):
        build_pipeline("not_a_model")


def test_artifacts_written(trained_model_dir: Path) -> None:
    assert (trained_model_dir / "model.joblib").exists()
    meta = json.loads((trained_model_dir / "metadata.json").read_text())
    assert meta["features"] == FEATURE_COLUMNS
    assert meta["model_version"] == "test-0.0.1"
    assert set(meta["test_metrics"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_saved_pipeline_includes_preprocessing(trained_model_dir: Path) -> None:
    pipeline = joblib.load(trained_model_dir / "model.joblib")
    assert [name for name, _ in pipeline.steps] == [
        "zero_to_nan",
        "imputer",
        "scaler",
        "classifier",
    ]


def test_service_predict(trained_model_dir: Path) -> None:
    service = ModelService.load(
        trained_model_dir / "model.joblib", trained_model_dir / "metadata.json"
    )
    result = service.predict(VALID_PAYLOAD)
    assert result.predicted_class in (0, 1)
    assert 0.0 <= result.probability <= 1.0
    assert result.model_version == "test-0.0.1"


def test_threshold_controls_class(trained_model_dir: Path) -> None:
    paths = (trained_model_dir / "model.joblib", trained_model_dir / "metadata.json")
    assert ModelService.load(*paths, threshold=0.0).predict(VALID_PAYLOAD).predicted_class == 1
    assert ModelService.load(*paths, threshold=1.0).predict(VALID_PAYLOAD).predicted_class == 0


def test_load_missing_model_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelNotLoadedError):
        ModelService.load(tmp_path / "nope.joblib", tmp_path / "nope.json")


def test_feature_schema_mismatch_raises(trained_model_dir: Path, tmp_path: Path) -> None:
    meta = json.loads((trained_model_dir / "metadata.json").read_text())
    meta["features"] = list(reversed(meta["features"]))
    bad_meta = tmp_path / "metadata.json"
    bad_meta.write_text(json.dumps(meta))
    with pytest.raises(ModelNotLoadedError):
        ModelService.load(trained_model_dir / "model.joblib", bad_meta)


def test_load_dataset_supports_both_formats(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text("6,148,72,35,0,33.6,0.627,50,1\n1,85,66,29,0,26.6,0.351,31,0\n")
    kaggle = tmp_path / "kaggle.csv"
    kaggle.write_text(
        "Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,"
        "DiabetesPedigreeFunction,Age,Outcome\n6,148,72,35,0,33.6,0.627,50,1\n"
    )
    assert list(load_dataset(raw).columns[:8]) == FEATURE_COLUMNS
    assert load_dataset(kaggle).loc[0, "glucose"] == 148
