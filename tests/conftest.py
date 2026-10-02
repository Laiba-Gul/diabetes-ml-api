"""Shared fixtures.

Tests never depend on the real dataset or a pre-trained artifact: a small
model is trained on synthetic data so the suite is fast and hermetic.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from diabetes_api.config import get_settings
from diabetes_api.features import FEATURE_COLUMNS, TARGET_COLUMN
from training.train import train

VALID_PAYLOAD = {
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree_function": 0.627,
    "age": 50,
}


def make_synthetic_dataset(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "pregnancies": rng.integers(0, 12, n),
            "glucose": rng.normal(120, 30, n).clip(50, 199).round(),
            "blood_pressure": rng.normal(70, 12, n).clip(30, 120).round(),
            "skin_thickness": rng.normal(25, 10, n).clip(0, 60).round(),
            "insulin": rng.normal(80, 60, n).clip(0, 400).round(),
            "bmi": rng.normal(32, 6, n).clip(18, 60).round(1),
            "diabetes_pedigree_function": rng.uniform(0.08, 2.4, n).round(3),
            "age": rng.integers(21, 80, n),
        }
    )
    # Sprinkle PIMA-style zeros (= missing) into some columns.
    for col in ["glucose", "blood_pressure", "insulin"]:
        df.loc[rng.random(n) < 0.05, col] = 0
    risk = 0.04 * (df["glucose"] - 120) + 0.1 * (df["bmi"] - 32) + rng.normal(0, 1, n)
    df[TARGET_COLUMN] = (risk > 0.5).astype(int)
    return df[[*FEATURE_COLUMNS, TARGET_COLUMN]]


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_dataset()


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory: pytest.TempPathFactory, synthetic_df: pd.DataFrame) -> Path:
    model_dir = tmp_path_factory.mktemp("models")
    train(
        synthetic_df,
        model_dir,
        model_version="test-0.0.1",
        cv_folds=3,
        candidates=["logistic_regression"],
    )
    return model_dir


@pytest.fixture()
def client(trained_model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("MODEL_PATH", str(trained_model_dir / "model.joblib"))
    monkeypatch.setenv("METADATA_PATH", str(trained_model_dir / "metadata.json"))
    get_settings.cache_clear()
    from diabetes_api.main import app

    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


@pytest.fixture()
def client_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("METADATA_PATH", str(tmp_path / "missing.json"))
    get_settings.cache_clear()
    from diabetes_api.main import app

    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()
