"""Construction of the complete preprocessing + model pipeline.

Everything the model needs at inference time lives inside one scikit-learn
``Pipeline`` object, so the saved artifact is self-contained: raw PIMA
features in, class probabilities out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diabetes_api.features import FEATURE_COLUMNS, ZERO_AS_MISSING_COLUMNS


class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Replace 0 with NaN in columns where 0 means "not measured"."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: object = None) -> ZeroToNaN:  # noqa: N803
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()  # noqa: N806
        cols = self.columns if self.columns is not None else ZERO_AS_MISSING_COLUMNS
        for col in cols:
            X[col] = X[col].astype(float).replace(0.0, np.nan)
        return X

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return self.feature_names_in_


CANDIDATE_MODELS = {
    "logistic_regression": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1
    ),
    "gradient_boosting": lambda seed: GradientBoostingClassifier(random_state=seed),
}


def build_pipeline(model_name: str = "logistic_regression", random_state: int = 42) -> Pipeline:
    """Return an unfitted preprocessing + classifier pipeline."""
    if model_name not in CANDIDATE_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {sorted(CANDIDATE_MODELS)}")
    return Pipeline(
        steps=[
            ("zero_to_nan", ZeroToNaN(columns=ZERO_AS_MISSING_COLUMNS)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", CANDIDATE_MODELS[model_name](random_state)),
        ]
    )


__all__ = ["CANDIDATE_MODELS", "FEATURE_COLUMNS", "ZeroToNaN", "build_pipeline"]
