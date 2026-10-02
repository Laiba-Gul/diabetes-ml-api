"""Train the diabetes classifier and save the full preprocessing + model pipeline.

Usage:
    python training/train.py --download            # fetch data, train, save
    python training/train.py --data-path data/diabetes.csv

Outputs (in --model-dir, default ``models/``):
    model.joblib    complete sklearn Pipeline (preprocessing + classifier)
    metadata.json   model version, features, CV scores, held-out test metrics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# Allow `python training/train.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diabetes_api.features import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402
from diabetes_api.pipeline import CANDIDATE_MODELS, build_pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

# Public mirror of the original PIMA Indians Diabetes dataset (768 rows, no header).
DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
)
DATA_SHA256 = "6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af"

# Header names used by the popular Kaggle copy of the same dataset.
KAGGLE_COLUMNS = {
    "Pregnancies": "pregnancies",
    "Glucose": "glucose",
    "BloodPressure": "blood_pressure",
    "SkinThickness": "skin_thickness",
    "Insulin": "insulin",
    "BMI": "bmi",
    "DiabetesPedigreeFunction": "diabetes_pedigree_function",
    "Age": "age",
    "Outcome": "outcome",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_dataset(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading dataset from %s", DATA_URL)
    urllib.request.urlretrieve(DATA_URL, dest)  # noqa: S310 - fixed https URL
    digest = sha256(dest)
    if digest != DATA_SHA256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for downloaded dataset: {digest}")
    log.info("Dataset saved to %s (sha256 verified)", dest)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load either the headerless original CSV or the Kaggle CSV with headers."""
    first_line = path.read_text().splitlines()[0]
    if first_line[:1].isalpha():
        df = pd.read_csv(path).rename(columns=KAGGLE_COLUMNS)
    else:
        df = pd.read_csv(path, header=None, names=[*FEATURE_COLUMNS, TARGET_COLUMN])
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    df = df[[*FEATURE_COLUMNS, TARGET_COLUMN]]
    if df.isna().any().any():
        raise ValueError("Dataset contains empty cells")
    if not set(df[TARGET_COLUMN].unique()) <= {0, 1}:
        raise ValueError("Target column must be binary 0/1")
    return df


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def train(
    df: pd.DataFrame,
    model_dir: Path,
    model_version: str,
    test_size: float = 0.2,
    cv_folds: int = 5,
    seed: int = 42,
    candidates: list[str] | None = None,
) -> dict:
    X = df[FEATURE_COLUMNS]  # noqa: N806
    y = df[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    log.info("Train rows: %d, test rows: %d", len(X_train), len(X_test))

    # 1) Model selection with stratified k-fold CV on the training split only.
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_results: dict[str, dict[str, float]] = {}
    for name in candidates or list(CANDIDATE_MODELS):
        scores = cross_val_score(
            build_pipeline(name, seed), X_train, y_train, cv=cv, scoring="roc_auc"
        )
        cv_results[name] = {
            "roc_auc_mean": round(float(scores.mean()), 4),
            "roc_auc_std": round(float(scores.std()), 4),
        }
        log.info("CV %-20s ROC-AUC %.4f +/- %.4f", name, scores.mean(), scores.std())
    best = max(cv_results, key=lambda n: cv_results[n]["roc_auc_mean"])
    log.info("Selected model: %s", best)

    # 2) Refit the selected pipeline on the full training split.
    pipeline = build_pipeline(best, seed).fit(X_train, y_train)

    # 3) Evaluate once on the untouched test split.
    test_metrics = evaluate(y_test.to_numpy(), pipeline.predict_proba(X_test)[:, 1])
    log.info("Test metrics: %s", json.dumps(test_metrics))

    # 4) Persist pipeline + metadata.
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "model.joblib")
    metadata = {
        "model_version": model_version,
        "model_type": best,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_state": seed,
        "decision_threshold": 0.5,
        "cv_folds": cv_folds,
        "cv_results": cv_results,
        "test_metrics": {k: v for k, v in test_metrics.items() if isinstance(v, float)},
        "confusion_matrix": {k: v for k, v in test_metrics.items() if isinstance(v, int)},
        "sklearn_version": sklearn.__version__,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info("Saved pipeline and metadata to %s", model_dir)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-path", type=Path, default=Path("data/diabetes.csv"))
    p.add_argument("--download", action="store_true", help="Download the dataset if missing")
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--model-version", default="1.0.0")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.data_path.exists():
        if not args.download:
            raise SystemExit(f"{args.data_path} not found. Re-run with --download.")
        download_dataset(args.data_path)
    df = load_dataset(args.data_path)
    log.info("Loaded %d rows from %s", len(df), args.data_path)
    train(df, args.model_dir, args.model_version, args.test_size, args.cv_folds, args.seed)


if __name__ == "__main__":
    main()
