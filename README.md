# Diabetes Prediction API — End-to-End ML Engineering Project

![CI](https://github.com/Laiba-Gul/diabetes-ml-api/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.142-009688)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E)

A production-style machine learning service that predicts diabetes from the 8 clinical features of the
**PIMA Indians Diabetes** dataset. It covers the full lifecycle: reproducible training, a single
serialized preprocessing + model pipeline, a validated REST API, privacy-safe logging, automated tests,
a hardened Docker image, and GitHub Actions CI.

> ⚠️ **Disclaimer:** educational/portfolio project. It is **not** a medical device and must not be used
> for diagnosis or treatment decisions.

---

## Table of contents
1. [Architecture](#architecture)
2. [Project structure](#project-structure)
3. [The model](#the-model)
4. [Quick start (local)](#quick-start-local)
5. [API reference](#api-reference)
6. [Testing](#testing)
7. [Docker](#docker)
8. [Continuous integration](#continuous-integration)
9. [Logging & privacy](#logging--privacy)

---

## Architecture

```
                 ┌───────────────────────── training/train.py ─────────────────────────┐
 PIMA CSV ──►  download + SHA-256 check ──► stratified split ──► 5-fold CV model selection
 (not in git)                                                    │
                                                                 ▼
                    models/model.joblib  (ZeroToNaN → median imputer → scaler → classifier)
                    models/metadata.json (version, features, CV + test metrics)
                                                                 │
 Client ──HTTP──► FastAPI (Pydantic validation) ──► ModelService.predict ──► JSON response
                   │
                   └─► JSON logs (request id, path, status, latency — never patient data)

 GitHub push ──► Actions CI: lint → pytest → train smoke test → docker build → container smoke test
```

## Project structure

```
diabetes-ml-api/
├── src/diabetes_api/
│   ├── main.py            # FastAPI app, routes, request-logging middleware
│   ├── schemas.py         # Pydantic request/response models + validation rules
│   ├── model.py           # Loads the saved pipeline, runs predictions
│   ├── pipeline.py        # Preprocessing + candidate classifiers (sklearn Pipeline)
│   ├── features.py        # Single source of truth for feature names/order
│   ├── config.py          # Settings from environment variables
│   └── logging_config.py  # JSON logging with sensitive-field filtering
├── training/train.py      # Data download, CV model selection, evaluation, artifact export
├── tests/                 # pytest: API, model/pipeline, logging privacy
├── .github/workflows/
│   └── ci.yml             # CI: lint, test, train, docker build + smoke test
├── Dockerfile             # Multi-stage: train in builder, slim non-root runtime
├── requirements.txt       # Pinned runtime dependencies
├── requirements-dev.txt   # + pytest, httpx, ruff
├── pyproject.toml         # pytest / ruff / coverage config
├── Makefile               # Shortcuts (make train, make test, ...)
├── .env.example           # Documented configuration variables
├── data/                  # Dataset is downloaded here (git-ignored)
└── models/                # Trained artifacts are written here (git-ignored)
```

## The model

**Dataset.** PIMA Indians Diabetes (originally from the National Institute of Diabetes and Digestive and
Kidney Diseases): 768 female patients aged 21+, 8 numeric features, binary outcome. The training script
downloads a public mirror and verifies its SHA-256 checksum. The dataset is never committed to Git.

| Feature | API field | Notes |
|---|---|---|
| Pregnancies | `pregnancies` | integer 0–20 |
| Glucose | `glucose` | mg/dL, 2-hour OGTT; `0` = not measured |
| BloodPressure | `blood_pressure` | diastolic, mm Hg; `0` = not measured |
| SkinThickness | `skin_thickness` | triceps skinfold, mm; `0` = not measured |
| Insulin | `insulin` | 2-hour serum insulin, µU/mL; `0` = not measured |
| BMI | `bmi` | kg/m²; `0` = not measured |
| DiabetesPedigreeFunction | `diabetes_pedigree_function` | genetic risk score, 0–3 |
| Age | `age` | years, 21–120 |

**Pipeline** (saved as one `sklearn.pipeline.Pipeline`, so inference needs no extra preprocessing code):

1. `ZeroToNaN` — in PIMA, a 0 in glucose / blood pressure / skin thickness / insulin / BMI means "missing";
   these zeros become `NaN`.
2. `SimpleImputer(strategy="median")` — medians learned from the training split only (no leakage).
3. `StandardScaler`.
4. Classifier chosen by 5-fold stratified cross-validated ROC-AUC on the training split from
   logistic regression, random forest and gradient boosting.

**Evaluation protocol.** 80/20 stratified train/test split (`random_state=42`). Model selection uses only
the training split; the held-out test split is scored exactly once.

**Results.** Your exact numbers are written to `models/metadata.json` every time you train. A reference
run of this code (scikit-learn 1.9.1, seed 42) produced:

| Model (5-fold CV on train) | ROC-AUC mean ± std |
|---|---|
| **Logistic regression (selected)** | **0.8434 ± 0.0185** |
| Random forest | 0.8272 ± 0.0252 |
| Gradient boosting | 0.8205 ± 0.0154 |

| Held-out test set (154 patients, threshold 0.5) | Value |
|---|---|
| ROC-AUC | 0.813 |
| Accuracy | 0.708 |
| Precision | 0.600 |
| Recall | 0.500 |
| F1 | 0.545 |
| Confusion matrix (TN / FP / FN / TP) | 82 / 18 / 27 / 27 |

At the default 0.5 threshold, recall is 0.50, so half of the diabetic patients in the test set were missed.
For a screening use case you would usually lower `DECISION_THRESHOLD` to trade precision for recall.
The threshold can be changed without retraining.

## Quick start (local)

Requires **Python 3.11+** (3.12 recommended) and Git.

```bash
# 1. Get the code
git clone https://github.com/Laiba-Gul/diabetes-ml-api.git
cd diabetes-ml-api

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1       # Windows PowerShell

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

# 4. Train (downloads the dataset to data/, writes models/model.joblib + models/metadata.json)
python training/train.py --download

# 5. Run the API (http://127.0.0.1:8000, interactive docs at /docs)
#   macOS / Linux:
PYTHONPATH=src uvicorn diabetes_api.main:app --reload --port 8000
#   Windows PowerShell:
#   $env:PYTHONPATH="src"; uvicorn diabetes_api.main:app --reload --port 8000
```

Training options: `python training/train.py --help` (`--model-version`, `--test-size`, `--cv-folds`,
`--seed`, `--data-path`, `--model-dir`). The script also accepts the Kaggle copy of the dataset (with
header row) via `--data-path`.

Configuration (environment variables, see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `models/model.joblib` | Saved pipeline |
| `METADATA_PATH` | `models/metadata.json` | Model metadata |
| `DECISION_THRESHOLD` | `0.5` | Probability cut-off for class 1 |
| `LOG_LEVEL` | `INFO` | Logging level |
| `PORT` | `8000` | Port the container listens on |

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness/readiness: `200` with model version, or `503` if no model is loaded |
| `GET` | `/model/info` | Model type, version, features, threshold, training date and test metrics |
| `POST` | `/predict` | Predicts diabetes for one patient |
| `GET` | `/docs` | Swagger UI (OpenAPI) |

### `POST /predict`

Request body: all 8 fields are required, and any extra field is rejected:

```json
{
  "pregnancies": 6,
  "glucose": 148,
  "blood_pressure": 72,
  "skin_thickness": 35,
  "insulin": 0,
  "bmi": 33.6,
  "diabetes_pedigree_function": 0.627,
  "age": 50
}
```

Response:

```json
{
  "predicted_class": 1,
  "label": "diabetic",
  "probability": 0.7058,
  "model_version": "1.0.0"
}
```

`probability` is the model's estimated probability of class 1 (diabetic). `predicted_class` is
`1` when `probability >= DECISION_THRESHOLD`.

Invalid input returns **422** with the failing field and rule. The submitted values are not echoed back:

```json
{"detail":[{"loc":["body","glucose"],"msg":"Input should be less than or equal to 300","type":"less_than_equal"}]}
```

### curl examples

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model/info

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pregnancies":6,"glucose":148,"blood_pressure":72,"skin_thickness":35,"insulin":0,"bmi":33.6,"diabetes_pedigree_function":0.627,"age":50}'

# Validation error example (glucose out of range) -> HTTP 422
curl -i -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pregnancies":1,"glucose":500,"blood_pressure":66,"skin_thickness":29,"insulin":0,"bmi":26.6,"diabetes_pedigree_function":0.351,"age":31}'
```

Windows PowerShell equivalent:

```powershell
$body = @{pregnancies=6; glucose=148; blood_pressure=72; skin_thickness=35; insulin=0; bmi=33.6; diabetes_pedigree_function=0.627; age=50} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $body
```

## Testing

```bash
python -m pytest                                      # run all tests
python -m pytest --cov --cov-report=term-missing      # with coverage
ruff check . && ruff format --check .                 # lint + formatting
```

The suite (33 tests) is **hermetic**: it trains a small model on synthetic data in a temp directory,
so it needs neither the real dataset nor a pre-trained artifact. It covers:

- **API** (`tests/test_api.py`): response contract, determinism, threshold consistency, higher risk inputs
  scoring higher, `0` accepted as "not measured", 422 for out-of-range / wrong-type / missing / unknown
  fields, no echo of invalid input, 503 when the model is missing, `/model/info`, OpenAPI.
- **Model** (`tests/test_model.py`): zero-to-missing transformer, saved pipeline contains preprocessing,
  artifacts and metadata, threshold behaviour, feature-schema mismatch detection, both CSV formats.
- **Privacy** (`tests/test_logging.py`): sensitive fields are stripped by the log formatter, and request
  logs contain no feature values or predictions.

## Docker

The multi-stage `Dockerfile`:
- **Stage 1 (trainer):** installs dependencies, downloads + verifies the dataset, trains the pipeline.
- **Stage 2 (runtime):** `python:3.12-slim`, runtime dependencies only, copies the trained model and app
  code, runs as a **non-root** user, has a `HEALTHCHECK`, and honours `$PORT`.

```bash
docker build -t diabetes-api:local .
docker build -t diabetes-api:1.0.0 --build-arg MODEL_VERSION=1.0.0 .   # set model version

docker run --rm -p 8000:8000 --name diabetes-api diabetes-api:local
docker run --rm -p 8000:8000 -e DECISION_THRESHOLD=0.35 diabetes-api:local   # custom threshold

curl http://127.0.0.1:8000/health
docker logs diabetes-api      # JSON logs (from another terminal)
```

## Continuous integration

**`.github/workflows/ci.yml`** runs on every push and pull request to `main`:

1. Set up Python 3.12 with a pip cache, then install `requirements-dev.txt`.
2. Lint with `ruff check` and `ruff format --check`.
3. Run `pytest` with coverage.
4. Run a smoke test of training on the real dataset, and upload `metadata.json` as a build artifact.
5. Build the Docker image with Buildx and the GitHub Actions cache.
6. Start the container, poll `/health`, call `/predict` and print the container logs.

## Logging & privacy

- Logs are structured JSON on stdout, ready for any log aggregation tool.
- Each request produces one line containing only `request_id`, `method`, `path`, `status_code` and
  `latency_ms`. `X-Request-ID` is returned so a client can correlate requests.
- **Request bodies, feature values and predictions are never logged.** The formatter also drops any
  sensitive key (`glucose`, `bmi`, `probability`, …) even if a developer passes it by mistake. This is
  enforced by tests.
- Uvicorn's access log is disabled because it would record client IPs.
- Validation errors don't echo the submitted values back.
- Datasets, model files, `.env` files and credentials are excluded by `.gitignore` and
  `.dockerignore`.

## Possible extensions
- Track experiments and register models with MLflow.
- Monitor data drift on aggregated (non-identifying) feature statistics.
- Calibrate probabilities and choose the threshold from a cost-sensitive analysis.
- Add a batch prediction endpoint with request size limits.
- Add API-key or IAM authentication for non-demo use.

## License
MIT. The PIMA Indians Diabetes dataset is subject to its original terms of use.
