# Diabetes Prediction API — End-to-End ML Engineering Project

![CI](https://github.com/Laiba-Gul/diabetes-ml-api/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.142-009688)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E)

A production-style machine learning service that predicts diabetes from the 8 clinical features of the
**PIMA Indians Diabetes** dataset. It covers the full lifecycle: reproducible training, a single
serialized preprocessing + model pipeline, a validated REST API, privacy-safe logging, automated tests,
a hardened Docker image, GitHub Actions CI, and serverless deployment to Google Cloud Run.

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
8. [CI/CD](#cicd)
9. [Deployment (Google Cloud Run)](#deployment-google-cloud-run)
10. [Other clouds](#other-clouds)
11. [Logging & privacy](#logging--privacy)
12. [Push to GitHub](#push-to-github)

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
 Manual run  ──► Actions CD: build → push to Artifact Registry → deploy to Cloud Run
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
│   ├── ci.yml             # CI: lint, test, train, docker build + smoke test
│   └── deploy-cloudrun.yml# Optional manual CD to Google Cloud Run
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
| `PORT` | `8000` | Port (Docker / Cloud Run) |

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
  code, runs as a **non-root** user, has a `HEALTHCHECK`, and honours `$PORT` (required by Cloud Run).

```bash
docker build -t diabetes-api:local .
docker build -t diabetes-api:1.0.0 --build-arg MODEL_VERSION=1.0.0 .   # set model version

docker run --rm -p 8000:8000 --name diabetes-api diabetes-api:local
docker run --rm -p 8000:8000 -e DECISION_THRESHOLD=0.35 diabetes-api:local   # custom threshold

curl http://127.0.0.1:8000/health
docker logs diabetes-api      # JSON logs (from another terminal)
```

## CI/CD

**`.github/workflows/ci.yml`** runs on every push and pull request to `main`:

1. Set up Python 3.12 with a pip cache, then install `requirements-dev.txt`.
2. Lint with `ruff check` and `ruff format --check`.
3. Run `pytest` with coverage.
4. Run a smoke test of training on the real dataset, and upload `metadata.json` as a build artifact.
5. Build the Docker image with Buildx and the GitHub Actions cache.
6. Start the container, poll `/health`, call `/predict` and print the container logs.

**`.github/workflows/deploy-cloudrun.yml`** is optional and runs only when you trigger it manually. It
authenticates to Google Cloud with **Workload Identity Federation**, so no JSON key is stored in
GitHub. It then builds and pushes the image to Artifact Registry, deploys to Cloud Run, and smoke-tests
the live URL. Setup is described [below](#optional-automated-deploys-from-github-actions).

## Deployment (Google Cloud Run)

### Why Cloud Run

Free-tier offers checked on 3 October 2026:

| Provider | Offer | Fit |
|---|---|---|
| **Google Cloud Run** | Always-free monthly allowance (request-based billing): 180,000 vCPU-seconds, 360,000 GiB-seconds, 2 million requests. Cloud Build: 2,500 build-minutes/month (e2-standard-2). Artifact Registry: 0.5 GB storage/month. New accounts also get $300 of credit. | ✅ **Recommended.** Serverless containers that scale to zero, plus always-free limits that don't expire |
| Azure Container Apps | Monthly free grant: 180,000 vCPU-seconds, 360,000 GiB-seconds, 2 million requests | ✅ Good alternative |
| AWS | New-account Free plan: $100 credit plus up to $100 more by completing activities, valid for 6 months. The account closes when the credits or the 6 months run out, unless you upgrade. | ⚠️ Time-limited |

Free tiers change. Re-check [cloud.google.com/run/pricing](https://cloud.google.com/run/pricing) and
[cloud.google.com/free](https://cloud.google.com/free) before you deploy. Cloud Run still requires a
billing account. Usage above the free allowance is billed, so set a budget alert (step 2).

### Step-by-step (manual deploy with `gcloud`)

Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), then:

```bash
# 1. Log in and choose a project
gcloud auth login
gcloud projects create YOUR_PROJECT_ID          # or reuse an existing project
gcloud config set project YOUR_PROJECT_ID
# Link a billing account: Console -> Billing -> Link a billing account (required by Cloud Run)

# 2. Protect yourself: create a small budget alert
#    Console -> Billing -> Budgets & alerts -> Create budget (e.g. $1 with email alerts)

# 3. Enable the required APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# 4. Create a Docker repository in Artifact Registry
export REGION=us-central1
export PROJECT_ID=$(gcloud config get-value project)
gcloud artifacts repositories create ml-images \
  --repository-format=docker --location=$REGION \
  --description="Container images for ML services"

# 5. Build the image with Cloud Build and push it (uses the Dockerfile; trains the model during the build)
export IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/ml-images/diabetes-api:1.0.0
gcloud builds submit --tag $IMAGE .

# 6. Deploy to Cloud Run (scale-to-zero, max 1 instance to stay within free usage)
gcloud run deploy diabetes-api \
  --image=$IMAGE \
  --region=$REGION \
  --allow-unauthenticated \
  --memory=512Mi --cpu=1 \
  --min-instances=0 --max-instances=1 \
  --port=8000

# 7. Test the live service
export URL=$(gcloud run services describe diabetes-api --region=$REGION --format='value(status.url)')
curl $URL/health
curl -X POST $URL/predict -H "Content-Type: application/json" \
  -d '{"pregnancies":6,"glucose":148,"blood_pressure":72,"skin_thickness":35,"insulin":0,"bmi":33.6,"diabetes_pedigree_function":0.627,"age":50}'

# 8. View logs (JSON, no patient data)
gcloud run services logs read diabetes-api --region=$REGION --limit=20
```

Keep Artifact Registry under its 0.5 GB free storage by deleting old image versions:

```bash
gcloud artifacts docker images list $REGION-docker.pkg.dev/$PROJECT_ID/ml-images/diabetes-api --include-tags
gcloud artifacts docker images delete $REGION-docker.pkg.dev/$PROJECT_ID/ml-images/diabetes-api:OLD_TAG
```

Tear everything down when finished:

```bash
gcloud run services delete diabetes-api --region=$REGION
gcloud artifacts repositories delete ml-images --location=$REGION
```

### Optional: automated deploys from GitHub Actions

One-time setup of Workload Identity Federation (replace `YOUR_PROJECT_ID`; `Laiba-Gul/diabetes-ml-api`
is the GitHub repo allowed to deploy):

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
export REPO=Laiba-Gul/diabetes-ml-api

gcloud services enable iamcredentials.googleapis.com

gcloud iam service-accounts create github-deployer --display-name="GitHub Actions deployer"
export SA=github-deployer@$PROJECT_ID.iam.gserviceaccount.com
for ROLE in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA" --role="$ROLE"
done

gcloud iam workload-identity-pools create github --location=global --display-name="GitHub"
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global --workload-identity-pool=github \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='$REPO'"

gcloud iam service-accounts add-iam-policy-binding $SA \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/$REPO"

echo "GCP_WORKLOAD_IDENTITY_PROVIDER=projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers/github-provider"
echo "GCP_SERVICE_ACCOUNT=$SA"
```

In GitHub, go to **Settings → Secrets and variables → Actions → Variables** and add `GCP_PROJECT_ID`,
`GCP_REGION` (`us-central1`), `GAR_REPOSITORY` (`ml-images`), `GCP_WORKLOAD_IDENTITY_PROVIDER` and
`GCP_SERVICE_ACCOUNT`. Then open **Actions → Deploy to Cloud Run → Run workflow**.

## Other clouds

**Azure Container Apps** (monthly free grant, see table above). With the Azure CLI:

```bash
az login
az group create --name diabetes-rg --location eastus
az containerapp up --name diabetes-api --resource-group diabetes-rg --location eastus \
  --environment diabetes-env --source .
```

`--source .` builds the image from the Dockerfile, and the target port comes from `EXPOSE 8000`. This
command also creates an Azure Container Registry, which has its own cost outside the Container Apps
free grant. Delete the resource group when you're done: `az group delete --name diabetes-rg`.

**AWS.** The new-account Free plan is credit-based and lasts 6 months, so it is less suitable for a
portfolio service that should stay online. The same image runs on any container service there (for
example, push to ECR and run on ECS Fargate). Check the current free plan at
[aws.amazon.com/free](https://aws.amazon.com/free/) before deploying.

## Logging & privacy

- Logs are structured JSON on stdout, which Cloud Run / Azure / CloudWatch ingest natively.
- Each request produces one line containing only `request_id`, `method`, `path`, `status_code` and
  `latency_ms`. `X-Request-ID` is returned so a client can correlate requests.
- **Request bodies, feature values and predictions are never logged.** The formatter also drops any
  sensitive key (`glucose`, `bmi`, `probability`, …) even if a developer passes it by mistake. This is
  enforced by tests.
- Uvicorn's access log is disabled because it would record client IPs.
- Validation errors don't echo the submitted values back.
- Datasets, model files, `.env` files and cloud credentials are excluded by `.gitignore` and
  `.dockerignore`. CI/CD uses keyless Workload Identity Federation.

## Push to GitHub

```bash
cd diabetes-ml-api
git init -b main
git add .
git status                      # confirm data/*.csv, models/*.joblib and .env are NOT listed
git commit -m "Diabetes prediction API: training pipeline, FastAPI service, tests, Docker, CI"

# Create an empty repository named diabetes-ml-api on github.com (no README/.gitignore), then:
git remote add origin https://github.com/Laiba-Gul/diabetes-ml-api.git
git push -u origin main
```

The CI workflow starts automatically. Check its result in the repository's **Actions** tab.

## Possible extensions
- Track experiments and register models with MLflow.
- Monitor data drift on aggregated (non-identifying) feature statistics.
- Calibrate probabilities and choose the threshold from a cost-sensitive analysis.
- Add a batch prediction endpoint with request size limits.
- Add API-key or IAM authentication for non-demo use.

## License
MIT. The PIMA Indians Diabetes dataset is subject to its original terms of use.
