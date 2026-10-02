# ---------- Stage 1: train the model ----------
# The dataset is downloaded (checksum-verified) and the pipeline is trained
# at build time, so neither data nor model artifacts live in the Git repo.
FROM python:3.12-slim AS trainer

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY training/ training/
ARG MODEL_VERSION=1.0.0
RUN python training/train.py --download --model-version "${MODEL_VERSION}" --model-dir /build/models


# ---------- Stage 2: minimal runtime image ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/model.joblib \
    METADATA_PATH=/app/models/metadata.json \
    LOG_LEVEL=INFO \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && groupadd --system app && useradd --system --gid app --no-create-home app

COPY --chown=app:app src/ src/
COPY --from=trainer --chown=app:app /build/models/ models/

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)"

# Cloud Run / Container Apps inject $PORT; default is 8000 locally.
CMD ["sh", "-c", "exec uvicorn diabetes_api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-access-log"]
