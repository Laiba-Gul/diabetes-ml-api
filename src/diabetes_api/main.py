"""FastAPI application exposing the diabetes prediction model."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from diabetes_api import __version__
from diabetes_api.config import Settings, get_settings
from diabetes_api.features import FEATURE_COLUMNS
from diabetes_api.logging_config import configure_logging
from diabetes_api.model import ModelNotLoadedError, ModelService
from diabetes_api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PatientFeatures,
    PredictionResponse,
)

logger = logging.getLogger("diabetes_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        app.state.model = ModelService.load(
            settings.model_path, settings.metadata_path, settings.decision_threshold
        )
    except ModelNotLoadedError as exc:
        # Start anyway so /health can report the problem; /predict returns 503.
        app.state.model = None
        logger.error("model_load_failed", extra={"reason": str(exc)})
    yield


app = FastAPI(
    title="Diabetes Prediction API",
    description=(
        "Predicts the likelihood of diabetes from the 8 PIMA Indians Diabetes features. "
        "For demonstration and educational purposes only - not a medical device."
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Log one line per request with operational metadata only (no bodies)."""
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_failed",
            extra={"request_id": request_id, "method": request.method, "path": request.url.path},
        )
        raise
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        },
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return validation errors without echoing the submitted values back."""
    errors = [
        {"loc": list(err.get("loc", [])), "msg": err.get("msg"), "type": err.get("type")}
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": errors}
    )


def get_model(request: Request) -> ModelService:
    model: ModelService | None = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded"
        )
    return model


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"service": "diabetes-prediction-api", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request, response: Response) -> HealthResponse:
    model: ModelService | None = getattr(request.app.state, "model", None)
    if model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="unavailable", model_loaded=False)
    return HealthResponse(status="ok", model_loaded=True, model_version=model.version)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
def model_info(
    model: ModelService = Depends(get_model), settings: Settings = Depends(get_settings)
) -> ModelInfoResponse:
    meta = model.metadata
    return ModelInfoResponse(
        model_version=model.version,
        model_type=str(meta.get("model_type", type(model.pipeline[-1]).__name__)),
        features=FEATURE_COLUMNS,
        decision_threshold=settings.decision_threshold,
        trained_at=meta.get("trained_at"),
        metrics=meta.get("test_metrics"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["model"])
def predict(
    payload: PatientFeatures, model: ModelService = Depends(get_model)
) -> PredictionResponse:
    result = model.predict(payload.model_dump())
    return PredictionResponse(
        predicted_class=result.predicted_class,
        label=result.label,
        probability=result.probability,
        model_version=result.model_version,
    )
