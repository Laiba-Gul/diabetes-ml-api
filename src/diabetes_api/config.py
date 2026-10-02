"""Runtime configuration, read from environment variables (12-factor style)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    app_name: str = "Diabetes Prediction API"
    model_path: Path = Field(default=Path("models/model.joblib"), alias="MODEL_PATH")
    metadata_path: Path = Field(default=Path("models/metadata.json"), alias="METADATA_PATH")
    decision_threshold: float = Field(default=0.5, ge=0.0, le=1.0, alias="DECISION_THRESHOLD")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    return Settings()
