"""Structured JSON logging.

Privacy rule for this service: log records must never contain request
bodies, feature values or prediction results, because together they are
patient health information. Only operational metadata is logged
(request id, path, status code, latency, model version).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

# Attributes every LogRecord has; anything else was passed via `extra=`.
_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

# Defence in depth: even if a developer passes one of these via `extra=`,
# it is dropped before the record is written.
SENSITIVE_KEYS = {
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree_function",
    "age",
    "features",
    "payload",
    "body",
    "probability",
    "prediction",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key in _STANDARD_ATTRS or key.startswith("_"):
                continue
            if key.lower() in SENSITIVE_KEYS:
                continue
            entry[key] = value
        if record.exc_info:
            entry["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(entry, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # Uvicorn's access log prints the client IP and full request line; we
    # replace it with our own privacy-safe request log middleware.
    logging.getLogger("uvicorn.access").disabled = True
