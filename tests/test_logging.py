"""Privacy tests: logs must never contain patient feature values or predictions."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from diabetes_api.logging_config import JsonFormatter
from tests.conftest import VALID_PAYLOAD


def test_formatter_drops_sensitive_extra_fields() -> None:
    record = logging.makeLogRecord(
        {"msg": "x", "levelname": "INFO", "glucose": 148, "probability": 0.9, "request_id": "abc"}
    )
    entry = json.loads(JsonFormatter().format(record))
    assert "glucose" not in entry and "probability" not in entry
    assert entry["request_id"] == "abc"


def test_request_logs_contain_no_patient_data(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    payload = {**VALID_PAYLOAD, "glucose": 173, "bmi": 41.3, "diabetes_pedigree_function": 1.234}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    formatter = JsonFormatter()
    lines = [json.loads(formatter.format(record)) for record in caplog.records]
    request_logs = [entry for entry in lines if entry.get("message") == "request_completed"]
    assert request_logs, "expected a request log line"
    allowed = {
        "timestamp",
        "level",
        "logger",
        "message",
        "request_id",
        "method",
        "path",
        "status_code",
        "latency_ms",
        "taskName",
    }
    for entry in lines:
        assert not set(entry) & set(VALID_PAYLOAD), entry
        assert "probability" not in entry and "prediction" not in entry
    for entry in request_logs:
        assert set(entry) <= allowed, entry
    raw = json.dumps(lines)
    for value in ("173", "41.3", "1.234"):
        assert value not in raw
