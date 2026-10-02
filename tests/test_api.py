"""API tests: endpoints, response contract and input validation."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import VALID_PAYLOAD


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "model_loaded": True, "model_version": "test-0.0.1"}


def test_health_reports_missing_model(client_without_model: TestClient) -> None:
    response = client_without_model.get("/health")
    assert response.status_code == 503
    assert response.json()["model_loaded"] is False


def test_predict_returns_contract(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"predicted_class", "label", "probability", "model_version"}
    assert body["predicted_class"] in (0, 1)
    assert body["label"] == {0: "non-diabetic", 1: "diabetic"}[body["predicted_class"]]
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == "test-0.0.1"
    assert "x-request-id" in response.headers


def test_predict_class_consistent_with_threshold(client: TestClient) -> None:
    body = client.post("/predict", json=VALID_PAYLOAD).json()
    assert body["predicted_class"] == int(body["probability"] >= 0.5)


def test_predict_is_deterministic(client: TestClient) -> None:
    first = client.post("/predict", json=VALID_PAYLOAD).json()
    second = client.post("/predict", json=VALID_PAYLOAD).json()
    assert first == second


def test_high_risk_scores_higher_than_low_risk(client: TestClient) -> None:
    low = {**VALID_PAYLOAD, "glucose": 80, "bmi": 21.0}
    high = {**VALID_PAYLOAD, "glucose": 195, "bmi": 45.0}
    p_low = client.post("/predict", json=low).json()["probability"]
    p_high = client.post("/predict", json=high).json()["probability"]
    assert p_high > p_low


def test_zero_means_missing_is_accepted(client: TestClient) -> None:
    payload = {**VALID_PAYLOAD, "skin_thickness": 0, "insulin": 0, "blood_pressure": 0}
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("glucose", -1),
        ("glucose", 301),
        ("bmi", 120),
        ("age", 15),
        ("age", 150),
        ("pregnancies", -2),
        ("pregnancies", 2.5),
        ("diabetes_pedigree_function", 5),
        ("blood_pressure", "high"),
    ],
)
def test_predict_rejects_invalid_values(client: TestClient, field: str, value: object) -> None:
    response = client.post("/predict", json={**VALID_PAYLOAD, field: value})
    assert response.status_code == 422
    assert any(field in err["loc"] for err in response.json()["detail"])


def test_predict_rejects_missing_field(client: TestClient) -> None:
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "glucose"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_unknown_field(client: TestClient) -> None:
    response = client.post("/predict", json={**VALID_PAYLOAD, "patient_name": "Jane"})
    assert response.status_code == 422


def test_validation_errors_do_not_echo_input(client: TestClient) -> None:
    response = client.post("/predict", json={**VALID_PAYLOAD, "glucose": 987654})
    assert response.status_code == 422
    assert "987654" not in response.text


def test_predict_returns_503_without_model(client_without_model: TestClient) -> None:
    response = client_without_model.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503


def test_model_info(client: TestClient) -> None:
    body = client.get("/model/info").json()
    assert body["model_version"] == "test-0.0.1"
    assert len(body["features"]) == 8
    assert body["decision_threshold"] == 0.5


def test_openapi_available(client: TestClient) -> None:
    assert client.get("/openapi.json").status_code == 200
