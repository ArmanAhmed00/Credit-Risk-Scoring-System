"""Tests for app/client.py - the Streamlit front end's API client.

No Streamlit runtime and no live server: httpx is patched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "app"))

import client  # noqa: E402

FORM = {
    "age": 30, "income": 60000, "emp_length": 5.0, "loan_amnt": 10000,
    "int_rate": 12.5, "percent_income": 0.17, "cred_hist": 4, "home": "RENT",
    "intent": "PERSONAL", "grade": "C", "prior_default": "N",
}


def _response(status: int, payload=None, text: str = ""):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload if payload is not None else {}
    r.text = text
    return r


def test_build_payload_has_exactly_the_api_fields():
    from api.schemas import PredictionRequest

    payload = client.build_payload(**FORM)
    assert set(payload) == set(PredictionRequest.model_fields)


def test_build_payload_is_accepted_by_the_api_schema():
    """The form cannot produce a body the API would reject."""
    from api.schemas import PredictionRequest

    PredictionRequest(**client.build_payload(**FORM))


def test_build_payload_coerces_types():
    payload = client.build_payload(**{**FORM, "age": 30.0, "cred_hist": 4.0})
    assert isinstance(payload["person_age"], int)
    assert isinstance(payload["cb_person_cred_hist_length"], int)
    assert isinstance(payload["person_income"], float)


def test_score_success():
    body = {"default_probability": 0.0917, "risk_label": "LOW",
            "model_version": "1", "request_id": "abc"}
    with patch.object(client.httpx, "post", return_value=_response(200, body)):
        ok, result = client.score_application(client.build_payload(**FORM))
    assert ok and result["risk_label"] == "LOW"


def test_score_422_names_the_bad_field():
    err = {"detail": [{"loc": ["body", "person_age"], "type": "greater_than_equal"}]}
    with patch.object(client.httpx, "post", return_value=_response(422, err)):
        ok, msg = client.score_application({})
    assert not ok
    assert "person_age" in msg


def test_score_503_explains_the_fix():
    with patch.object(client.httpx, "post", return_value=_response(503, {})):
        ok, msg = client.score_application({})
    assert not ok
    assert "Train and promote" in msg


def test_score_handles_unreachable_api():
    """A dead API must surface a message, never a traceback in the UI."""
    with patch.object(client.httpx, "post", side_effect=OSError("connection refused")):
        ok, msg = client.score_application({}, api_url="http://127.0.0.1:9999")
    assert not ok
    assert "Could not reach the API" in msg


def test_score_handles_unexpected_status():
    with patch.object(client.httpx, "post", return_value=_response(500, {}, "boom")):
        ok, msg = client.score_application({})
    assert not ok
    assert "500" in msg


def test_health_ok():
    body = {"status": "ok", "model_name": "CreditRiskModel", "model_version": "1"}
    with patch.object(client.httpx, "get", return_value=_response(200, body)):
        ok, payload = client.api_health()
    assert ok and payload["model_version"] == "1"


def test_health_handles_unreachable_api():
    with patch.object(client.httpx, "get", side_effect=OSError("refused")):
        ok, msg = client.api_health()
    assert not ok and isinstance(msg, str)


def test_health_non_200_is_not_ok():
    with patch.object(client.httpx, "get", return_value=_response(503, {"status": "degraded"})):
        ok, _ = client.api_health()
    assert not ok


def test_api_url_default_and_override(monkeypatch):
    monkeypatch.delenv("API_URL", raising=False)
    assert client.get_api_url() == "http://127.0.0.1:8000"
    monkeypatch.setenv("API_URL", "http://api:8000")
    assert client.get_api_url() == "http://api:8000"


@pytest.mark.parametrize("label", ["LOW", "MEDIUM", "HIGH"])
def test_every_risk_band_has_an_action(label):
    assert client.RISK_ACTIONS[label]
