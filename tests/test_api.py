"""Tests for the credit risk scoring API.

The MLflow model is stubbed: these run offline with no tracking server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from api import main as api_main  # noqa: E402
from api.features import FEATURE_ORDER, build_feature_row  # noqa: E402

VALID_PAYLOAD = {
    "person_age": 30,
    "person_income": 60000,
    "person_emp_length": 5.0,
    "loan_amnt": 10000,
    "loan_int_rate": 12.5,
    "loan_percent_income": 0.17,
    "cb_person_cred_hist_length": 4,
    "person_home_ownership": "RENT",
    "loan_intent": "PERSONAL",
    "loan_grade": "C",
    "cb_person_default_on_file": "N",
}

ENCODING_MAPS = {
    "person_home_ownership": {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3},
    "loan_intent": {
        "PERSONAL": 0, "EDUCATION": 1, "MEDICAL": 2,
        "VENTURE": 3, "HOMEIMPROVEMENT": 4, "DEBTCONSOLIDATION": 5,
    },
    "loan_grade": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7},
    "cb_person_default_on_file": {"N": 0, "Y": 1},
    "feature_order": FEATURE_ORDER,
    "target_column": "loan_status",
}


class StubModel:
    """Returns a fixed positive-class probability, like a real predict_proba."""

    def __init__(self, probability: float = 0.42):
        self.probability = probability
        self.calls: list = []

    def predict_proba(self, frame):
        self.calls.append(frame)
        return [[1 - self.probability, self.probability]]


@pytest.fixture
def stub_model():
    return StubModel()


@pytest.fixture
def client(stub_model, tmp_path, monkeypatch):
    """TestClient with the model stubbed and logs redirected to tmp_path.

    load_model/load_encoding_maps are patched BEFORE the app starts, so the
    lifespan hook never contacts a real MLflow server. Without this the suite
    silently depends on localhost:5000 being reachable.
    """
    log_path = tmp_path / "prediction_log.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://offline-test-should-not-connect:5000")

    monkeypatch.setattr(api_main, "load_model", lambda *a, **k: stub_model)
    monkeypatch.setattr(api_main, "load_encoding_maps", lambda *a, **k: ENCODING_MAPS)
    monkeypatch.setattr(api_main, "resolve_model_version", lambda *a, **k: "7")

    with TestClient(api_main.app) as c:
        c.log_path = log_path
        yield c


@pytest.fixture
def raising_client(stub_model, tmp_path, monkeypatch):
    """Same app, but surfaces 500 responses instead of re-raising them.

    Starlette's ServerErrorMiddleware returns the 500 to the caller and then
    re-raises so the server logs it; TestClient re-raises by default, which
    hides the response the real client would receive.
    """
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "p.jsonl"))
    monkeypatch.setattr(api_main, "load_model", lambda *a, **k: stub_model)
    monkeypatch.setattr(api_main, "load_encoding_maps", lambda *a, **k: ENCODING_MAPS)
    monkeypatch.setattr(api_main, "resolve_model_version", lambda *a, **k: "7")

    with TestClient(api_main.app, raise_server_exceptions=False) as c:
        yield c


def test_lifespan_loaded_model_and_maps(client):
    """The startup hook actually populated the bundle."""
    assert api_main.bundle.is_ready
    assert api_main.bundle.model_version == "7"


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200

    body = r.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "CreditRiskModel"
    assert body["model_version"] == "7"
    assert isinstance(body["uptime_seconds"], float)
    assert body["uptime_seconds"] >= 0


def test_health_503_when_model_missing(client, monkeypatch):
    monkeypatch.setattr(api_main.bundle, "model", None)
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "degraded"


def test_predict_valid(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200

    body = r.json()
    assert set(body) == {"default_probability", "risk_label", "model_version", "request_id"}
    assert body["default_probability"] == pytest.approx(0.42)
    assert body["risk_label"] == "MEDIUM"
    assert body["model_version"] == "7"
    assert len(body["request_id"]) == 36  # uuid4


def test_predict_sends_14_features_in_order(client, stub_model):
    client.post("/predict", json=VALID_PAYLOAD)
    frame = stub_model.calls[0]
    assert list(frame.columns) == FEATURE_ORDER
    assert frame.shape == (1, 14)


def test_predict_categoricals_are_case_insensitive(client):
    payload = {**VALID_PAYLOAD, "person_home_ownership": "rent", "loan_grade": "c"}
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize(
    "probability,expected",
    [
        (0.0, "LOW"), (0.15, "LOW"), (0.2999, "LOW"),
        (0.3, "MEDIUM"), (0.45, "MEDIUM"), (0.6, "MEDIUM"),
        (0.6001, "HIGH"), (0.85, "HIGH"), (1.0, "HIGH"),
    ],
)
def test_risk_label_thresholds(probability, expected):
    assert api_main.classify_risk(probability) == expected


@pytest.mark.parametrize(
    "probability,expected", [(0.1, "LOW"), (0.45, "MEDIUM"), (0.95, "HIGH")]
)
def test_risk_label_end_to_end(client, stub_model, probability, expected):
    stub_model.probability = probability
    body = client.post("/predict", json=VALID_PAYLOAD).json()
    assert body["risk_label"] == expected
    assert body["default_probability"] == pytest.approx(probability)


def test_threshold_boundaries_are_exclusive_of_low_inclusive_of_medium():
    """0.3 and 0.6 are MEDIUM; the bands must not leave a gap or overlap."""
    assert api_main.classify_risk(0.3) == "MEDIUM"
    assert api_main.classify_risk(0.6) == "MEDIUM"
    assert api_main.classify_risk(0.2999999) == "LOW"
    assert api_main.classify_risk(0.6000001) == "HIGH"


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("person_age", 17), ("person_age", 101), ("person_age", -5),
        ("person_income", 0), ("person_income", -1000),
        ("loan_amnt", 0), ("loan_amnt", -500),
        ("person_emp_length", -1),
        ("loan_percent_income", 1.5),
        ("cb_person_cred_hist_length", -1),
    ],
)
def test_invalid_values_return_422(client, field, bad_value):
    payload = {**VALID_PAYLOAD, field: bad_value}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422, f"{field}={bad_value} should be rejected"


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("person_home_ownership", "CASTLE"),
        ("loan_intent", "GAMBLING"),
        ("loan_grade", "Z"),
        ("cb_person_default_on_file", "MAYBE"),
    ],
)
def test_invalid_categoricals_return_422(client, field, bad_value):
    r = client.post("/predict", json={**VALID_PAYLOAD, field: bad_value})
    assert r.status_code == 422


@pytest.mark.parametrize("field", list(VALID_PAYLOAD))
def test_missing_field_returns_422(client, field):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != field}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422, f"missing {field} should be rejected"


def test_unknown_extra_field_returns_422(client):
    r = client.post("/predict", json={**VALID_PAYLOAD, "ssn": "123-45-6789"})
    assert r.status_code == 422


def test_empty_body_returns_422(client):
    assert client.post("/predict", json={}).status_code == 422


def test_wrong_type_returns_422(client):
    r = client.post("/predict", json={**VALID_PAYLOAD, "person_age": "thirty"})
    assert r.status_code == 422


def test_predict_503_when_model_not_loaded(client, monkeypatch):
    monkeypatch.setattr(api_main.bundle, "model", None)
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 503
    assert "Model not loaded" in r.json()["detail"]


def test_predict_503_when_encoding_maps_missing(client, monkeypatch):
    monkeypatch.setattr(api_main.bundle, "encoding_maps", None)
    assert client.post("/predict", json=VALID_PAYLOAD).status_code == 503


def test_unexpected_error_returns_500(raising_client, stub_model, monkeypatch):
    def boom(frame):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(stub_model, "predict_proba", boom)
    r = raising_client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 500
    assert r.json()["detail"] == "Internal server error"  # no internals leaked
    assert "exploded" not in r.text


def test_response_time_header_present(client):
    r = client.get("/health")
    assert "x-response-time-ms" in {k.lower() for k in r.headers}
    assert float(r.headers["x-response-time-ms"]) >= 0


def test_prediction_is_logged(client):
    body = client.post("/predict", json=VALID_PAYLOAD).json()

    assert client.log_path.exists(), "prediction log was not written"
    lines = client.log_path.read_text().strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["request_id"] == body["request_id"]
    assert entry["risk_label"] == body["risk_label"]
    assert entry["default_probability"] == pytest.approx(body["default_probability"])
    assert entry["model_version"] == "7"
    assert entry["timestamp"]
    assert entry["inputs"] == VALID_PAYLOAD


def test_log_appends_across_requests(client):
    for _ in range(3):
        client.post("/predict", json=VALID_PAYLOAD)
    assert len(client.log_path.read_text().strip().splitlines()) == 3


def test_failed_requests_are_not_logged(client):
    client.post("/predict", json={**VALID_PAYLOAD, "person_age": 5})
    assert not client.log_path.exists() or client.log_path.read_text().strip() == ""


def test_feature_row_matches_expected_math():
    row = build_feature_row(VALID_PAYLOAD, ENCODING_MAPS)
    assert row["debt_to_income"] == pytest.approx(10000 / 60000)
    assert row["credit_utilization"] == pytest.approx(10000 / (60000 * 0.3))
    assert row["loan_to_income"] == pytest.approx(0.17)
    assert row["home_ownership_enc"] == 0  # RENT
    assert row["loan_grade_enc"] == 3      # C
    assert row["cb_default_enc"] == 0      # N
    assert row["loan_intent_enc"] == 0     # PERSONAL
    assert list(row) == FEATURE_ORDER


def test_api_constants_match_data_cleaning():
    """api/features.py duplicates the cleaning constants; pin them together."""
    import data_cleaning

    from api import features

    assert features.INCOME_SERVICEABLE_SHARE == data_cleaning.INCOME_SERVICEABLE_SHARE
    expected = [c for c in data_cleaning.OUTPUT_COLUMNS if c != "loan_status"]
    assert features.FEATURE_ORDER == expected


def test_api_encoding_maps_match_cleaning_module():
    import data_cleaning

    assert ENCODING_MAPS["person_home_ownership"] == data_cleaning.HOME_OWNERSHIP_MAP
    assert ENCODING_MAPS["loan_intent"] == data_cleaning.LOAN_INTENT_MAP
    assert ENCODING_MAPS["loan_grade"] == data_cleaning.LOAN_GRADE_MAP
    assert ENCODING_MAPS["cb_person_default_on_file"] == data_cleaning.DEFAULT_ON_FILE_MAP


def test_exported_encoding_maps_file_is_loadable():
    """The real artifact scripts/train.py wrote must satisfy the API contract."""
    path = PROJECT_ROOT / "data" / "processed" / "encoding_maps.json"
    if not path.exists():
        pytest.skip("encoding_maps.json not found - run scripts/train.py")

    maps = api_main.load_encoding_maps(path)
    row = build_feature_row(VALID_PAYLOAD, maps)
    assert list(row) == FEATURE_ORDER


def test_health_via_shared_client(test_client):
    r = test_client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_name"] == "CreditRiskModel"
    assert r.json()["model_version"] == "test-1"


def test_predict_via_shared_client(test_client, valid_payload):
    r = test_client.post("/predict", json=valid_payload)
    assert r.status_code == 200

    body = r.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_label"] in {"LOW", "MEDIUM", "HIGH"}


def test_invalid_input_422_via_shared_client(test_client, valid_payload):
    r = test_client.post("/predict", json={**valid_payload, "person_age": 12})
    assert r.status_code == 422


def test_prediction_log_written_via_shared_client(test_client, valid_payload):
    """prediction_log.jsonl must exist and hold one record per success."""
    body = test_client.post("/predict", json=valid_payload).json()

    assert test_client.log_path.exists()
    entries = [json.loads(x) for x in test_client.log_path.read_text().strip().splitlines()]
    assert len(entries) == 1
    assert entries[0]["request_id"] == body["request_id"]
    assert entries[0]["inputs"] == valid_payload
    assert entries[0]["model_version"] == "test-1"


def test_dummy_model_is_used_by_shared_client(test_client, mock_model, valid_payload):
    """The DummyClassifier fixture really is what the app scores with."""
    expected = float(mock_model.predict_proba([[0.0] * 14])[0][1])
    body = test_client.post("/predict", json=valid_payload).json()
    assert body["default_probability"] == pytest.approx(expected)


# Bundle mode: serving a plain model file with no MLflow available. This is how
# the app runs on serverless hosts, where mlflow is too large to bundle.

def test_bundle_mode_skips_the_registry(monkeypatch, tmp_path):
    """With MODEL_BUNDLE_PATH set, no registry lookup happens."""
    monkeypatch.setenv(api_main.MODEL_BUNDLE_ENV, str(tmp_path / "model.ubj"))
    assert api_main.resolve_model_version() == "bundle"

    monkeypatch.setenv("MODEL_VERSION", "7")
    assert api_main.resolve_model_version() == "7"


def test_bundle_mode_reports_a_missing_file(monkeypatch, tmp_path):
    missing = tmp_path / "nope.ubj"
    monkeypatch.setenv(api_main.MODEL_BUNDLE_ENV, str(missing))
    with pytest.raises(FileNotFoundError, match="Model bundle not found"):
        api_main.load_model()


def test_bundle_round_trip_matches_the_registry_model(tmp_path, monkeypatch):
    """A booster saved to disk and reloaded must score identically."""
    xgb = pytest.importorskip("xgboost")
    import numpy as np
    import pandas as pd

    from api.features import FEATURE_ORDER

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((60, len(FEATURE_ORDER))), columns=FEATURE_ORDER)
    y = (rng.random(60) > 0.5).astype(int)
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3).fit(X, y)

    path = tmp_path / "model.ubj"
    model.get_booster().save_model(str(path))

    monkeypatch.setenv(api_main.MODEL_BUNDLE_ENV, str(path))
    booster = api_main.load_model()

    row = {c: float(v) for c, v in zip(FEATURE_ORDER, X.iloc[0], strict=True)}
    assert api_main.predict_probability(booster, row) == pytest.approx(
        float(model.predict_proba(X.head(1))[0][1]), abs=1e-6
    )


def test_audit_log_falls_back_to_stdout_on_readonly_fs(caplog):
    """A read-only filesystem must not silently lose the audit trail."""
    entry = {"request_id": "abc", "risk_label": "LOW"}
    with caplog.at_level("INFO", logger="credit_risk_api"):
        api_main.write_prediction_log(entry, path="/proc/definitely-not-writable/log.jsonl")
    assert any("PREDICTION_LOG" in r.message for r in caplog.records)
