"""Shared fixtures for the credit risk test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


ENCODING_MAPS = {
    "person_home_ownership": {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3},
    "loan_intent": {
        "PERSONAL": 0,
        "EDUCATION": 1,
        "MEDICAL": 2,
        "VENTURE": 3,
        "HOMEIMPROVEMENT": 4,
        "DEBTCONSOLIDATION": 5,
    },
    "loan_grade": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7},
    "cb_person_default_on_file": {"N": 0, "Y": 1},
}

RAW_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

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


@pytest.fixture
def valid_payload() -> dict:
    """A single well-formed /predict request body."""
    return dict(VALID_PAYLOAD)


@pytest.fixture
def encoding_maps() -> dict:
    """Encoding maps in the shape scripts/train.py exports them."""
    from api.features import FEATURE_ORDER

    return {**ENCODING_MAPS, "feature_order": FEATURE_ORDER, "target_column": "loan_status"}


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """20 rows matching the raw CSV schema.

    Deliberately seeded with the exact defects the cleaning stage handles:
    2 null interest rates, 1 null employment length, 1 impossible age and
    1 impossible employment length. Cleaning should yield 18 rows.
    """
    homes = ["RENT", "OWN", "MORTGAGE", "OTHER"]
    intents = [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ]
    grades = ["A", "B", "C", "D", "E", "F", "G"]

    rows = []
    for i in range(20):
        income = 30000 + i * 2500
        amount = 4000 + i * 500
        rows.append(
            {
                "person_age": 22 + i,
                "person_income": income,
                "person_home_ownership": homes[i % len(homes)],
                "person_emp_length": float(i % 12),
                "loan_intent": intents[i % len(intents)],
                "loan_grade": grades[i % len(grades)],
                "loan_amnt": amount,
                "loan_int_rate": round(6.0 + (i % 10) * 1.3, 2),
                "loan_status": i % 2,
                "loan_percent_income": round(amount / income, 2),
                "cb_person_default_on_file": "Y" if i % 3 == 0 else "N",
                "cb_person_cred_hist_length": 2 + (i % 15),
            }
        )

    df = pd.DataFrame(rows, columns=RAW_COLUMNS)

    # Injected defects (indices chosen so the two outliers do not overlap).
    df.loc[15, "loan_int_rate"] = np.nan
    df.loc[16, "loan_int_rate"] = np.nan
    df.loc[17, "person_emp_length"] = np.nan
    df.loc[18, "person_age"] = 144          # impossible age  -> dropped
    df.loc[19, "person_emp_length"] = 123.0  # impossible tenure -> dropped
    return df


@pytest.fixture
def sample_features_df(sample_raw_df) -> pd.DataFrame:
    """The cleaned counterpart of sample_raw_df, via the real pipeline."""
    import data_cleaning

    df = data_cleaning.remove_outliers(sample_raw_df)
    df = data_cleaning.fill_nulls(df)
    df = data_cleaning.encode_categoricals(df)
    df = data_cleaning.engineer_features(df)
    return data_cleaning.validate(df)


@pytest.fixture
def mock_model():
    """A fitted DummyClassifier exposing predict_proba like the real model."""
    from sklearn.dummy import DummyClassifier

    from api.features import FEATURE_ORDER

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((40, len(FEATURE_ORDER))), columns=FEATURE_ORDER)
    y = np.array([0, 1] * 20)

    model = DummyClassifier(strategy="prior")
    model.fit(X, y)
    return model


@pytest.fixture
def test_client(mock_model, encoding_maps, tmp_path, monkeypatch):
    """FastAPI TestClient with the model stubbed out.

    The loaders are patched before the app starts so the lifespan hook never
    reaches a real MLflow server; the suite stays offline.
    """
    from fastapi.testclient import TestClient

    from api import main as api_main

    log_path = tmp_path / "prediction_log.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://offline-test-must-not-connect:5000")

    monkeypatch.setattr(api_main, "load_model", lambda *a, **k: mock_model)
    monkeypatch.setattr(api_main, "load_encoding_maps", lambda *a, **k: encoding_maps)
    monkeypatch.setattr(api_main, "resolve_model_version", lambda *a, **k: "test-1")

    with TestClient(api_main.app) as client:
        client.log_path = log_path
        yield client
