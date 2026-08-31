"""Tests for scripts/train.py.

MLflow is mocked throughout: these tests must not require a tracking server,
must not write to a real registry, and must run in CI offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    """The real cleaned dataset; skip if the cleaning stage hasn't been run."""
    path = PROJECT_ROOT / "data" / "processed" / "features.csv"
    if not path.exists():
        pytest.skip(f"{path} not found - run scripts/data_cleaning.py first")
    return pd.read_csv(path)


@pytest.fixture
def mock_mlflow():
    """Patch every MLflow surface train.py touches."""
    run = MagicMock()
    run.info.run_id = "test-run-id"

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=run)
    ctx.__exit__ = MagicMock(return_value=False)

    model_info = MagicMock()
    model_info.model_uri = "models:/test-model-id"

    with patch.object(train, "mlflow") as m:
        m.start_run.return_value = ctx
        m.xgboost.log_model.return_value = model_info
        m.lightgbm.log_model.return_value = model_info
        version = MagicMock()
        version.version = "1"
        m.register_model.return_value = version
        yield m


# --------------------------------------------------------------------------
# 1. Models train without error
# --------------------------------------------------------------------------
def test_xgboost_trains_without_error(dataset):
    X_train, X_val, y_train, y_val = train.split_data(dataset)
    model = train.build_xgb()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_val)[:, 1]
    assert len(proba) == len(y_val)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_lightgbm_trains_without_error(dataset):
    X_train, X_val, y_train, y_val = train.split_data(dataset)
    model = train.build_lgbm()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_val)[:, 1]
    assert len(proba) == len(y_val)
    assert ((proba >= 0) & (proba <= 1)).all()


# --------------------------------------------------------------------------
# 2. AUC > 0.7 on the dataset
# --------------------------------------------------------------------------
@pytest.mark.parametrize("builder", ["build_xgb", "build_lgbm"])
def test_auc_above_threshold(dataset, builder):
    X_train, X_val, y_train, y_val = train.split_data(dataset)
    model = getattr(train, builder)()
    model.fit(X_train, y_train)

    metrics = train.evaluate(model, X_val, y_val)
    assert metrics["auc_roc"] > 0.7, f"{builder} AUC {metrics['auc_roc']:.4f} <= 0.7"

    for key in ("auc_roc", "f1_score", "precision", "recall"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


# --------------------------------------------------------------------------
# 3. encoding_maps.json is created
# --------------------------------------------------------------------------
def test_encoding_maps_created(tmp_path):
    out = tmp_path / "encoding_maps.json"
    result = train.export_encoding_maps(out)

    assert result == out
    assert out.exists()

    maps = json.loads(out.read_text())
    assert maps["person_home_ownership"] == {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
    assert maps["loan_grade"] == {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    assert maps["cb_person_default_on_file"] == {"N": 0, "Y": 1}
    assert maps["loan_intent"]["PERSONAL"] == 0
    assert maps["loan_intent"]["DEBTCONSOLIDATION"] == 5
    assert len(maps["feature_order"]) == 14
    assert maps["target_column"] == "loan_status"


def test_encoding_maps_match_cleaning_module(tmp_path):
    """The exported contract must equal the maps cleaning actually applied."""
    import data_cleaning

    maps = json.loads(train.export_encoding_maps(tmp_path / "m.json").read_text())
    assert maps["person_home_ownership"] == data_cleaning.HOME_OWNERSHIP_MAP
    assert maps["loan_intent"] == data_cleaning.LOAN_INTENT_MAP
    assert maps["loan_grade"] == data_cleaning.LOAN_GRADE_MAP
    assert maps["cb_person_default_on_file"] == data_cleaning.DEFAULT_ON_FILE_MAP


def test_encoding_maps_creates_parent_dir(tmp_path):
    out = tmp_path / "nested" / "dir" / "encoding_maps.json"
    train.export_encoding_maps(out)
    assert out.exists()


# --------------------------------------------------------------------------
# Split behaviour
# --------------------------------------------------------------------------
def test_split_is_stratified_and_reproducible(dataset):
    """Pin the split against an independent reference.

    A tolerance-based check is not enough: on 32k rows an unstratified split
    lands within ~1% of the true rate by chance, so a loose assertion passes
    even when stratify= is removed. Compare to a reference split built with
    the exact documented parameters instead.
    """
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train.split_data(dataset)

    X = dataset[train.FEATURE_COLUMNS].astype("float64")
    y = dataset[train.TARGET_COLUMN]
    ref_X_train, ref_X_val, ref_y_train, ref_y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Fails if stratify= is dropped OR random_state changes.
    assert list(X_train.index) == list(ref_X_train.index)
    assert list(X_val.index) == list(ref_X_val.index)

    assert len(X_train) + len(X_val) == len(dataset)
    assert abs(len(X_val) / len(dataset) - 0.2) < 0.01
    # True stratification matches the base rate to rounding, not to 1%.
    assert abs(y_train.mean() - y_val.mean()) < 0.001
    assert list(X_train.columns) == train.FEATURE_COLUMNS


def test_split_constants_are_pinned():
    """These values are part of the reproducibility contract."""
    assert train.RANDOM_STATE == 42
    assert train.TEST_SIZE == 0.2


def test_features_are_float64(dataset):
    """Integer schemas break MLflow enforcement when FastAPI sends JSON floats."""
    X_train, _, _, _ = train.split_data(dataset)
    assert all(str(d) == "float64" for d in X_train.dtypes)


def test_features_exclude_label(dataset):
    assert train.TARGET_COLUMN not in train.FEATURE_COLUMNS
    assert len(train.FEATURE_COLUMNS) == 14


# --------------------------------------------------------------------------
# Hyperparameters match the specified configuration
# --------------------------------------------------------------------------
def test_xgb_hyperparameters():
    p = train.build_xgb().get_params()
    assert p["n_estimators"] == 300
    assert p["max_depth"] == 6
    assert p["learning_rate"] == 0.05
    assert p["subsample"] == 0.8
    assert p["colsample_bytree"] == 0.8
    assert p["scale_pos_weight"] == 3
    assert p["eval_metric"] == "auc"


def test_lgbm_hyperparameters():
    p = train.build_lgbm().get_params()
    assert p["n_estimators"] == 300
    assert p["max_depth"] == 6
    assert p["learning_rate"] == 0.05
    assert p["num_leaves"] == 63
    assert p["is_unbalance"] is True
    assert p["metric"] == "auc"


# --------------------------------------------------------------------------
# Registry: only the winner is registered
# --------------------------------------------------------------------------
def test_only_winner_is_registered(mock_mlflow):
    results = [
        {"name": "xgboost", "metrics": {"auc_roc": 0.94, "f1_score": 0.8,
                                        "precision": 0.8, "recall": 0.8},
         "run_id": "r1", "model_uri": "models:/xgb"},
        {"name": "lightgbm", "metrics": {"auc_roc": 0.91, "f1_score": 0.7,
                                         "precision": 0.7, "recall": 0.7},
         "run_id": "r2", "model_uri": "models:/lgbm"},
    ]
    winner = train.register_best(results)

    assert winner["name"] == "xgboost"
    mock_mlflow.register_model.assert_called_once_with("models:/xgb", "CreditRiskModel")


def test_lightgbm_wins_when_auc_higher(mock_mlflow):
    results = [
        {"name": "xgboost", "metrics": {"auc_roc": 0.88, "f1_score": 0.8,
                                        "precision": 0.8, "recall": 0.8},
         "run_id": "r1", "model_uri": "models:/xgb"},
        {"name": "lightgbm", "metrics": {"auc_roc": 0.93, "f1_score": 0.7,
                                         "precision": 0.7, "recall": 0.7},
         "run_id": "r2", "model_uri": "models:/lgbm"},
    ]
    winner = train.register_best(results)

    assert winner["name"] == "lightgbm"
    mock_mlflow.register_model.assert_called_once_with("models:/lgbm", "CreditRiskModel")


def test_run_training_registers_once(dataset, mock_mlflow, tmp_path, monkeypatch):
    """Full pipeline with MLflow mocked: two runs logged, one registration."""
    data_path = tmp_path / "features.csv"
    dataset.to_csv(data_path, index=False)
    monkeypatch.setenv("PROCESSED_DATA_PATH", str(data_path))

    winner = train.run_training()

    assert winner["name"] in {"xgboost", "lightgbm"}
    assert mock_mlflow.register_model.call_count == 1
    assert mock_mlflow.xgboost.log_model.call_count == 1
    assert mock_mlflow.lightgbm.log_model.call_count == 1
    assert (tmp_path / "encoding_maps.json").exists()  # lands beside the data


def test_no_tracking_server_contacted(mock_mlflow, tmp_path, monkeypatch):
    """Guard against a real MLflow server being hit during tests."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://should-not-be-called:5000")
    train.export_encoding_maps(tmp_path / "m.json")
    mock_mlflow.set_tracking_uri.assert_not_called()


# --------------------------------------------------------------------------
# Env var handling
# --------------------------------------------------------------------------
def test_tracking_uri_default_and_override(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert train.get_tracking_uri() == "http://localhost:5000"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///custom.db")
    assert train.get_tracking_uri() == "sqlite:///custom.db"


def test_data_path_default_and_override(monkeypatch):
    monkeypatch.delenv("PROCESSED_DATA_PATH", raising=False)
    assert train.get_data_path() == Path("data/processed/features.csv")

    monkeypatch.setenv("PROCESSED_DATA_PATH", "/custom/features.csv")
    assert train.get_data_path() == Path("/custom/features.csv")


def test_load_data_missing_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("PROCESSED_DATA_PATH", str(tmp_path / "nope.csv"))
    with pytest.raises(FileNotFoundError, match="Run scripts/data_cleaning.py"):
        train.load_data()
