"""Train and compare XGBoost vs LightGBM for the credit risk scoring system.

Both models are logged to MLflow; only the higher AUC-ROC model is registered
as "CreditRiskModel" in the MLflow Model Registry.

Environment:
    MLFLOW_TRACKING_URI  (default: http://localhost:5000)
    PROCESSED_DATA_PATH  (default: data/processed/features.csv)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: Airflow workers have no display

import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Single source of truth for the encoding contract: these maps are defined in
# the cleaning module and re-exported here, never redefined.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_cleaning import (  # noqa: E402
    DEFAULT_ON_FILE_MAP,
    HOME_OWNERSHIP_MAP,
    LOAN_GRADE_MAP,
    LOAN_INTENT_MAP,
    OUTPUT_COLUMNS,
)

logger = logging.getLogger("train")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

EXPERIMENT_NAME = "credit-risk-scoring"
REGISTERED_MODEL_NAME = "CreditRiskModel"
TARGET_COLUMN = "loan_status"
FEATURE_COLUMNS = [c for c in OUTPUT_COLUMNS if c != TARGET_COLUMN]

RANDOM_STATE = 42
TEST_SIZE = 0.2
SHAP_SAMPLE_SIZE = 500
CLASSIFICATION_THRESHOLD = 0.5

DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_DATA_PATH = "data/processed/features.csv"
ENCODING_MAPS_FILENAME = "encoding_maps.json"

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3,
    "eval_metric": "auc",
    "random_state": RANDOM_STATE,
}

LGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "is_unbalance": True,
    "metric": "auc",
    "random_state": RANDOM_STATE,
}


def get_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def get_data_path() -> Path:
    return Path(os.environ.get("PROCESSED_DATA_PATH", DEFAULT_DATA_PATH))


def export_encoding_maps(path: str | Path | None = None) -> Path:
    """Write the encoding maps the FastAPI inference server must reuse.

    These are the exact maps applied at cleaning time. The server encodes
    incoming requests with them, so training and serving stay aligned.
    """
    out = Path(path) if path is not None else get_data_path().parent / ENCODING_MAPS_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "person_home_ownership": HOME_OWNERSHIP_MAP,
        "loan_intent": LOAN_INTENT_MAP,
        "loan_grade": LOAN_GRADE_MAP,
        "cb_person_default_on_file": DEFAULT_ON_FILE_MAP,
        "feature_order": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("Exported encoding maps -> %s", out)
    return out


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    data_path = Path(path) if path is not None else get_data_path()
    logger.info("Loading training data from %s", data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}. Run scripts/data_cleaning.py first."
        )

    df = pd.read_csv(data_path)
    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing expected columns: {missing}")

    logger.info("Loaded %d rows x %d columns", len(df), df.shape[1])
    return df


def split_data(df: pd.DataFrame):
    """Stratified 80/20 split on the label."""
    # Cast to float64 so the logged model signature accepts JSON floats from
    # the inference server. Integer schemas fail enforcement at serving time.
    X = df[FEATURE_COLUMNS].astype("float64")
    y = df[TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(
        "Split -> train %d rows (%.2f%% positive) | val %d rows (%.2f%% positive)",
        len(X_train),
        100 * y_train.mean(),
        len(X_val),
        100 * y_val.mean(),
    )
    return X_train, X_val, y_train, y_val


def build_xgb() -> XGBClassifier:
    return XGBClassifier(**XGB_PARAMS)


def build_lgbm() -> LGBMClassifier:
    return LGBMClassifier(**LGBM_PARAMS, verbose=-1)


def evaluate(model, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, float]:
    """Compute the four tracked metrics on the held-out validation set."""
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= CLASSIFICATION_THRESHOLD).astype(int)

    return {
        "auc_roc": float(roc_auc_score(y_val, proba)),
        "f1_score": float(f1_score(y_val, preds, zero_division=0)),
        "precision": float(precision_score(y_val, preds, zero_division=0)),
        "recall": float(recall_score(y_val, preds, zero_division=0)),
    }


def log_shap_summary(model, X_val: pd.DataFrame, model_name: str) -> None:
    """Log a SHAP summary plot built from the first SHAP_SAMPLE_SIZE val rows."""
    sample = X_val.iloc[:SHAP_SAMPLE_SIZE]
    logger.info("[%s] Computing SHAP values on %d validation rows", model_name, len(sample))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Binary LightGBM can return a per-class list; take the positive class.
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    # Or a 3D array (n, features, classes).
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.title(f"SHAP summary - {model_name}")

    with tempfile.TemporaryDirectory() as tmp:
        plot_path = Path(tmp) / f"shap_summary_{model_name}.png"
        plt.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close("all")
        mlflow.log_artifact(str(plot_path), artifact_path="shap")

    logger.info("[%s] Logged SHAP summary plot", model_name)


def train_and_log(
    model_name: str,
    model,
    params: dict,
    X_train,
    y_train,
    X_val,
    y_val,
    flavor,
) -> dict:
    """Train one model inside its own MLflow run and log everything."""
    with mlflow.start_run(run_name=model_name) as run:
        logger.info("=" * 70)
        logger.info("[%s] Training on %d rows", model_name, len(X_train))

        model.fit(X_train, y_train)
        metrics = evaluate(model, X_val, y_val)

        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))
        mlflow.log_param("classification_threshold", CLASSIFICATION_THRESHOLD)
        mlflow.log_metrics(metrics)

        model_info = flavor.log_model(model, name="model", input_example=X_val.head(5))

        for k, v in metrics.items():
            logger.info("[%s] %-12s %.4f", model_name, k, v)

        try:
            log_shap_summary(model, X_val, model_name)
        except Exception:
            # A plotting failure must not discard a trained model.
            logger.exception("[%s] SHAP plot failed; continuing", model_name)

        return {
            "name": model_name,
            "model": model,
            "metrics": metrics,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
        }


def register_best(results: list[dict]) -> dict:
    """Register only the highest-AUC model in the MLflow Model Registry."""
    winner = max(results, key=lambda r: r["metrics"]["auc_roc"])
    loser = min(results, key=lambda r: r["metrics"]["auc_roc"])
    margin = winner["metrics"]["auc_roc"] - loser["metrics"]["auc_roc"]

    logger.info("=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)
    header = f"{'model':<12}{'auc_roc':>10}{'f1_score':>11}{'precision':>11}{'recall':>10}"
    logger.info(header)
    for r in sorted(results, key=lambda r: -r["metrics"]["auc_roc"]):
        m = r["metrics"]
        logger.info(
            "%-12s%10.4f%11.4f%11.4f%10.4f",
            r["name"], m["auc_roc"], m["f1_score"], m["precision"], m["recall"],
        )

    logger.info("-" * 70)
    logger.info("WINNER: %s (AUC-ROC %.4f, +%.4f over %s)",
                winner["name"], winner["metrics"]["auc_roc"], margin, loser["name"])

    version = mlflow.register_model(winner["model_uri"], REGISTERED_MODEL_NAME)
    logger.info("Registered '%s' version %s from run %s",
                REGISTERED_MODEL_NAME, version.version, winner["run_id"])
    logger.info("Not registered: %s (lower AUC-ROC)", loser["name"])
    logger.info("=" * 70)

    winner["registered_version"] = version.version
    return winner


def run_training(data_path: str | Path | None = None) -> dict:
    """Full training pipeline: train both models, compare, register the winner."""
    tracking_uri = get_tracking_uri()
    logger.info("=" * 70)
    logger.info("CREDIT RISK MODEL TRAINING")
    logger.info("MLFLOW_TRACKING_URI = %s", tracking_uri)
    logger.info("Experiment          = %s", EXPERIMENT_NAME)
    logger.info("=" * 70)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    export_encoding_maps()

    df = load_data(data_path)
    X_train, X_val, y_train, y_val = split_data(df)

    results = [
        train_and_log("xgboost", build_xgb(), XGB_PARAMS,
                      X_train, y_train, X_val, y_val, mlflow.xgboost),
        train_and_log("lightgbm", build_lgbm(), LGBM_PARAMS,
                      X_train, y_train, X_val, y_val, mlflow.lightgbm),
    ]

    return register_best(results)


if __name__ == "__main__":
    try:
        run_training()
    except Exception:
        logger.exception("TRAINING FAILED")
        sys.exit(1)
