"""FastAPI credit risk scoring service.

Environment:
    MODEL_URI            MLflow model URI (default: models:/CreditRiskModel/Production)
    MLFLOW_TRACKING_URI  MLflow server (default: http://localhost:5000)
    ENCODING_MAPS_PATH   (default: data/processed/encoding_maps.json)
    PREDICTION_LOG_PATH  (default: data/processed/prediction_log.jsonl)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.features import FEATURE_ORDER, FeatureEncodingError, build_feature_row
from api.schemas import HealthResponse, PredictionRequest, PredictionResponse

logger = logging.getLogger("credit_risk_api")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

MODEL_NAME = "CreditRiskModel"
DEFAULT_MODEL_URI = f"models:/{MODEL_NAME}/Production"
DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_ENCODING_MAPS_PATH = "data/processed/encoding_maps.json"
DEFAULT_PREDICTION_LOG_PATH = "data/processed/prediction_log.jsonl"

# Risk bands: LOW < 0.3, MEDIUM 0.3-0.6 inclusive, HIGH > 0.6
LOW_THRESHOLD = 0.3
HIGH_THRESHOLD = 0.6

_log_lock = threading.Lock()


class ModelBundle:
    """Holds everything loaded at startup. Empty until the lifespan hook runs."""

    def __init__(self) -> None:
        self.model: Any = None
        self.model_version: str = "unknown"
        self.encoding_maps: dict[str, Any] | None = None
        self.started_at: float = time.time()

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.encoding_maps is not None

    def uptime_seconds(self) -> float:
        return time.time() - self.started_at


bundle = ModelBundle()


# --------------------------------------------------------------------------
# Startup loading
# --------------------------------------------------------------------------
def get_model_uri() -> str:
    return os.environ.get("MODEL_URI", DEFAULT_MODEL_URI)


def get_encoding_maps_path() -> Path:
    return Path(os.environ.get("ENCODING_MAPS_PATH", DEFAULT_ENCODING_MAPS_PATH))


def get_prediction_log_path() -> Path:
    return Path(os.environ.get("PREDICTION_LOG_PATH", DEFAULT_PREDICTION_LOG_PATH))


def load_encoding_maps(path: str | Path | None = None) -> dict[str, Any]:
    maps_path = Path(path) if path is not None else get_encoding_maps_path()
    if not maps_path.exists():
        raise FileNotFoundError(
            f"Encoding maps not found: {maps_path}. Run scripts/train.py to export them."
        )
    maps = json.loads(maps_path.read_text())

    # The API's feature order must match what the model was trained on.
    trained_order = maps.get("feature_order")
    if trained_order and list(trained_order) != list(FEATURE_ORDER):
        raise ValueError(
            "feature_order in encoding_maps.json does not match api/features.py.\n"
            f"  trained: {trained_order}\n  api:     {FEATURE_ORDER}"
        )
    logger.info("Loaded encoding maps from %s", maps_path)
    return maps


def load_model(model_uri: str | None = None):
    """Load the registered model as a native estimator exposing predict_proba.

    mlflow.pyfunc would return class labels rather than probabilities for these
    flavors, which would silently break the risk bands, so the concrete flavor
    is resolved from the MLmodel metadata instead.
    """
    uri = model_uri or get_model_uri()
    logger.info("Loading model from %s", uri)

    from mlflow.models import Model

    flavors = set(Model.load(uri).flavors)
    if "xgboost" in flavors:
        model = mlflow.xgboost.load_model(uri)
    elif "lightgbm" in flavors:
        model = mlflow.lightgbm.load_model(uri)
    else:
        logger.warning("Unrecognized flavors %s; falling back to pyfunc", sorted(flavors))
        model = mlflow.pyfunc.load_model(uri)

    logger.info("Model loaded (flavors: %s)", sorted(flavors))
    return model


def resolve_model_version(model_uri: str | None = None) -> str:
    """Best-effort registry lookup of the concrete version behind the URI."""
    uri = model_uri or get_model_uri()
    try:
        from mlflow.tracking import MlflowClient

        if not uri.startswith("models:/"):
            return "unknown"
        ref = uri.removeprefix("models:/")
        name, _, qualifier = ref.partition("/")

        client = MlflowClient()
        if qualifier.isdigit():
            return qualifier
        if qualifier:
            versions = client.get_latest_versions(name, stages=[qualifier])
            if versions:
                return str(versions[0].version)
        versions = client.search_model_versions(f"name='{name}'")
        if versions:
            return str(max(int(v.version) for v in versions))
    except Exception:
        logger.warning("Could not resolve model version for %s", uri, exc_info=True)
    return "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle.started_at = time.time()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))

    try:
        bundle.encoding_maps = load_encoding_maps()
    except Exception:
        # Start anyway; /health reports 503 and /predict refuses. Crashing the
        # container on a transient MLflow outage would take the service down
        # harder than serving honest 503s.
        logger.exception("Failed to load encoding maps")

    try:
        bundle.model = load_model()
        bundle.model_version = resolve_model_version()
        logger.info("Startup complete: %s version %s", MODEL_NAME, bundle.model_version)
    except Exception:
        logger.exception("Failed to load model at startup; service will report 503")

    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Credit Risk Scoring API",
    version="1.0.0",
    description="Scores credit applications with the registered CreditRiskModel.",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Middleware
# --------------------------------------------------------------------------
@app.middleware("http")
async def log_response_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info(
        "%s %s -> %d in %.2f ms", request.method, request.url.path, response.status_code, elapsed_ms
    )
    return response


# --------------------------------------------------------------------------
# Error handlers
# --------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """422 for bad or missing input."""
    errors = jsonable_errors(exc.errors())
    # Log which fields failed, never their values: the payload carries income,
    # age and employment history, and application logs are not a PII store.
    logger.warning(
        "422 validation error on %s: %s",
        request.url.path,
        [{"loc": e["loc"], "type": e.get("type")} for e in errors],
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": errors},
    )


def jsonable_errors(errors: list[dict]) -> list[dict]:
    """Keep only serializable, non-sensitive fields from Pydantic errors.

    Drops "ctx" (may hold non-JSON objects) and "input" (echoes the submitted
    applicant data straight back into responses and downstream log sinks).
    """
    clean = []
    for err in errors:
        e = {k: v for k, v in err.items() if k not in ("ctx", "input")}
        e["loc"] = [str(p) for p in err.get("loc", [])]
        clean.append(e)
    return clean


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """500 for anything unexpected; never leak internals to the caller."""
    request_id = str(uuid.uuid4())
    logger.exception("500 unhandled error on %s (request_id=%s)", request.url.path, request_id)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "request_id": request_id},
    )


# --------------------------------------------------------------------------
# Prediction logging
# --------------------------------------------------------------------------
def write_prediction_log(entry: dict[str, Any], path: str | Path | None = None) -> None:
    """Append one JSON line. Runs in a background task, off the response path."""
    log_path = Path(path) if path is not None else get_prediction_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, default=str)
        with _log_lock:  # serialize appends across worker threads
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        # An audit-log failure must never fail the prediction itself.
        logger.exception("Failed to write prediction log")


def classify_risk(probability: float) -> str:
    """LOW < 0.3 <= MEDIUM <= 0.6 < HIGH."""
    if probability < LOW_THRESHOLD:
        return "LOW"
    if probability <= HIGH_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


def predict_probability(model, features: dict[str, float]) -> float:
    """Positive-class probability for one feature row."""
    frame = pd.DataFrame([features], columns=FEATURE_ORDER).astype("float64")

    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(frame)[0][1])

    # pyfunc fallback: may return a probability or a bare label.
    raw = model.predict(frame)
    value = float(getattr(raw, "iloc", raw)[0] if hasattr(raw, "iloc") else raw[0])
    return value


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    payload = HealthResponse(
        status="ok" if bundle.is_ready else "degraded",
        model_name=MODEL_NAME,
        model_version=bundle.model_version,
        uptime_seconds=round(bundle.uptime_seconds(), 3),
    )
    if not bundle.is_ready:
        # Non-200 so orchestrators pull the pod out of rotation.
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload.model_dump()
        )
    return payload


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    if not bundle.is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": f"Model not loaded. Check MODEL_URI ({get_model_uri()})."},
        )

    request_id = str(uuid.uuid4())
    payload = request.model_dump()

    try:
        features = build_feature_row(payload, bundle.encoding_maps)
    except FeatureEncodingError as exc:
        # A value the schema allowed but the encoding maps don't know.
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc), "request_id": request_id},
        )

    probability = predict_probability(bundle.model, features)
    risk_label = classify_risk(probability)

    background_tasks.add_task(
        write_prediction_log,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "inputs": payload,
            "features": features,
            "default_probability": probability,
            "risk_label": risk_label,
            "model_version": bundle.model_version,
        },
    )

    return PredictionResponse(
        default_probability=probability,
        risk_label=risk_label,
        model_version=bundle.model_version,
        request_id=request_id,
    )
