"""Data drift monitoring for the credit risk scoring model.

Compares the feature distributions the model was trained on against what the
live API has actually been scoring, using Evidently's DataDriftPreset.

Reference : data/processed/features.csv        (sampled for speed)
Current   : data/processed/prediction_log.jsonl (last N days)

Exit code is 0 whenever the check completes - drift is a finding, not a
failure. A non-zero exit means the check itself broke.

Environment:
    PROCESSED_DATA_PATH  (default: data/processed/features.csv)
    PREDICTION_LOG_PATH  (default: data/processed/prediction_log.jsonl)
    DRIFT_REPORT_DIR     (default: monitoring/reports)
    DRIFT_SHARE_THRESHOLD (default: 0.3)
    DRIFT_LOOKBACK_DAYS  (default: 7)
    DRIFT_MIN_ROWS       (default: 100)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger("drift_monitor")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

DEFAULT_REFERENCE_PATH = "data/processed/features.csv"
DEFAULT_PREDICTION_LOG_PATH = "data/processed/prediction_log.jsonl"
DEFAULT_REPORT_DIR = "monitoring/reports"

REFERENCE_SAMPLE_SIZE = 5000
SAMPLE_RANDOM_STATE = 42  # reproducible reference sample across runs
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_DRIFT_SHARE_THRESHOLD = 0.3
DEFAULT_MIN_CURRENT_ROWS = 100

TARGET_COLUMN = "loan_status"

# Machine-readable marker so the Airflow DAG does not have to scrape prose.
RESULT_MARKER = "DRIFT_RESULT_JSON:"


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        logger.warning("Invalid %s; falling back to %s", key, default)
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        logger.warning("Invalid %s; falling back to %s", key, default)
        return default


def get_reference_path() -> Path:
    return Path(os.environ.get("PROCESSED_DATA_PATH", DEFAULT_REFERENCE_PATH))


def get_prediction_log_path() -> Path:
    return Path(os.environ.get("PREDICTION_LOG_PATH", DEFAULT_PREDICTION_LOG_PATH))


def get_report_dir() -> Path:
    return Path(os.environ.get("DRIFT_REPORT_DIR", DEFAULT_REPORT_DIR))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_reference(path: str | Path | None = None, sample_size: int = REFERENCE_SAMPLE_SIZE) -> pd.DataFrame:
    """Load the training features, sampled for speed.

    The label is dropped: the prediction log has no ground truth, so including
    it would compare a column that only exists on one side.
    """
    ref_path = Path(path) if path is not None else get_reference_path()
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference data not found: {ref_path}")

    df = pd.read_csv(ref_path)
    df = df.drop(columns=[TARGET_COLUMN], errors="ignore")

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=SAMPLE_RANDOM_STATE)
        logger.info("Reference: sampled %d of the full dataset from %s", len(df), ref_path)
    else:
        logger.info("Reference: using all %d rows from %s", len(df), ref_path)

    return df.reset_index(drop=True)


def load_current(
    path: str | Path | None = None, lookback_days: int | None = None
) -> pd.DataFrame:
    """Load engineered features from predictions in the last `lookback_days`.

    Each log line carries both `inputs` (raw request) and `features` (the
    14 engineered/encoded values actually fed to the model). We compare
    `features`, because that is what features.csv contains.
    """
    log_path = Path(path) if path is not None else get_prediction_log_path()
    days = lookback_days if lookback_days is not None else _env_int(
        "DRIFT_LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS
    )

    if not log_path.exists():
        logger.warning("Prediction log not found: %s", log_path)
        return pd.DataFrame()

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows: list[dict] = []
    malformed = 0
    stale = 0

    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                ts = datetime.fromisoformat(record["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    stale += 1
                    continue

                features = record.get("features")
                if not features:
                    malformed += 1
                    continue
                rows.append(features)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                # One corrupt line must not sink the whole monitoring run.
                malformed += 1

    if malformed:
        logger.warning("Skipped %d malformed/unusable log lines", malformed)
    logger.info(
        "Current: %d predictions in the last %d days (%d older lines ignored)",
        len(rows), days, stale,
    )
    return pd.DataFrame(rows)


def align_columns(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both frames to their shared columns, in reference order."""
    shared = [c for c in reference.columns if c in current.columns]
    missing = [c for c in reference.columns if c not in current.columns]
    extra = [c for c in current.columns if c not in reference.columns]

    if missing:
        logger.warning("Columns in reference but not in current data: %s", missing)
    if extra:
        logger.warning("Columns in current but not in reference data: %s", extra)
    if not shared:
        raise ValueError("Reference and current data share no columns; cannot compare.")

    return reference[shared].copy(), current[shared].astype("float64").copy()


# --------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------
def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame, drift_share_threshold: float):
    """Run Evidently's DataDriftPreset and return the resulting snapshot."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    logger.info(
        "Running DataDriftPreset on %d columns (reference=%d rows, current=%d rows)",
        reference.shape[1], len(reference), len(current),
    )
    # Pass the threshold through so the HTML report's own verdict matches ours.
    report = Report(metrics=[DataDriftPreset(drift_share=drift_share_threshold)])
    return report.run(current_data=current, reference_data=reference)


def extract_results(snapshot, drift_share_threshold: float) -> dict:
    """Pull per-column drift and the overall share out of a snapshot.

    Evidently emits one ValueDrift metric per column plus a DriftedColumnsCount
    summary. Both methods it selects here (Wasserstein / Jensen-Shannon) are
    distances, so a column is drifted when value >= threshold. The derived
    count is cross-checked against Evidently's own count.
    """
    payload = snapshot.dict()
    per_column: list[dict] = []
    authoritative_count = None
    authoritative_share = None

    for metric in payload.get("metrics", []):
        config = metric.get("config", {})
        mtype = config.get("type", "")
        value = metric.get("value")

        if mtype.endswith("DriftedColumnsCount") and isinstance(value, dict):
            authoritative_count = int(value.get("count", 0))
            authoritative_share = float(value.get("share", 0.0))
            continue

        if mtype.endswith("ValueDrift"):
            threshold = float(config.get("threshold", 0.1))
            score = float(value)
            per_column.append(
                {
                    "column": config.get("column"),
                    "method": config.get("method"),
                    "score": score,
                    "threshold": threshold,
                    "drifted": score >= threshold,
                }
            )

    drifted = [c for c in per_column if c["drifted"]]
    derived_count = len(drifted)

    if authoritative_count is not None and authoritative_count != derived_count:
        # Evidently's own count wins; ours is only used to name the columns.
        logger.warning(
            "Derived drifted count (%d) differs from Evidently's (%d). "
            "A method with inverted threshold semantics may be in play.",
            derived_count, authoritative_count,
        )

    n_columns = len(per_column)
    share = (
        authoritative_share
        if authoritative_share is not None
        else (derived_count / n_columns if n_columns else 0.0)
    )

    return {
        "n_columns": n_columns,
        "n_drifted": authoritative_count if authoritative_count is not None else derived_count,
        "drift_share": share,
        "drift_share_threshold": drift_share_threshold,
        "dataset_drift": share > drift_share_threshold,
        "drifted_columns": [c["column"] for c in drifted],
        "columns": sorted(per_column, key=lambda c: -c["score"]),
    }


def save_report(snapshot, report_dir: Path | None = None, run_date: datetime | None = None) -> Path:
    out_dir = Path(report_dir) if report_dir is not None else get_report_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = (run_date or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    path = out_dir / f"drift_report_{stamp}.html"
    snapshot.save_html(str(path))
    logger.info("Saved HTML report -> %s (%d bytes)", path, path.stat().st_size)
    return path


def print_summary(results: dict) -> None:
    logger.info("=" * 70)
    logger.info("DRIFT SUMMARY")
    logger.info("=" * 70)
    logger.info("%-24s %s", "columns compared", results["n_columns"])
    logger.info("%-24s %s", "drifted columns", results["n_drifted"])
    logger.info("%-24s %.4f", "drift_share", results["drift_share"])
    logger.info("%-24s %.2f", "threshold", results["drift_share_threshold"])
    logger.info("%-24s %s", "dataset_drift", results["dataset_drift"])

    logger.info("-" * 70)
    logger.info("%-28s %-34s %9s", "column", "method", "score")
    for col in results["columns"]:
        flag = "DRIFT" if col["drifted"] else "     "
        logger.info(
            "%-28s %-34s %9.4f  %s", col["column"], col["method"], col["score"], flag
        )
    logger.info("-" * 70)

    if results["drifted_columns"]:
        logger.info("Drifted: %s", ", ".join(results["drifted_columns"]))
    else:
        logger.info("No individual column exceeded its drift threshold.")

    if results["dataset_drift"]:
        logger.warning(
            "DRIFT DETECTED: drift_share %.4f > %.2f -> RETRAINING RECOMMENDED",
            results["drift_share"], results["drift_share_threshold"],
        )
    else:
        logger.info(
            "No dataset-level drift: drift_share %.4f <= %.2f -> no retraining needed",
            results["drift_share"], results["drift_share_threshold"],
        )
    logger.info("=" * 70)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def run_monitoring() -> bool:
    """Full drift check. Returns True when retraining is recommended."""
    threshold = _env_float("DRIFT_SHARE_THRESHOLD", DEFAULT_DRIFT_SHARE_THRESHOLD)
    min_rows = _env_int("DRIFT_MIN_ROWS", DEFAULT_MIN_CURRENT_ROWS)

    logger.info("=" * 70)
    logger.info("CREDIT RISK DRIFT MONITOR")
    logger.info("reference : %s", get_reference_path())
    logger.info("current   : %s", get_prediction_log_path())
    logger.info("threshold : drift_share > %.2f", threshold)
    logger.info("=" * 70)

    reference = load_reference()
    current = load_current()

    if len(current) < min_rows:
        # Too little traffic to say anything. Not an error, and definitely not
        # a reason to trigger a retrain on noise.
        logger.warning(
            "Only %d current rows (minimum %d). Skipping drift check; no retraining triggered.",
            len(current), min_rows,
        )
        result = {
            "status": "insufficient_data",
            "current_rows": len(current),
            "min_rows": min_rows,
            "drift_share": 0.0,
            "dataset_drift": False,
            "drifted_columns": [],
            "retrain_recommended": False,
        }
        _emit(result)
        return False

    reference, current = align_columns(reference, current)
    snapshot = run_drift_report(reference, current, threshold)
    results = extract_results(snapshot, threshold)

    report_path = save_report(snapshot)
    print_summary(results)

    results.update(
        {
            "status": "ok",
            "reference_rows": len(reference),
            "current_rows": len(current),
            "report_path": str(report_path),
            "retrain_recommended": results["dataset_drift"],
        }
    )
    _emit(results)
    return bool(results["dataset_drift"])


def _emit(result: dict) -> None:
    """Write the result JSON and print the marker line for Airflow."""
    out_dir = get_report_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    slim = {k: v for k, v in result.items() if k != "columns"}
    (out_dir / f"drift_result_{stamp}.json").write_text(json.dumps(slim, indent=2, default=str) + "\n")
    print(f"{RESULT_MARKER} {json.dumps(slim, default=str)}")


if __name__ == "__main__":
    try:
        drift = run_monitoring()
        logger.info("retrain_recommended=%s", drift)
        sys.exit(0)  # drift is a finding, not a failure
    except Exception:
        logger.exception("DRIFT MONITORING FAILED")
        sys.exit(1)
