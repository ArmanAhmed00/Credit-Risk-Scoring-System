"""Credit risk scoring pipeline.

    file_sensor -> detect_new_data -> clean_data -> save_to_postgres
                                   \\                    -> train_model
                                    -> skip              -> promote_to_production

Airflow Variables (all optional; hardcoded fallbacks apply):
    credit_risk_project_root   /opt/airflow/project
    credit_risk_raw_path       <root>/data/credit_risk_dataset.csv
    credit_risk_processed_path <root>/data/processed/features.csv
    credit_risk_python         sys.executable
    credit_risk_pg_table       processed_features
    credit_risk_model_name     CreditRiskModel
    credit_risk_experiment     credit-risk-scoring

Environment (secrets belong here, not in Variables):
    POSTGRES_HOST / POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB / POSTGRES_PORT
    MLFLOW_TRACKING_URI
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.models import Variable

# Import paths moved to the standard provider package in Airflow 3.
try:
    from airflow.operators.python import BranchPythonOperator, PythonOperator
    from airflow.operators.empty import EmptyOperator
    from airflow.sensors.filesystem import FileSensor
except ImportError:  # pragma: no cover - Airflow 3 layout
    from airflow.providers.standard.operators.python import (
        BranchPythonOperator,
        PythonOperator,
    )
    from airflow.providers.standard.operators.empty import EmptyOperator
    from airflow.providers.standard.sensors.filesystem import FileSensor

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Defaults. Every one is overridable by the matching Airflow Variable.
# --------------------------------------------------------------------------
DEFAULT_PROJECT_ROOT = "/opt/airflow/project"
DEFAULT_RAW_RELPATH = "data/credit_risk_dataset.csv"
DEFAULT_PROCESSED_RELPATH = "data/processed/features.csv"
DEFAULT_PG_TABLE = "processed_features"
DEFAULT_MODEL_NAME = "CreditRiskModel"
DEFAULT_EXPERIMENT = "credit-risk-scoring"

TASK_CLEAN = "clean_data"
TASK_SKIP = "skip"

# Jinja, so the Variable is read at task render time rather than on every DAG
# parse. Variable.get() at module scope hits the metadata DB every ~30s.
RAW_PATH_TEMPLATE = (
    "{{ var.value.get('credit_risk_raw_path', "
    "var.value.get('credit_risk_project_root', '" + DEFAULT_PROJECT_ROOT + "') "
    "~ '/" + DEFAULT_RAW_RELPATH + "') }}"
)


def _project_root() -> Path:
    return Path(Variable.get("credit_risk_project_root", default_var=DEFAULT_PROJECT_ROOT))


def _raw_path() -> Path:
    return Path(
        Variable.get(
            "credit_risk_raw_path", default_var=str(_project_root() / DEFAULT_RAW_RELPATH)
        )
    )


def _processed_path() -> Path:
    return Path(
        Variable.get(
            "credit_risk_processed_path",
            default_var=str(_project_root() / DEFAULT_PROCESSED_RELPATH),
        )
    )


def _python_executable() -> str:
    """Interpreter that runs the scripts.

    The Airflow venv usually does not carry pandas/xgboost/mlflow, so this
    normally points at the project venv rather than Airflow's own.
    """
    return Variable.get("credit_risk_python", default_var=sys.executable)


def _run_script(script_relpath: str, extra_env: dict[str, str], task_name: str) -> str:
    """Run a project script in a subprocess and stream its output to the task log."""
    root = _project_root()
    script = root / script_relpath
    if not script.exists():
        raise AirflowException(f"[{task_name}] script not found: {script}")

    env = os.environ.copy()
    env.update(extra_env)
    # so `import data_cleaning` works from inside train.py
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [str(root), str(root / "scripts"), env.get("PYTHONPATH", "")] if p
    )

    cmd = [_python_executable(), str(script)]
    log.info("[%s] exec: %s", task_name, " ".join(cmd))
    for k, v in extra_env.items():
        log.info("[%s] env: %s=%s", task_name, k, v)

    result = subprocess.run(
        cmd, cwd=str(root), env=env, capture_output=True, text=True, check=False
    )

    if result.stdout:
        log.info("[%s] stdout:\n%s", task_name, result.stdout)
    if result.stderr:
        log.warning("[%s] stderr:\n%s", task_name, result.stderr)

    if result.returncode != 0:
        raise AirflowException(
            f"[{task_name}] {script.name} exited {result.returncode}. See stderr above."
        )
    return result.stdout


# --------------------------------------------------------------------------
# detect_new_data
# --------------------------------------------------------------------------
def detect_new_data(**context) -> str:
    """Run the pipeline only when raw data is newer than the processed output.

    A manual trigger always runs the full pipeline, so an operator can force a
    rebuild without touching the source file.
    """
    dag_run = context.get("dag_run")
    run_type = getattr(dag_run, "run_type", None)
    log.info("run_type=%s", run_type)

    if str(run_type) == "manual":
        log.info("Manual trigger -> running full pipeline regardless of timestamps.")
        return TASK_CLEAN

    raw, processed = _raw_path(), _processed_path()

    if not raw.exists():
        raise AirflowException(f"Raw data missing at branch time: {raw}")

    if not processed.exists():
        log.info("No processed output at %s yet -> running full pipeline.", processed)
        return TASK_CLEAN

    raw_mtime = raw.stat().st_mtime
    processed_mtime = processed.stat().st_mtime
    log.info(
        "raw mtime=%s | processed mtime=%s",
        datetime.fromtimestamp(raw_mtime).isoformat(),
        datetime.fromtimestamp(processed_mtime).isoformat(),
    )

    if raw_mtime > processed_mtime:
        log.info("Raw data is newer -> running full pipeline.")
        return TASK_CLEAN

    log.info("Processed output is up to date -> skipping.")
    return TASK_SKIP


# --------------------------------------------------------------------------
# clean_data
# --------------------------------------------------------------------------
def clean_data(**context) -> str:
    processed = _processed_path()
    _run_script(
        "scripts/data_cleaning.py",
        {"RAW_DATA_PATH": str(_raw_path()), "PROCESSED_DATA_PATH": str(processed)},
        "clean_data",
    )
    if not processed.exists():
        raise AirflowException(f"Cleaning reported success but {processed} was not written.")

    log.info("clean_data wrote %s (%d bytes)", processed, processed.stat().st_size)
    return str(processed)


# --------------------------------------------------------------------------
# save_to_postgres
# --------------------------------------------------------------------------
def save_to_postgres(**context) -> int:
    import pandas as pd
    from sqlalchemy import create_engine

    processed = _processed_path()
    if not processed.exists():
        raise AirflowException(f"Cannot load to Postgres; {processed} does not exist.")

    missing = [
        k for k in ("POSTGRES_HOST", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")
        if not os.environ.get(k)
    ]
    if missing:
        raise AirflowException(f"Missing Postgres env vars: {missing}")

    from urllib.parse import quote_plus

    user = quote_plus(os.environ["POSTGRES_USER"])
    password = quote_plus(os.environ["POSTGRES_PASSWORD"])  # never logged
    host = os.environ["POSTGRES_HOST"]
    port = os.environ.get("POSTGRES_PORT", "5432")
    database = os.environ["POSTGRES_DB"]
    table = Variable.get("credit_risk_pg_table", default_var=DEFAULT_PG_TABLE)

    df = pd.read_csv(processed)
    log.info("Loaded %d rows x %d cols from %s", len(df), df.shape[1], processed)
    log.info("Writing to postgresql://%s@%s:%s/%s table=%s", user, host, port, database, table)

    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")
    try:
        df.to_sql(table, engine, if_exists="replace", index=False,
                  chunksize=5000, method="multi")
    finally:
        engine.dispose()

    log.info("Wrote %d rows to '%s' (if_exists=replace)", len(df), table)
    context["ti"].xcom_push(key="rows_written", value=len(df))
    return len(df)


# --------------------------------------------------------------------------
# train_model
# --------------------------------------------------------------------------
RUN_ID_PATTERN = re.compile(r"from run ([0-9a-f]{32})")


def train_model(**context) -> str:
    stdout = _run_script(
        "scripts/train.py", {"PROCESSED_DATA_PATH": str(_processed_path())}, "train_model"
    )

    run_id = None
    matches = RUN_ID_PATTERN.findall(stdout)
    if matches:
        run_id = matches[-1]
        log.info("Parsed mlflow_run_id from train.py output: %s", run_id)
    else:
        # Fallback: ask MLflow for the newest run in the experiment.
        log.warning("No run id in stdout; querying MLflow for the latest run.")
        try:
            import mlflow

            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
            experiment = mlflow.get_experiment_by_name(
                Variable.get("credit_risk_experiment", default_var=DEFAULT_EXPERIMENT)
            )
            if experiment:
                runs = mlflow.search_runs(
                    [experiment.experiment_id], order_by=["start_time DESC"], max_results=1
                )
                if len(runs):
                    run_id = runs.iloc[0]["run_id"]
                    log.info("Resolved mlflow_run_id from MLflow: %s", run_id)
        except Exception:
            log.exception("Could not resolve mlflow_run_id from MLflow")

    if not run_id:
        raise AirflowException("train.py succeeded but no MLflow run id could be determined.")

    context["ti"].xcom_push(key="mlflow_run_id", value=run_id)
    return run_id


# --------------------------------------------------------------------------
# promote_to_production
# --------------------------------------------------------------------------
def promote_to_production(**context) -> dict:
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    model_name = Variable.get("credit_risk_model_name", default_var=DEFAULT_MODEL_NAME)

    run_id = context["ti"].xcom_pull(task_ids="train_model", key="mlflow_run_id")
    log.info("Promoting for run_id=%s on %s", run_id, tracking_uri)

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise AirflowException(f"No registered versions found for '{model_name}'.")

    # Prefer the version produced by this run; fall back to the newest overall.
    matching = [v for v in versions if run_id and v.run_id == run_id]
    target = (
        max(matching, key=lambda v: int(v.version))
        if matching
        else max(versions, key=lambda v: int(v.version))
    )
    if not matching:
        log.warning(
            "No version matched run_id=%s; promoting newest version %s instead.",
            run_id, target.version,
        )

    # MLflow returns version as int on some backends and str on others;
    # normalize before any string handling.
    previous = [
        str(v.version)
        for v in versions
        if getattr(v, "current_stage", None) == "Production"
        and str(v.version) != str(target.version)
    ]

    # archive_existing_versions handles the demotion of whatever held Production.
    client.transition_model_version_stage(
        name=model_name,
        version=target.version,
        stage="Production",
        archive_existing_versions=True,
    )

    log.info("Promoted %s v%s -> Production", model_name, target.version)
    if previous:
        log.info("Archived previous Production version(s): %s", ", ".join(previous))
    else:
        log.info("No previous Production version to archive.")

    result = {
        "model_name": model_name,
        "promoted_version": str(target.version),
        "archived_versions": previous,
        "run_id": run_id,
    }
    context["ti"].xcom_push(key="promotion_result", value=result)
    return result


# --------------------------------------------------------------------------
# DAG
# --------------------------------------------------------------------------
default_args = {
    "owner": "credit-risk",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="credit_risk_pipeline",
    description="Clean, persist, train and promote the credit risk scoring model.",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["credit-risk", "mlops"],
) as dag:

    wait_for_raw_data = FileSensor(
        task_id="wait_for_raw_data",
        filepath=RAW_PATH_TEMPLATE,
        fs_conn_id="fs_default",
        poke_interval=60,
        timeout=600,
        mode="reschedule",  # frees the worker slot between pokes
    )

    branch = BranchPythonOperator(
        task_id="detect_new_data",
        python_callable=detect_new_data,
    )

    clean = PythonOperator(task_id=TASK_CLEAN, python_callable=clean_data)

    to_postgres = PythonOperator(
        task_id="save_to_postgres", python_callable=save_to_postgres
    )

    train = PythonOperator(task_id="train_model", python_callable=train_model)

    promote = PythonOperator(
        task_id="promote_to_production", python_callable=promote_to_production
    )

    nothing_to_do = EmptyOperator(task_id=TASK_SKIP)

    wait_for_raw_data >> branch
    branch >> clean >> to_postgres >> train >> promote
    branch >> nothing_to_do
