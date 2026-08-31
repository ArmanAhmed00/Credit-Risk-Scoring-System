"""Weekly data drift monitoring for the credit risk model.

    run_drift_check -> evaluate_drift -> trigger_retraining -> credit_risk_pipeline
                                      \\
                                       -> no_drift_detected

Runs every Monday at 09:00. When the drift share exceeds the threshold the
credit_risk_pipeline DAG is triggered to retrain and re-promote.

Airflow Variables (optional; hardcoded fallbacks apply):
    credit_risk_project_root   /opt/airflow/project
    credit_risk_python         sys.executable
    drift_share_threshold      0.3
    drift_lookback_days        7
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.models import Variable

try:
    from airflow.operators.python import BranchPythonOperator, PythonOperator
    from airflow.operators.empty import EmptyOperator
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator
except ImportError:  # pragma: no cover - Airflow 3 layout
    from airflow.providers.standard.operators.python import (
        BranchPythonOperator,
        PythonOperator,
    )
    from airflow.providers.standard.operators.empty import EmptyOperator
    from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

log = logging.getLogger(__name__)

DEFAULT_PROJECT_ROOT = "/opt/airflow/project"
DEFAULT_DRIFT_THRESHOLD = "0.3"
DEFAULT_LOOKBACK_DAYS = "7"

TARGET_DAG_ID = "credit_risk_pipeline"
TASK_TRIGGER = "trigger_retraining"
TASK_NO_DRIFT = "no_drift_detected"

# Emitted by monitoring/drift_monitor.py so we parse structured output rather
# than scraping log prose.
RESULT_MARKER = "DRIFT_RESULT_JSON:"


def _project_root() -> Path:
    return Path(Variable.get("credit_risk_project_root", default_var=DEFAULT_PROJECT_ROOT))


def _python_executable() -> str:
    """Interpreter with evidently/pandas installed (usually the project venv)."""
    return Variable.get("credit_risk_python", default_var=sys.executable)


def run_drift_check(**context) -> dict:
    """Run drift_monitor.py and push its structured result to XCom."""
    root = _project_root()
    script = root / "monitoring" / "drift_monitor.py"
    if not script.exists():
        raise AirflowException(f"Drift monitor not found: {script}")

    env = os.environ.copy()
    env["DRIFT_SHARE_THRESHOLD"] = Variable.get(
        "drift_share_threshold", default_var=DEFAULT_DRIFT_THRESHOLD
    )
    env["DRIFT_LOOKBACK_DAYS"] = Variable.get(
        "drift_lookback_days", default_var=DEFAULT_LOOKBACK_DAYS
    )
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [str(root), env.get("PYTHONPATH", "")] if p
    )

    result = subprocess.run(
        [_python_executable(), str(script)],
        cwd=str(root), env=env, capture_output=True, text=True, check=False,
    )
    if result.stdout:
        log.info("drift_monitor stdout:\n%s", result.stdout)
    if result.stderr:
        log.warning("drift_monitor stderr:\n%s", result.stderr)

    if result.returncode != 0:
        raise AirflowException(
            f"drift_monitor.py exited {result.returncode}; see stderr above."
        )

    payload = None
    for line in result.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            payload = json.loads(line[len(RESULT_MARKER):].strip())
    if payload is None:
        raise AirflowException(
            f"drift_monitor.py produced no '{RESULT_MARKER}' line; cannot decide on retraining."
        )

    log.info(
        "status=%s drift_share=%s drifted=%s retrain=%s",
        payload.get("status"), payload.get("drift_share"),
        payload.get("drifted_columns"), payload.get("retrain_recommended"),
    )
    context["ti"].xcom_push(key="drift_result", value=payload)
    return payload


def evaluate_drift(**context) -> str:
    """Branch to retraining only on a definite drift verdict."""
    payload = context["ti"].xcom_pull(task_ids="run_drift_check", key="drift_result") or {}

    if payload.get("status") == "insufficient_data":
        # Too little traffic to judge: never retrain on noise.
        log.info(
            "Insufficient current data (%s rows); not triggering retraining.",
            payload.get("current_rows"),
        )
        return TASK_NO_DRIFT

    if payload.get("retrain_recommended"):
        log.warning(
            "Drift detected (share=%s, columns=%s) -> triggering %s",
            payload.get("drift_share"), payload.get("drifted_columns"), TARGET_DAG_ID,
        )
        return TASK_TRIGGER

    log.info("No drift (share=%s) -> nothing to do.", payload.get("drift_share"))
    return TASK_NO_DRIFT


default_args = {
    "owner": "credit-risk",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="drift_monitoring",
    description="Weekly Evidently drift check; retrains the model when drift is detected.",
    default_args=default_args,
    schedule="0 9 * * 1",  # every Monday 09:00
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["credit-risk", "monitoring", "mlops"],
) as dag:

    drift_check = PythonOperator(
        task_id="run_drift_check", python_callable=run_drift_check
    )

    branch = BranchPythonOperator(
        task_id="evaluate_drift", python_callable=evaluate_drift
    )

    trigger_retraining = TriggerDagRunOperator(
        task_id=TASK_TRIGGER,
        trigger_dag_id=TARGET_DAG_ID,
        # Marks the triggered run as manual, so credit_risk_pipeline's
        # detect_new_data branch runs the full pipeline instead of comparing
        # file timestamps and skipping.
        conf={
            "triggered_by": "drift_monitoring",
            "reason": "data_drift_detected",
        },
        wait_for_completion=False,
        reset_dag_run=True,
        poke_interval=60,
    )

    no_drift = EmptyOperator(task_id=TASK_NO_DRIFT)

    drift_check >> branch
    branch >> trigger_retraining
    branch >> no_drift
