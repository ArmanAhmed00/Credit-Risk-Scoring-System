# Credit Risk Scoring System

Trains a credit-default model, serves it over an API, and provides an
underwriting console on top.

Author: arman

## Getting started

Everything below runs locally. Work through the steps in order — each one
depends on the previous.

### 1. Install dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

### 2. Clean the raw data

Turns `data/credit_risk_dataset.csv` into the 15-column feature set the model
trains on.

```bash
python scripts/data_cleaning.py
```

Writes `data/processed/features.csv` (32,574 rows after 7 bad rows are dropped).

### 3. Start MLflow

Leave this running in its own terminal. Port 5000 is taken by macOS AirPlay, so
use 5001.

```bash
mlflow server --host 127.0.0.1 --port 5001 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

MLflow UI: http://127.0.0.1:5001

### 4. Train the model

Trains XGBoost and LightGBM, then registers whichever wins on AUC-ROC.

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 python scripts/train.py
```

Takes about 30 seconds. Also writes `data/processed/encoding_maps.json`, which
the API needs.

### 5. Promote it to Production

The API serves `models:/CreditRiskModel/Production`, so a newly registered model
has to be promoted before it will load.

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 python - <<'PY'
import os, mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()
latest = max(int(v.version) for v in client.search_model_versions("name='CreditRiskModel'"))
client.transition_model_version_stage(
    "CreditRiskModel", str(latest), "Production", archive_existing_versions=True
)
print(f"promoted version {latest} to Production")
PY
```

### 6. Start the API

Another terminal. Port 8000 is often held by Docker, so use 8001.

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 \
MODEL_URI=models:/CreditRiskModel/Production \
ENCODING_MAPS_PATH=data/processed/encoding_maps.json \
PREDICTION_LOG_PATH=data/processed/prediction_log.jsonl \
uvicorn api.main:app --host 127.0.0.1 --port 8001
```

Check it: http://127.0.0.1:8001/health should return `{"status":"ok", ...}`.
API docs: http://127.0.0.1:8001/docs

### 7. Start the console

Another terminal. Port 8501 may already be in use, so use 8502.

```bash
API_URL=http://127.0.0.1:8001 streamlit run app/streamlit_app.py --server.port 8502
```

Open http://127.0.0.1:8502 and submit an application.

## Score from the command line

```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "person_age": 30, "person_income": 60000, "person_emp_length": 5.0,
    "loan_amnt": 10000, "loan_int_rate": 12.5, "loan_percent_income": 0.17,
    "cb_person_cred_hist_length": 4, "person_home_ownership": "RENT",
    "loan_intent": "PERSONAL", "loan_grade": "C", "cb_person_default_on_file": "N"
  }'
```

```json
{"default_probability": 0.0917, "risk_label": "LOW", "model_version": "1", "request_id": "..."}
```

## Check the drift monitor

Needs at least 100 predictions in the log before it will report anything.

```bash
python monitoring/drift_monitor.py
```

## Run the tests

```bash
pytest
ruff check .
```

## Stop everything

```bash
pkill -f "mlflow server"
pkill -f "uvicorn api.main"
pkill -f "streamlit run"
```

## Notes

- **Ports.** 5000 (AirPlay), 8000 (Docker) and 8501 are commonly occupied on
  this machine, which is why the steps above use 5001, 8001 and 8502.
- **Order matters.** `/predict` returns 503 until a model is promoted in step 5.
- Docker Compose and Vercel deployment also exist — see `docker-compose.yml`,
  `vercel.json` and `scripts/export_model.py`.
