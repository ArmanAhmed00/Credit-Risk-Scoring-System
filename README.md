# Credit Risk Scoring System

End-to-end MLOps pipeline that cleans loan application data, trains and registers a
credit-default model, serves it over an API, and monitors it for drift.

Author: arman

---

## What it does

Raw applications are cleaned into a fixed 15-column feature set, two models
(XGBoost and LightGBM) compete on AUC-ROC, and only the winner is registered in
MLflow. A FastAPI service loads the registered model and scores applications in
real time, logging every prediction. A weekly job compares live traffic against
the training distribution and retrains automatically when it drifts.

```
data/credit_risk_dataset.csv
        │
        ▼  scripts/data_cleaning.py
data/processed/features.csv  ──────────────►  PostgreSQL (processed_features)
        │
        ▼  scripts/train.py
MLflow Model Registry  ── CreditRiskModel/Production
        │
        ▼  api/main.py
POST /predict  ──►  data/processed/prediction_log.jsonl
        │
        ▼  monitoring/drift_monitor.py  (weekly)
   drift > 30%?  ──yes──►  re-trigger the training pipeline
```

## Layout

| Path | Purpose |
|---|---|
| `scripts/data_cleaning.py` | Load, clean, encode, engineer features, validate |
| `scripts/train.py` | Train XGBoost + LightGBM, log to MLflow, register the winner |
| `api/` | FastAPI inference service (`main.py`, `schemas.py`, `features.py`) |
| `monitoring/drift_monitor.py` | Evidently drift report + retrain recommendation |
| `airflow/dags/` | `credit_risk_pipeline` (daily), `drift_monitoring` (Mondays 09:00) |
| `app/streamlit_app.py` | Streamlit UI (scores via the API, not its own model) |
| `tests/` | 121 tests, 86% coverage on `api/` + `scripts/` |
| `docker-compose.yml` | Postgres, MLflow, Airflow, FastAPI |

---

## Running it

### Option A — Docker (full stack)

Everything runs in containers: Postgres, MLflow, Airflow and the API.

```bash
cp .env.example .env      # then edit it - every value is a placeholder
make up                   # builds and starts all four services
```

First boot takes a few minutes: Airflow installs its Python packages and
initialises the metadata database. Watch it with `make logs`.

| Service | URL | Notes |
|---|---|---|
| Airflow | http://localhost:8080 | login from `.env` (default `admin`/`admin`) |
| MLflow | http://localhost:5000 | experiments and model registry |
| API docs | http://localhost:8000/docs | interactive Swagger UI |

Then train a model and score a request:

```bash
make train      # unpauses and triggers the credit_risk_pipeline DAG
make status     # service health
make predict    # sends a sample application to /predict
make drift      # runs the drift check on demand
make down       # stop, keeping data
```

`/predict` returns 503 until a model is registered and promoted to `Production`,
so run `make train` before `make predict`.

### Option B — Local (no Docker)

Useful for development and for running the test suite.

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt

python scripts/data_cleaning.py                # writes data/processed/features.csv
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db &    # registry needs a DB backend
python scripts/train.py                        # trains, registers the winner
uvicorn api.main:app --reload --port 8000      # serves it
```

Promote the model to `Production` once (the API's default `MODEL_URI` expects it):

```bash
python - <<'PY'
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()
latest = max(int(v.version) for v in client.search_model_versions("name='CreditRiskModel'"))
client.transition_model_version_stage(
    "CreditRiskModel", str(latest), "Production", archive_existing_versions=True
)
print(f"promoted version {latest} to Production")
PY
```

### Option C — Vercel (inference API only)

Vercel can host the scoring API, but not the rest of the system: Airflow needs
a long-running scheduler, MLflow needs persistent state, and training exceeds
function limits. Run those elsewhere and deploy only the API.

The function cannot bundle mlflow — it pulls in pyarrow, and together they blow
past Vercel's 250 MB unzipped limit. Instead, export the promoted model to a
plain booster and serve that:

```bash
python scripts/export_model.py     # writes deploy/model.ubj + encoding_maps.json
git add deploy/ && git commit -m "deploy model v1"
vercel deploy
```

`api/index.py` sets `MODEL_BUNDLE_PATH`, so the app loads the file directly and
never imports mlflow. Re-run the export and redeploy whenever you promote a new
model — Vercel serves whatever booster is committed, not whatever is in the
registry.

Two consequences worth knowing:

- **No model registry at runtime.** No stage-based rollback, and the reported
  version comes from `MODEL_VERSION`, not the registry.
- **The filesystem is read-only.** Prediction logs fall back to stdout, which
  Vercel captures. That keeps the audit trail, but it is log retention rather
  than a durable file — send it somewhere persistent before going live.

### Scoring an application

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "person_age": 30, "person_income": 60000, "person_emp_length": 5.0,
    "loan_amnt": 10000, "loan_int_rate": 12.5, "loan_percent_income": 0.17,
    "cb_person_cred_hist_length": 4, "person_home_ownership": "RENT",
    "loan_intent": "PERSONAL", "loan_grade": "C", "cb_person_default_on_file": "N"
  }'
```

```json
{
  "default_probability": 0.0917,
  "risk_label": "LOW",
  "model_version": "1",
  "request_id": "3f2a..."
}
```

Risk bands: `LOW < 0.3`, `MEDIUM 0.3-0.6`, `HIGH > 0.6`.

---

### Underwriting console (Streamlit)

An internal console for loan officers: capture an application, get a decision
recommendation, and review portfolio and model health. It calls the FastAPI
service over HTTP rather than loading the model itself, so there is one scoring
path and every decision made here is written to the same audit log as
production traffic.

```bash
pip install -r app/requirements.txt
API_URL=http://127.0.0.1:8000 streamlit run app/streamlit_app.py
```

| Tab | Purpose |
|---|---|
| New application | Capture applicant, credit file and facility; returns a decision |
| Session queue | Everything assessed this session, exportable as CSV |
| Portfolio upload | Bulk-assess a CSV book, with approve/refer/decline totals |

The console deliberately exposes no model, version or infrastructure details.
Operators see availability and decisions only; model health and drift are
reviewed through MLflow and `monitoring/` instead.

Decision bands map to underwriting actions: **Approve** (< 0.30), **Refer to
underwriter** (0.30-0.60), **Decline** (> 0.60). Every assessment shows a
decision ID, model version and UTC timestamp for audit.

The console is deliberately internal-facing. Showing an applicant their own
default probability is not standard lending practice, and a declined
application triggers adverse-action notice requirements.

## Tests

```bash
pytest                       # 121 tests, coverage gate at 70%
ruff check .
```

CI runs on push and PR to `main` (Python 3.10): ruff, pytest with coverage, and a
separate job that verifies both DAGs import cleanly.

---

## Configuration

Copy `.env.example` to `.env`. **Never commit `.env`** — it is gitignored.

| Variable | Purpose |
|---|---|
| `POSTGRES_USER` / `_PASSWORD` / `_DB` | Airflow metadata DB and feature table |
| `MODEL_URI` | Which model the API serves (default `models:/CreditRiskModel/Production`) |
| `MLFLOW_TRACKING_URI` | Tracking server (default `http://localhost:5000`) |
| `RAW_DATA_PATH` / `PROCESSED_DATA_PATH` | Override pipeline paths |
| `DRIFT_SHARE_THRESHOLD` | Drifted-column share that triggers retraining (0.3) |
| `AIRFLOW_SECRET_KEY` | Signs Airflow session cookies — generate a real one |

Airflow paths are also settable as Airflow Variables (`credit_risk_project_root`,
`credit_risk_raw_path`, …), which docker-compose seeds via `AIRFLOW_VAR_*`.

---

## Known issues

Read these before deploying.

- **Version skew across services.** The Airflow container installs `xgboost` and
  `mlflow` unpinned (resolving to 3.x) while `api/requirements.txt` pins
  `xgboost==2.0.3` and `mlflow==2.13.0`. Measured on the validation set, the same
  model scored under xgboost 2.0.3 vs 3.4.1 assigns a **different risk band to
  3.84% of applications**, always toward less risky. Pin the training container to
  the serving versions before going live.
- **`mlflow==2.13.0` cannot read a registry created by MLflow 3.x** (alembic
  revision error). Keep client and server on the same major version.
- **The three ratio features are collinear.** `loan_percent_income` is
  `loan_amnt / person_income`, and `credit_utilization` is `debt_to_income / 0.3`
  exactly. Harmless for tree models, unstable for linear ones — and it inflates
  the drift share, since one income shift moves 4 of 14 columns.
- **SQLite backs MLflow** in docker-compose. Expect `database is locked` under
  concurrent access; Postgres is already in the stack if you need to move.
- **`prediction_log.jsonl` contains applicant PII** (income, age, employment).
  It is gitignored — keep it that way, and give it a retention policy.
- **165 duplicate rows** exist in the raw dataset. They are logged but not
  removed, since deduplication was not part of the cleaning spec.
