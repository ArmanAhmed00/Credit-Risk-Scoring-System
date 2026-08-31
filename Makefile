# Credit risk scoring system - operational shortcuts.
.DEFAULT_GOAL := help
SHELL := /bin/bash

COMPOSE      := docker compose
API_URL      ?= http://localhost:$(or $(API_PORT),8000)
AIRFLOW_EXEC := $(COMPOSE) exec -T airflow
PIPELINE_DAG := credit_risk_pipeline
DRIFT_DAG    := drift_monitoring

.PHONY: help up down logs train predict clean status drift ui

help:
	@echo "Credit risk scoring system"
	@echo ""
	@echo "  make up       Start all services (builds the API image)"
	@echo "  make down     Stop services, keep volumes and data"
	@echo "  make logs     Tail logs from all services"
	@echo "  make status   Show service state and health"
	@echo "  make train    Trigger the $(PIPELINE_DAG) DAG"
	@echo "  make drift    Trigger the $(DRIFT_DAG) DAG"
	@echo "  make ui       Launch the Streamlit web UI"
	@echo "  make predict  Send a sample scoring request"
	@echo "  make clean    DESTRUCTIVE - remove containers AND volumes"

up:
	@test -f .env || { echo "ERROR: .env missing. Run: cp .env.example .env"; exit 1; }
	$(COMPOSE) up -d --build
	@echo ""
	@echo "Airflow  http://localhost:$${AIRFLOW_PORT:-8080}"
	@echo "MLflow   http://localhost:$${MLFLOW_PORT:-5000}"
	@echo "API      $(API_URL)/docs"
	@echo ""
	@echo "Airflow takes a few minutes on first boot (pip install + db init)."
	@echo "Watch it with: make logs"

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=100

status:
	@$(COMPOSE) ps
	@echo ""
	@curl -fsS $(API_URL)/health && echo "" || echo "API not healthy yet"

train:
	@echo "Unpausing and triggering $(PIPELINE_DAG)..."
	$(AIRFLOW_EXEC) airflow dags unpause $(PIPELINE_DAG)
	$(AIRFLOW_EXEC) airflow dags trigger $(PIPELINE_DAG)
	@echo "Follow progress at http://localhost:$${AIRFLOW_PORT:-8080}"

drift:
	$(AIRFLOW_EXEC) airflow dags unpause $(DRIFT_DAG)
	$(AIRFLOW_EXEC) airflow dags trigger $(DRIFT_DAG)

ui:
	@echo "Starting Streamlit UI (API at $${API_URL:-http://127.0.0.1:8000})"
	API_URL=$${API_URL:-http://127.0.0.1:8000} streamlit run app/streamlit_app.py

predict:
	@curl -sS -X POST $(API_URL)/predict \
	  -H 'Content-Type: application/json' \
	  -d '{ \
	    "person_age": 30, \
	    "person_income": 60000, \
	    "person_emp_length": 5.0, \
	    "loan_amnt": 10000, \
	    "loan_int_rate": 12.5, \
	    "loan_percent_income": 0.17, \
	    "cb_person_cred_hist_length": 4, \
	    "person_home_ownership": "RENT", \
	    "loan_intent": "PERSONAL", \
	    "loan_grade": "C", \
	    "cb_person_default_on_file": "N" \
	  }' | python3 -m json.tool

clean:
	@echo "============================================================"
	@echo "WARNING - THIS PERMANENTLY DELETES:"
	@echo "  * the postgres_data volume (Airflow history + features table)"
	@echo "  * all containers for this stack"
	@echo ""
	@echo "NOT deleted (they are bind mounts on your host):"
	@echo "  * ./mlflow  - the model registry and artifacts"
	@echo "  * ./data    - raw data and prediction logs"
	@echo "Remove those by hand if you really mean to."
	@echo "============================================================"
	@read -p "Type 'yes' to continue: " ans; \
	if [ "$$ans" = "yes" ]; then \
	  $(COMPOSE) down -v --remove-orphans; \
	  echo "Removed containers and volumes."; \
	else \
	  echo "Aborted; nothing was deleted."; \
	fi
