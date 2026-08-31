FROM python:3.10-slim

# libgomp1 is required by both xgboost and lightgbm at runtime; the slim image
# does not ship it and the failure only surfaces on model load.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/ ./api/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=http://localhost:5000 \
    MODEL_URI=models:/CreditRiskModel/Production \
    ENCODING_MAPS_PATH=/app/data/processed/encoding_maps.json \
    PREDICTION_LOG_PATH=/app/data/processed/prediction_log.jsonl

# encoding_maps.json and the prediction log live here; mount a volume so the
# log survives container restarts.
VOLUME ["/app/data/processed"]

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
