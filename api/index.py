"""Vercel entry point.

Vercel looks for `app` in this file. Everything real lives in api/main.py; this
just points the platform at it and makes sure the serverless-safe settings are
on before the app starts.
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# Bundle mode: no registry call, no mlflow import.
os.environ.setdefault("MODEL_BUNDLE_PATH", str(_ROOT / "deploy" / "model.ubj"))
os.environ.setdefault("ENCODING_MAPS_PATH", str(_ROOT / "deploy" / "encoding_maps.json"))

# The filesystem is read-only, so prediction logging falls back to stdout.
os.environ.setdefault("PREDICTION_LOG_PATH", "/tmp/prediction_log.jsonl")

from api.main import app  # noqa: E402

__all__ = ["app"]
