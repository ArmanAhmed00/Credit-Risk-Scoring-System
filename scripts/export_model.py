"""Export the registered model to a self-contained bundle.

Serverless hosts can't reach an MLflow registry, and mlflow itself pulls in
pyarrow, which is far too big for a function bundle. This writes the raw
booster plus the encoding maps so the API can serve without mlflow installed.

    python scripts/export_model.py

Environment:
    MLFLOW_TRACKING_URI  (default: http://localhost:5000)
    MODEL_URI            (default: models:/CreditRiskModel/Production)
    EXPORT_DIR           (default: deploy)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger("export_model")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

DEFAULT_MODEL_URI = "models:/CreditRiskModel/Production"
DEFAULT_EXPORT_DIR = "deploy"
DEFAULT_MAPS_PATH = "data/processed/encoding_maps.json"


def find_model_file(root: Path, suffixes: tuple[str, ...]) -> Path:
    """Pick the booster out of a downloaded artifact directory.

    Match on the stem too, not just the suffix. MLflow drops
    serving_input_example.json next to the model, and a bare suffix match
    happily returns that instead of the booster.
    """
    candidates = [p for p in root.rglob("*") if p.stem == "model" and p.suffix in suffixes]
    if not candidates:
        found = sorted(p.name for p in root.rglob("*") if p.is_file())
        raise FileNotFoundError(
            f"No model file with suffix {suffixes} under {root}. Found: {found}"
        )
    return max(candidates, key=lambda p: p.stat().st_size)


def export(model_uri: str | None = None, export_dir: str | Path | None = None) -> Path:
    import mlflow
    from mlflow.models import Model
    from mlflow.tracking import MlflowClient

    uri = model_uri or os.environ.get("MODEL_URI", DEFAULT_MODEL_URI)
    out = Path(export_dir or os.environ.get("EXPORT_DIR", DEFAULT_EXPORT_DIR))
    out.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    logger.info("Exporting %s", uri)

    flavors = set(Model.load(uri).flavors)
    local = Path(mlflow.artifacts.download_artifacts(uri))

    if "xgboost" in flavors:
        src = find_model_file(local, (".ubj", ".json", ".xgb"))
        model_name = "model.ubj"
    elif "lightgbm" in flavors:
        src = find_model_file(local, (".txt", ".lgb"))
        model_name = "model.txt"
    else:
        raise SystemExit(f"Unsupported flavors for a plain export: {sorted(flavors)}")

    shutil.copy(src, out / model_name)
    logger.info("Wrote %s (%d bytes)", out / model_name, (out / model_name).stat().st_size)

    maps_src = Path(os.environ.get("ENCODING_MAPS_PATH", DEFAULT_MAPS_PATH))
    if maps_src.exists():
        shutil.copy(maps_src, out / "encoding_maps.json")
        logger.info("Wrote %s", out / "encoding_maps.json")
    else:
        logger.warning("No encoding maps at %s; the API will not start without them", maps_src)

    version = "unknown"
    try:
        ref = uri.removeprefix("models:/")
        name, _, qualifier = ref.partition("/")
        client = MlflowClient()
        if qualifier.isdigit():
            version = qualifier
        elif qualifier:
            found = client.get_latest_versions(name, stages=[qualifier])
            version = str(found[0].version) if found else "unknown"
    except Exception:
        logger.warning("Could not resolve the registry version", exc_info=True)

    (out / "model_meta.json").write_text(
        json.dumps(
            {"source_uri": uri, "model_version": version, "flavors": sorted(flavors),
             "model_file": model_name},
            indent=2,
        )
        + "\n"
    )
    logger.info("Exported version %s -> %s", version, out)
    return out


if __name__ == "__main__":
    try:
        export()
    except Exception:
        logger.exception("EXPORT FAILED")
        sys.exit(1)
