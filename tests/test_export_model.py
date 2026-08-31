"""Tests for scripts/export_model.py.

Covers the artifact-picking logic, which is where a suffix-only match once
selected serving_input_example.json instead of the booster and produced a
1.6 KB bundle that would not load.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import export_model  # noqa: E402


@pytest.fixture
def artifact_dir(tmp_path):
    """Mimic what MLflow actually writes next to a logged model."""
    d = tmp_path / "artifacts"
    d.mkdir()
    (d / "model.ubj").write_bytes(b"x" * 1_000_000)
    (d / "serving_input_example.json").write_bytes(b"y" * 1600)
    (d / "input_example.json").write_bytes(b"z" * 837)
    (d / "MLmodel").write_text("flavors:\n  xgboost: {}\n")
    return d


def test_picks_the_booster_not_the_input_example(artifact_dir):
    picked = export_model.find_model_file(artifact_dir, (".ubj", ".json", ".xgb"))
    assert picked.name == "model.ubj"


def test_refuses_rather_than_returning_a_non_model_file(artifact_dir):
    """serving_input_example.json shares the .json suffix.

    With no model.json present the picker must raise, not fall back to a file
    that merely matches the suffix - that is the bug this guards against.
    """
    with pytest.raises(FileNotFoundError):
        export_model.find_model_file(artifact_dir, (".json",))


def test_prefers_the_largest_matching_candidate(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "model.ubj").write_bytes(b"x" * 10)
    (tmp_path / "b" / "model.ubj").write_bytes(b"x" * 5000)
    assert export_model.find_model_file(tmp_path, (".ubj",)).stat().st_size == 5000


def test_lightgbm_suffixes(tmp_path):
    (tmp_path / "model.txt").write_text("tree")
    assert export_model.find_model_file(tmp_path, (".txt", ".lgb")).name == "model.txt"


def test_missing_model_lists_what_was_found(artifact_dir):
    with pytest.raises(FileNotFoundError) as exc:
        export_model.find_model_file(artifact_dir, (".onnx",))
    assert "MLmodel" in str(exc.value)


def test_export_defaults():
    assert export_model.DEFAULT_MODEL_URI == "models:/CreditRiskModel/Production"
    assert export_model.DEFAULT_EXPORT_DIR == "deploy"


def test_vercel_entrypoint_exposes_app_and_sets_bundle_mode():
    """A broken entry point means a broken deploy, so check it loads.

    Run in a subprocess: importing api/index.py mutates os.environ, which
    would leak into every other test in the session.
    """
    import subprocess

    code = f"""
import sys, os
sys.path.insert(0, {str(PROJECT_ROOT)!r})
from api.index import app
assert app.routes
assert os.environ["MODEL_BUNDLE_PATH"].endswith("deploy/model.ubj")
assert os.environ["ENCODING_MAPS_PATH"].endswith("deploy/encoding_maps.json")
assert os.environ["PREDICTION_LOG_PATH"].startswith("/tmp")
print("ok")
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert "ok" in r.stdout
