"""Render tests for the underwriting console.

These run with no API available - the CI case - so the console must degrade
cleanly rather than raise. A Streamlit page that throws on load is invisible
to every other test in this suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
UNREACHABLE = "http://127.0.0.1:9"  # nothing listens here


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("API_URL", UNREACHABLE)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_renders_without_exception(app):
    assert not app.exception, app.exception


def test_has_the_three_operator_tabs(app):
    assert len(app.tabs) == 3


def test_capture_form_is_present(app):
    labels = {n.label for n in app.number_input}
    assert {"Age", "Gross annual income", "Amount requested"} <= labels
    assert {"Application reference"} == {t.label for t in app.text_input}


def test_offline_engine_is_surfaced_not_hidden(app):
    """With the engine down the console must say so, not silently accept input."""
    assert app.error, "no error surfaced while assessments are unavailable"
    assert any("cannot be assessed" in e.value.lower() for e in app.error)


def test_assess_button_disabled_when_engine_offline(app):
    """An underwriter must not be able to request a decision with no engine."""
    assert app.button[0].disabled is True


def test_grade_selector_covers_all_grades(app):
    grades = {s.label: s for s in app.selectbox}["Internal credit grade"]
    assert list(grades.options) == ["A", "B", "C", "D", "E", "F", "G"]


def test_risk_bands_have_distinct_decisions():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
    from styles import BAND

    assert set(BAND) == {"LOW", "MEDIUM", "HIGH"}
    decisions = {b["decision"] for b in BAND.values()}
    assert len(decisions) == 3, "each band must map to a distinct action"
    assert "Adverse action" in BAND["HIGH"]["detail"]


def test_no_internal_details_are_exposed():
    """The console must not surface model, version or infrastructure details."""
    source = Path(APP).read_text()
    rendered = [
        line for line in source.splitlines()
        if not line.strip().startswith("#")
    ]
    body = "\n".join(rendered).lower()
    for term in ("model version", "mlflow", "drift", "shap", "creditriskmodel"):
        assert term not in body, f"internal detail leaked into the UI: {term}"


def test_offline_message_does_not_leak_diagnostics(app):
    """A failure must not print URLs, stack traces or service names."""
    joined = " ".join(e.value for e in app.error)
    assert "http://" not in joined
    assert "Traceback" not in joined
