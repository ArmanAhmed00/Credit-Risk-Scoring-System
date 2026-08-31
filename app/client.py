"""HTTP client for the credit risk API.

Kept free of Streamlit imports so it can be tested and reused without a
Streamlit runtime.
"""

from __future__ import annotations

import os

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 15.0

RISK_ACTIONS = {
    "LOW": "Approve / standard terms",
    "MEDIUM": "Manual review recommended",
    "HIGH": "Decline / escalate",
}


def get_api_url() -> str:
    return os.environ.get("API_URL", DEFAULT_API_URL)


def build_payload(
    *,
    age: int,
    income: float,
    emp_length: float,
    loan_amnt: float,
    int_rate: float,
    percent_income: float,
    cred_hist: int,
    home: str,
    intent: str,
    grade: str,
    prior_default: str,
) -> dict:
    """Assemble the exact 11-field body the API expects."""
    return {
        "person_age": int(age),
        "person_income": float(income),
        "person_emp_length": float(emp_length),
        "loan_amnt": float(loan_amnt),
        "loan_int_rate": float(int_rate),
        "loan_percent_income": float(percent_income),
        "cb_person_cred_hist_length": int(cred_hist),
        "person_home_ownership": home,
        "loan_intent": intent,
        "loan_grade": grade,
        "cb_person_default_on_file": prior_default,
    }


def api_health(api_url: str | None = None) -> tuple[bool, dict | str]:
    """Return (ok, payload-or-error). Never raises."""
    url = api_url or get_api_url()
    try:
        r = httpx.get(f"{url}/health", timeout=5.0)
        return r.status_code == 200, r.json()
    except Exception as exc:
        return False, str(exc)


def score_application(payload: dict, api_url: str | None = None) -> tuple[bool, dict | str]:
    """POST one application. Returns (ok, response-or-message), never raises."""
    url = api_url or get_api_url()
    try:
        r = httpx.post(f"{url}/predict", json=payload, timeout=REQUEST_TIMEOUT)
    except Exception as exc:
        return False, f"Could not reach the API at {url}: {exc}"

    if r.status_code == 200:
        return True, r.json()
    if r.status_code == 422:
        detail = r.json().get("detail", [])
        fields = ", ".join(".".join(e.get("loc", [])[1:]) or "?" for e in detail) or "?"
        return False, f"Validation failed (422) on: {fields}"
    if r.status_code == 503:
        return False, "Model not loaded (503). Train and promote a model first."
    return False, f"Unexpected response {r.status_code}: {r.text[:200]}"
