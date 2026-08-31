"""Underwriting console for the credit risk scoring system.

Scoring goes through the FastAPI service rather than loading a model here, so
there is one scoring path and every assessment is recorded for audit.

Run:
    API_URL=http://127.0.0.1:8000 streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import api_health, build_payload, score_application  # noqa: E402
from styles import BAND, BRAND, CSS, PRODUCT  # noqa: E402

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

HOME_OWNERSHIP = {
    "RENT": "Renting",
    "OWN": "Owns outright",
    "MORTGAGE": "Mortgaged",
    "OTHER": "Other",
}
LOAN_INTENT = {
    "PERSONAL": "Personal",
    "EDUCATION": "Education",
    "MEDICAL": "Medical",
    "VENTURE": "Business venture",
    "HOMEIMPROVEMENT": "Home improvement",
    "DEBTCONSOLIDATION": "Debt consolidation",
}
LOAN_GRADE = ["A", "B", "C", "D", "E", "F", "G"]

st.set_page_config(page_title=f"{BRAND} · {PRODUCT}", page_icon="▣", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

if "decisions" not in st.session_state:
    st.session_state.decisions = []


def money(x: float) -> str:
    return f"${x:,.0f}"


st.markdown(
    f"""<div class="mc-header">
      <div class="mc-brand">{BRAND}<span>{PRODUCT}</span></div>
    </div>""",
    unsafe_allow_html=True,
)

# Availability only. No version, uptime or service internals are surfaced.
available, _ = api_health(API_URL)

with st.sidebar:
    st.markdown('<div class="mc-section">Status</div>', unsafe_allow_html=True)
    if available:
        st.success("Ready to assess")
    else:
        st.error("Assessments unavailable")
        st.markdown(
            '<span class="mc-note">Applications cannot be assessed right now. '
            "Please try again shortly or contact support.</span>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown('<div class="mc-section">This session</div>', unsafe_allow_html=True)
    st.metric("Applications assessed", len(st.session_state.decisions))
    if st.session_state.decisions:
        approved = sum(1 for d in st.session_state.decisions if d["risk_label"] == "LOW")
        st.metric("Approved", f"{approved}/{len(st.session_state.decisions)}")

    st.divider()
    st.markdown(
        '<span class="mc-note">Assessments are advisory and recorded for audit.</span>',
        unsafe_allow_html=True,
    )


tab_new, tab_queue, tab_portfolio = st.tabs(
    ["New application", "Session queue", "Portfolio upload"]
)


with tab_new:
    if not available:
        st.error(
            "Applications cannot be assessed at the moment. "
            "Please try again shortly rather than deciding manually."
        )

    with st.form("application", border=False):
        st.markdown('<div class="mc-section">Applicant details</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        with a1:
            reference = st.text_input("Application reference", placeholder="e.g. APP-10482")
            age = st.number_input("Age", 18, 100, 30)
        with a2:
            income = st.number_input("Gross annual income", 1_000, 5_000_000, 60_000, step=1_000)
            emp_length = st.number_input("Years in employment", 0.0, 60.0, 5.0, step=0.5)
        with a3:
            home = st.selectbox(
                "Housing status", list(HOME_OWNERSHIP), format_func=HOME_OWNERSHIP.get
            )
            cred_hist = st.number_input("Credit history (years)", 0, 100, 4)

        st.markdown('<div class="mc-section">Credit file</div>', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            prior_default = st.radio(
                "Prior default on file", ["N", "Y"],
                format_func=lambda v: "No" if v == "N" else "Yes",
                horizontal=True,
            )
        with b2:
            grade = st.selectbox("Internal credit grade", LOAN_GRADE, index=2)

        st.markdown('<div class="mc-section">Facility requested</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amnt = st.number_input("Amount requested", 500, 1_000_000, 10_000, step=500)
        with c2:
            int_rate = st.number_input("Offered rate (%)", 0.1, 100.0, 12.5, step=0.1)
        with c3:
            intent = st.selectbox("Purpose", list(LOAN_INTENT), format_func=LOAN_INTENT.get)

        percent_income = min(loan_amnt / income, 1.0) if income else 0.0
        st.markdown(
            f'<span class="mc-note">Exposure to income: <b>{percent_income:.1%}</b> · '
            f"{money(loan_amnt)} against {money(income)} annual income.</span>",
            unsafe_allow_html=True,
        )

        submitted = st.form_submit_button(
            "Assess application", type="primary", disabled=not available
        )

    if submitted:
        payload = build_payload(
            age=age, income=income, emp_length=emp_length, loan_amnt=loan_amnt,
            int_rate=int_rate, percent_income=percent_income, cred_hist=cred_hist,
            home=home, intent=intent, grade=grade, prior_default=prior_default,
        )
        ok, result = score_application(payload, API_URL)

        if not ok:
            st.error(
                "This application could not be assessed. Please check the details "
                "and try again."
            )
        else:
            band = BAND[result["risk_label"]]
            prob = result["default_probability"]
            assessed_at = datetime.now(timezone.utc)

            st.markdown(
                f"""<div class="mc-decision" style="background:{band['bg']};
                     border:1px solid {band['border']}">
                  <div class="band" style="color:{band['color']}">
                    {result['risk_label']} RISK · {prob:.1%} probability of default</div>
                  <div class="verdict" style="color:{band['color']}">{band['decision']}</div>
                  <div class="detail">{band['detail']}</div>
                </div>""",
                unsafe_allow_html=True,
            )

            st.markdown('<div class="mc-section">Application profile</div>', unsafe_allow_html=True)
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Exposure to income", f"{percent_income:.1%}")
            f2.metric("Credit grade", grade)
            f3.metric("Prior default", "Yes" if prior_default == "Y" else "No")
            f4.metric("Employment", f"{emp_length:.0f} yr")

            ref = reference.strip() or "(not supplied)"
            st.markdown(
                f"""<div class="mc-ref">
                  Reference {ref} &nbsp;·&nbsp; Assessment {result['request_id']}<br>
                  Completed {assessed_at.strftime('%d %b %Y, %H:%M UTC')}
                </div>""",
                unsafe_allow_html=True,
            )

            st.session_state.decisions.insert(
                0,
                {
                    "reference": ref,
                    "assessed_at": assessed_at.strftime("%H:%M:%S"),
                    "amount": loan_amnt,
                    "income": income,
                    "grade": grade,
                    "default_probability": prob,
                    "risk_label": result["risk_label"],
                    "decision": band["decision"],
                    "assessment_id": result["request_id"],
                },
            )


with tab_queue:
    st.markdown('<div class="mc-section">Assessed this session</div>', unsafe_allow_html=True)
    if not st.session_state.decisions:
        st.info("No applications assessed yet. Submit one from the New application tab.")
    else:
        queue = pd.DataFrame(st.session_state.decisions)
        display = queue.assign(
            default_probability=lambda d: d.default_probability.map("{:.1%}".format),
            amount=lambda d: d.amount.map(money),
            income=lambda d: d.income.map(money),
        )
        st.dataframe(
            display[
                ["reference", "assessed_at", "amount", "income", "grade",
                 "default_probability", "risk_label", "decision"]
            ],
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Export session decisions (CSV)",
            queue.to_csv(index=False).encode(),
            f"decisions_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
        )


with tab_portfolio:
    st.markdown('<div class="mc-section">Bulk assessment</div>', unsafe_allow_html=True)
    st.write(
        "Upload a CSV of applications to assess a book in one pass. Each row is "
        "assessed individually and recorded."
    )
    uploaded = st.file_uploader("Application file (CSV)", type="csv")
    limit = st.number_input("Maximum rows to assess", 1, 2000, 100)

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.caption(f"{len(df):,} applications in file · assessing {min(limit, len(df)):,}")
        st.dataframe(df.head(), width="stretch", hide_index=True)

        if st.button("Assess portfolio", type="primary", disabled=not available):
            rows, failures = [], 0
            bar = st.progress(0.0, text="Assessing applications...")
            subset = df.head(int(limit))
            for i, (_, r) in enumerate(subset.iterrows(), start=1):
                ok, res = score_application(json.loads(r.to_json()), API_URL)
                if ok:
                    rows.append({**r.to_dict(), **res})
                else:
                    failures += 1
                bar.progress(i / len(subset), text=f"Assessed {i} of {len(subset)}")
            bar.empty()

            if rows:
                out = pd.DataFrame(rows)
                counts = out["risk_label"].value_counts()
                exposure = out.get("loan_amnt", pd.Series(dtype=float)).sum()

                st.success(f"{len(rows):,} assessed · {failures} could not be assessed")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Approve", int(counts.get("LOW", 0)))
                m2.metric("Refer", int(counts.get("MEDIUM", 0)))
                m3.metric("Decline", int(counts.get("HIGH", 0)))
                m4.metric("Total exposure", money(exposure) if exposure else "-")

                st.dataframe(
                    out.drop(columns=["model_version"], errors="ignore"),
                    width="stretch",
                    hide_index=True,
                )
                st.download_button(
                    "Export assessed portfolio (CSV)",
                    out.drop(columns=["model_version"], errors="ignore").to_csv(index=False).encode(),
                    "assessed_portfolio.csv",
                    "text/csv",
                )
            else:
                st.error(
                    f"No applications could be assessed; {failures} rejected. "
                    "Check that the file uses the expected column names."
                )
