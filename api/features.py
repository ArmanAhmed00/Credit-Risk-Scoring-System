"""Feature engineering for inference.

Mirrors scripts/data_cleaning.py exactly. It is duplicated rather than imported
so the api/ folder stays self-contained in the Docker image (scripts/ is not
copied in). tests/test_api.py asserts these constants stay identical to the
cleaning module, so the two cannot silently drift apart.
"""

from __future__ import annotations

from typing import Any

# Must equal data_cleaning.INCOME_SERVICEABLE_SHARE.
INCOME_SERVICEABLE_SHARE = 0.3

# Must equal data_cleaning.OUTPUT_COLUMNS minus the label, and matches the
# "feature_order" key written into encoding_maps.json by scripts/train.py.
FEATURE_ORDER = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "debt_to_income",
    "loan_to_income",
    "credit_utilization",
    "home_ownership_enc",
    "loan_intent_enc",
    "loan_grade_enc",
    "cb_default_enc",
]

# request field -> key inside encoding_maps.json -> encoded output column
ENCODED_FIELDS = [
    ("person_home_ownership", "person_home_ownership", "home_ownership_enc"),
    ("loan_intent", "loan_intent", "loan_intent_enc"),
    ("loan_grade", "loan_grade", "loan_grade_enc"),
    ("cb_person_default_on_file", "cb_person_default_on_file", "cb_default_enc"),
]


class FeatureEncodingError(ValueError):
    """Raised when a request value has no entry in the encoding maps."""


def build_feature_row(payload: dict[str, Any], encoding_maps: dict[str, Any]) -> dict[str, float]:
    """Turn one validated request into the 14-column feature row the model expects."""
    income = float(payload["person_income"])
    loan_amnt = float(payload["loan_amnt"])

    if income <= 0:
        # Schema validation should prevent this; guard anyway so we never
        # divide by zero and hand the model a NaN.
        raise FeatureEncodingError("person_income must be greater than 0")

    row: dict[str, float] = {
        "person_age": float(payload["person_age"]),
        "person_income": income,
        "person_emp_length": float(payload["person_emp_length"]),
        "loan_amnt": loan_amnt,
        "loan_int_rate": float(payload["loan_int_rate"]),
        "loan_percent_income": float(payload["loan_percent_income"]),
        "cb_person_cred_hist_length": float(payload["cb_person_cred_hist_length"]),
    }

    # Same three ratios as data_cleaning.engineer_features.
    row["debt_to_income"] = loan_amnt / income
    row["loan_to_income"] = float(payload["loan_percent_income"])
    row["credit_utilization"] = loan_amnt / (income * INCOME_SERVICEABLE_SHARE)

    for field, map_key, out_col in ENCODED_FIELDS:
        mapping = encoding_maps.get(map_key)
        if not mapping:
            raise FeatureEncodingError(f"encoding map '{map_key}' missing from encoding_maps.json")

        raw = str(payload[field]).strip().upper()
        if raw not in mapping:
            raise FeatureEncodingError(
                f"'{raw}' is not a known value for {field}; expected one of {sorted(mapping)}"
            )
        row[out_col] = float(mapping[raw])

    return {col: row[col] for col in FEATURE_ORDER}
