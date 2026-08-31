"""Data cleaning pipeline for the credit risk scoring system.

Each stage is an independent, testable function operating on a DataFrame.
``run_cleaning()`` wires them together for Airflow.

Paths are read from the environment so Airflow can override them:
    RAW_DATA_PATH        (default: data/credit_risk_dataset.csv)
    PROCESSED_DATA_PATH  (default: data/processed/features.csv)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Logging: stdout so Airflow's task log captures every stage.
# --------------------------------------------------------------------------
logger = logging.getLogger("data_cleaning")

if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# --------------------------------------------------------------------------
# Fixed encoding maps. These are part of the model contract: training and
# inference MUST use these exact values, so they are hardcoded, never fitted.
# --------------------------------------------------------------------------
HOME_OWNERSHIP_MAP = {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}

LOAN_GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

DEFAULT_ON_FILE_MAP = {"N": 0, "Y": 1}

LOAN_INTENT_MAP = {
    "PERSONAL": 0,
    "EDUCATION": 1,
    "MEDICAL": 2,
    "VENTURE": 3,
    "HOMEIMPROVEMENT": 4,
    "DEBTCONSOLIDATION": 5,
}

# Outlier thresholds (data entry errors, per data inspection).
MAX_PERSON_AGE = 100
MAX_EMP_LENGTH = 60

# Assumed share of income available to service debt, for credit_utilization.
INCOME_SERVICEABLE_SHARE = 0.3

# Exact output contract: these 15 columns, in this order.
OUTPUT_COLUMNS = [
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
    "loan_status",
]

REQUIRED_RAW_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

DEFAULT_RAW_PATH = "data/credit_risk_dataset.csv"
DEFAULT_PROCESSED_PATH = "data/processed/features.csv"


def get_raw_path() -> Path:
    """Resolve the raw input path from RAW_DATA_PATH."""
    return Path(os.environ.get("RAW_DATA_PATH", DEFAULT_RAW_PATH))


def get_processed_path() -> Path:
    """Resolve the cleaned output path from PROCESSED_DATA_PATH."""
    return Path(os.environ.get("PROCESSED_DATA_PATH", DEFAULT_PROCESSED_PATH))


# --------------------------------------------------------------------------
# Stage 1: load
# --------------------------------------------------------------------------
def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw CSV and verify the expected schema is present."""
    raw_path = Path(path) if path is not None else get_raw_path()
    logger.info("STAGE load_raw | reading %s", raw_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    df = pd.read_csv(raw_path)
    logger.info("STAGE load_raw | loaded %d rows x %d columns", len(df), df.shape[1])

    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")

    null_counts = df.isnull().sum()
    for col, n in null_counts[null_counts > 0].items():
        logger.info("STAGE load_raw | nulls in %s: %d (%.2f%%)", col, n, 100 * n / len(df))

    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        logger.warning(
            "STAGE load_raw | %d exact duplicate rows present (NOT removed - not in cleaning spec)",
            n_dupes,
        )

    return df


# --------------------------------------------------------------------------
# Stage 2: outliers
# --------------------------------------------------------------------------
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with impossible person_age or person_emp_length values."""
    logger.info("STAGE remove_outliers | input rows: %d", len(df))
    before = len(df)

    age_mask = df["person_age"] > MAX_PERSON_AGE
    emp_mask = df["person_emp_length"] > MAX_EMP_LENGTH  # NaN > 60 is False; nulls survive

    logger.info(
        "STAGE remove_outliers | person_age > %d: %d rows", MAX_PERSON_AGE, int(age_mask.sum())
    )
    logger.info(
        "STAGE remove_outliers | person_emp_length > %d: %d rows",
        MAX_EMP_LENGTH,
        int(emp_mask.sum()),
    )

    df = df.loc[~(age_mask | emp_mask)].copy()

    removed = before - len(df)
    logger.info(
        "STAGE remove_outliers | removed %d rows (%.3f%%), %d remaining",
        removed,
        100 * removed / before if before else 0.0,
        len(df),
    )
    return df


# --------------------------------------------------------------------------
# Stage 3: nulls
# --------------------------------------------------------------------------
def fill_nulls(df: pd.DataFrame, int_rate_median: float | None = None) -> pd.DataFrame:
    """Fill loan_int_rate with the median and person_emp_length with 0.

    ``int_rate_median`` is a *fitted* parameter. Pass the value persisted at
    training time when cleaning inference data, so the same constant is used
    on both sides. When omitted it is computed from ``df``.
    """
    logger.info("STAGE fill_nulls | input rows: %d", len(df))
    df = df.copy()

    n_rate_null = int(df["loan_int_rate"].isnull().sum())
    if int_rate_median is None:
        int_rate_median = float(df["loan_int_rate"].median())
        logger.info("STAGE fill_nulls | computed loan_int_rate median: %.4f", int_rate_median)
    else:
        int_rate_median = float(int_rate_median)
        logger.info("STAGE fill_nulls | using supplied loan_int_rate median: %.4f", int_rate_median)

    df["loan_int_rate"] = df["loan_int_rate"].fillna(int_rate_median)
    logger.info("STAGE fill_nulls | filled %d loan_int_rate nulls with median", n_rate_null)

    n_emp_null = int(df["person_emp_length"].isnull().sum())
    df["person_emp_length"] = df["person_emp_length"].fillna(0)
    logger.info("STAGE fill_nulls | filled %d person_emp_length nulls with 0", n_emp_null)

    remaining = int(df[REQUIRED_RAW_COLUMNS].isnull().sum().sum())
    logger.info("STAGE fill_nulls | remaining nulls across raw columns: %d", remaining)

    # Stash the median so run_cleaning can persist it for inference.
    df.attrs["int_rate_median"] = int_rate_median
    return df


# --------------------------------------------------------------------------
# Stage 4: categorical encoding
# --------------------------------------------------------------------------
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the fixed encoding maps, failing loudly on unseen categories."""
    logger.info("STAGE encode_categoricals | input rows: %d", len(df))
    df = df.copy()

    encodings = [
        ("person_home_ownership", "home_ownership_enc", HOME_OWNERSHIP_MAP),
        ("loan_intent", "loan_intent_enc", LOAN_INTENT_MAP),
        ("loan_grade", "loan_grade_enc", LOAN_GRADE_MAP),
        ("cb_person_default_on_file", "cb_default_enc", DEFAULT_ON_FILE_MAP),
    ]

    for source, target, mapping in encodings:
        values = df[source].astype("string").str.strip().str.upper()
        unknown = sorted(set(values.dropna().unique()) - set(mapping))
        if unknown:
            # Silently encoding these as NaN would poison the model, so stop here.
            raise ValueError(
                f"Unmapped categories in '{source}': {unknown}. "
                f"Known categories: {sorted(mapping)}. "
                "Update the fixed encoding map deliberately - it is part of the "
                "training/inference contract."
            )

        encoded = values.map(mapping)
        if encoded.isnull().any():
            n_null = int(encoded.isnull().sum())
            raise ValueError(f"'{source}' has {n_null} null/blank values that cannot be encoded.")

        df[target] = encoded.astype("int64")
        counts = df[target].value_counts().sort_index().to_dict()
        logger.info("STAGE encode_categoricals | %s -> %s, counts: %s", source, target, counts)

    return df


# --------------------------------------------------------------------------
# Stage 5: feature engineering
# --------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the three ratio features.

    NOTE: these three are collinear by construction. In the source data
    loan_percent_income == loan_amnt / person_income (2dp), so
    loan_to_income ~= debt_to_income, and
    credit_utilization == debt_to_income / INCOME_SERVICEABLE_SHARE exactly.
    Fine for tree models; expect unstable coefficients in a linear model.
    """
    logger.info("STAGE engineer_features | input rows: %d", len(df))
    df = df.copy()

    income = df["person_income"].astype("float64")
    non_positive = int((income <= 0).sum())
    if non_positive:
        # Guard for inference data; the training file has min income 4000.
        logger.warning(
            "STAGE engineer_features | %d rows with person_income <= 0; "
            "ratio features set to NaN and dropped by validate()",
            non_positive,
        )
    safe_income = income.where(income > 0, np.nan)

    df["debt_to_income"] = df["loan_amnt"] / safe_income
    df["loan_to_income"] = df["loan_percent_income"].astype("float64")
    df["credit_utilization"] = df["loan_amnt"] / (safe_income * INCOME_SERVICEABLE_SHARE)

    for col in ("debt_to_income", "loan_to_income", "credit_utilization"):
        s = df[col]
        logger.info(
            "STAGE engineer_features | %s -> min=%.4f median=%.4f max=%.4f nulls=%d",
            col,
            s.min(),
            s.median(),
            s.max(),
            int(s.isnull().sum()),
        )

    return df


# --------------------------------------------------------------------------
# Stage 6: validation
# --------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Select the 15 output columns and assert the output contract.

    Raises on any violation so an Airflow task fails loudly rather than
    writing a silently corrupt feature file.
    """
    logger.info("STAGE validate | input rows: %d", len(df))

    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Output is missing required columns: {missing}")

    out = df[OUTPUT_COLUMNS].copy()

    # Non-finite values would break model training downstream.
    numeric = out.select_dtypes(include=[np.number])
    non_finite = ~np.isfinite(numeric.to_numpy(dtype="float64"))
    if non_finite.any():
        bad_rows = non_finite.any(axis=1)
        per_col = {
            c: int(n) for c, n in zip(numeric.columns, non_finite.sum(axis=0)) if n
        }
        logger.warning(
            "STAGE validate | dropping %d rows with NaN/inf values: %s",
            int(bad_rows.sum()),
            per_col,
        )
        out = out.loc[~bad_rows].copy()

    if out.empty:
        raise ValueError("Validation produced an empty dataset.")

    if list(out.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"Column order mismatch. Got {list(out.columns)}")
    if out.shape[1] != 15:
        raise ValueError(f"Expected exactly 15 columns, got {out.shape[1]}")

    n_null = int(out.isnull().sum().sum())
    if n_null:
        raise ValueError(f"Output still contains {n_null} null values.")

    checks = {
        "person_age": (0, MAX_PERSON_AGE),
        "person_emp_length": (0, MAX_EMP_LENGTH),
        "home_ownership_enc": (0, 3),
        "loan_intent_enc": (0, 5),
        "loan_grade_enc": (1, 7),
        "cb_default_enc": (0, 1),
        "loan_status": (0, 1),
    }
    for col, (lo, hi) in checks.items():
        col_min, col_max = out[col].min(), out[col].max()
        if col_min < lo or col_max > hi:
            raise ValueError(f"'{col}' out of range [{lo}, {hi}]: got [{col_min}, {col_max}]")

    for col in ("person_income", "loan_amnt"):
        if (out[col] <= 0).any():
            raise ValueError(f"'{col}' contains non-positive values.")

    n_dupes = int(out.duplicated().sum())
    if n_dupes:
        logger.warning(
            "STAGE validate | %d duplicate rows in output (NOT removed - not in cleaning spec)",
            n_dupes,
        )

    rate = out["loan_status"].mean()
    logger.info("STAGE validate | all checks passed on %d rows x 15 columns", len(out))
    logger.info(
        "STAGE validate | loan_status default rate: %.4f (%d positive / %d total)",
        rate,
        int(out["loan_status"].sum()),
        len(out),
    )
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_cleaning(
    raw_path: str | Path | None = None,
    processed_path: str | Path | None = None,
    int_rate_median: float | None = None,
) -> pd.DataFrame:
    """Run the full pipeline and write the cleaned feature file."""
    raw = Path(raw_path) if raw_path is not None else get_raw_path()
    out_path = Path(processed_path) if processed_path is not None else get_processed_path()

    logger.info("=" * 70)
    logger.info("CREDIT RISK DATA CLEANING PIPELINE")
    logger.info("RAW_DATA_PATH       = %s", raw)
    logger.info("PROCESSED_DATA_PATH = %s", out_path)
    logger.info("=" * 70)

    df = load_raw(raw)
    input_rows = len(df)

    df = remove_outliers(df)
    df = fill_nulls(df, int_rate_median=int_rate_median)
    median_used = df.attrs.get("int_rate_median")
    df = encode_categoricals(df)
    df = engineer_features(df)
    out = validate(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info("WRITE | wrote %d rows x %d columns to %s", len(out), out.shape[1], out_path)

    # Persist the fitted median next to the features so inference can reuse it.
    params_path = out_path.parent / "cleaning_params.json"
    params_path.write_text(
        json.dumps(
            {
                "int_rate_median": median_used,
                "home_ownership_map": HOME_OWNERSHIP_MAP,
                "loan_intent_map": LOAN_INTENT_MAP,
                "loan_grade_map": LOAN_GRADE_MAP,
                "cb_default_map": DEFAULT_ON_FILE_MAP,
                "income_serviceable_share": INCOME_SERVICEABLE_SHARE,
                "output_columns": OUTPUT_COLUMNS,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("WRITE | wrote cleaning parameters to %s", params_path)

    logger.info("=" * 70)
    logger.info(
        "PIPELINE COMPLETE | %d raw rows -> %d clean rows (%d dropped)",
        input_rows,
        len(out),
        input_rows - len(out),
    )
    logger.info("=" * 70)
    return out


if __name__ == "__main__":
    try:
        run_cleaning()
    except Exception:
        logger.exception("PIPELINE FAILED")
        sys.exit(1)
