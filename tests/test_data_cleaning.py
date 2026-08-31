"""Tests for scripts/data_cleaning.py.

Each stage is exercised independently, then end to end.
"""

from __future__ import annotations

import json

import data_cleaning
import numpy as np
import pandas as pd
import pytest
from conftest import ENCODING_MAPS


def test_load_raw_reads_csv(sample_raw_df, tmp_path):
    path = tmp_path / "raw.csv"
    sample_raw_df.to_csv(path, index=False)

    df = data_cleaning.load_raw(path)
    assert len(df) == 20
    assert list(df.columns) == list(sample_raw_df.columns)


def test_load_raw_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Raw data file not found"):
        data_cleaning.load_raw(tmp_path / "nope.csv")


def test_load_raw_rejects_missing_columns(sample_raw_df, tmp_path):
    path = tmp_path / "bad.csv"
    sample_raw_df.drop(columns=["loan_grade"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        data_cleaning.load_raw(path)


def test_remove_outliers_drops_impossible_rows(sample_raw_df):
    out = data_cleaning.remove_outliers(sample_raw_df)

    assert len(out) == 18  # 20 - (1 bad age + 1 bad tenure)
    assert (out["person_age"] <= 100).all()
    assert (out["person_emp_length"].dropna() <= 60).all()


def test_remove_outliers_keeps_null_emp_length(sample_raw_df):
    """NaN > 60 is False, so null tenures must survive to fill_nulls."""
    out = data_cleaning.remove_outliers(sample_raw_df)
    assert out["person_emp_length"].isnull().sum() == 1


def test_remove_outliers_is_noop_on_clean_data(sample_raw_df):
    clean = sample_raw_df.drop(index=[18, 19])
    assert len(data_cleaning.remove_outliers(clean)) == len(clean)


def test_remove_outliers_does_not_mutate_input(sample_raw_df):
    before = len(sample_raw_df)
    data_cleaning.remove_outliers(sample_raw_df)
    assert len(sample_raw_df) == before


def test_fill_nulls_removes_all_nulls(sample_raw_df):
    out = data_cleaning.fill_nulls(sample_raw_df)
    assert out["loan_int_rate"].isnull().sum() == 0
    assert out["person_emp_length"].isnull().sum() == 0


def test_fill_nulls_uses_median_for_int_rate(sample_raw_df):
    expected = float(sample_raw_df["loan_int_rate"].median())
    out = data_cleaning.fill_nulls(sample_raw_df)

    assert out.loc[15, "loan_int_rate"] == pytest.approx(expected)
    assert out.loc[16, "loan_int_rate"] == pytest.approx(expected)


def test_fill_nulls_uses_zero_for_emp_length(sample_raw_df):
    out = data_cleaning.fill_nulls(sample_raw_df)
    assert out.loc[17, "person_emp_length"] == 0


def test_fill_nulls_accepts_supplied_median(sample_raw_df):
    """Inference must reuse the training median, not recompute it."""
    out = data_cleaning.fill_nulls(sample_raw_df, int_rate_median=99.0)
    assert out.loc[15, "loan_int_rate"] == 99.0
    assert out.attrs["int_rate_median"] == 99.0


def test_fill_nulls_does_not_alter_present_values(sample_raw_df):
    out = data_cleaning.fill_nulls(sample_raw_df)
    assert out.loc[0, "loan_int_rate"] == sample_raw_df.loc[0, "loan_int_rate"]


def test_encode_categoricals_creates_encoded_columns(sample_raw_df):
    out = data_cleaning.encode_categoricals(sample_raw_df)
    for col in ("home_ownership_enc", "loan_intent_enc", "loan_grade_enc", "cb_default_enc"):
        assert col in out.columns
        assert out[col].dtype.kind == "i"


@pytest.mark.parametrize(
    "source,target,mapping",
    [
        ("person_home_ownership", "home_ownership_enc", ENCODING_MAPS["person_home_ownership"]),
        ("loan_intent", "loan_intent_enc", ENCODING_MAPS["loan_intent"]),
        ("loan_grade", "loan_grade_enc", ENCODING_MAPS["loan_grade"]),
        ("cb_person_default_on_file", "cb_default_enc", ENCODING_MAPS["cb_person_default_on_file"]),
    ],
)
def test_encoding_values_are_correct(sample_raw_df, source, target, mapping):
    out = data_cleaning.encode_categoricals(sample_raw_df)
    for raw_value, encoded in zip(out[source], out[target], strict=True):
        assert encoded == mapping[raw_value]


def test_encoding_maps_match_specification():
    """These values are the train/serve contract; they must not drift."""
    assert data_cleaning.HOME_OWNERSHIP_MAP == {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
    assert data_cleaning.LOAN_GRADE_MAP == {
        "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7
    }
    assert data_cleaning.DEFAULT_ON_FILE_MAP == {"N": 0, "Y": 1}
    assert data_cleaning.LOAN_INTENT_MAP == {
        "PERSONAL": 0, "EDUCATION": 1, "MEDICAL": 2,
        "VENTURE": 3, "HOMEIMPROVEMENT": 4, "DEBTCONSOLIDATION": 5,
    }


def test_encode_categoricals_is_case_insensitive(sample_raw_df):
    df = sample_raw_df.copy()
    df["loan_grade"] = df["loan_grade"].str.lower()
    df["person_home_ownership"] = "  rent  "

    out = data_cleaning.encode_categoricals(df)
    assert (out["home_ownership_enc"] == 0).all()


def test_encode_categoricals_rejects_unknown_category(sample_raw_df):
    """An unseen grade must fail loudly, never encode to NaN."""
    df = sample_raw_df.copy()
    df.loc[0, "loan_grade"] = "Z"

    with pytest.raises(ValueError, match="Unmapped categories"):
        data_cleaning.encode_categoricals(df)


def test_encode_categoricals_rejects_nulls(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[0, "loan_intent"] = None

    with pytest.raises(ValueError):
        data_cleaning.encode_categoricals(df)


def test_engineered_ratios_are_correct(sample_raw_df):
    out = data_cleaning.engineer_features(sample_raw_df)
    row = out.iloc[0]

    assert row["debt_to_income"] == pytest.approx(row["loan_amnt"] / row["person_income"])
    assert row["credit_utilization"] == pytest.approx(
        row["loan_amnt"] / (row["person_income"] * 0.3)
    )
    assert row["loan_to_income"] == pytest.approx(row["loan_percent_income"])


def test_credit_utilization_is_debt_to_income_over_share(sample_raw_df):
    """credit_utilization is a pure rescale of debt_to_income."""
    out = data_cleaning.engineer_features(sample_raw_df)
    ratio = out["credit_utilization"] / out["debt_to_income"]
    assert np.allclose(ratio, 1 / data_cleaning.INCOME_SERVICEABLE_SHARE)


def test_engineer_features_guards_zero_income(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[0, "person_income"] = 0

    out = data_cleaning.engineer_features(df)
    assert pd.isna(out.loc[0, "debt_to_income"])
    assert pd.isna(out.loc[0, "credit_utilization"])


def test_validate_returns_exact_15_columns(sample_features_df):
    assert sample_features_df.shape[1] == 15
    assert list(sample_features_df.columns) == data_cleaning.OUTPUT_COLUMNS


def test_validate_output_has_no_nulls(sample_features_df):
    assert sample_features_df.isnull().sum().sum() == 0


def test_validate_output_is_all_numeric(sample_features_df):
    assert all(np.issubdtype(d, np.number) for d in sample_features_df.dtypes)


def test_validate_encoded_ranges(sample_features_df):
    assert sample_features_df["home_ownership_enc"].between(0, 3).all()
    assert sample_features_df["loan_intent_enc"].between(0, 5).all()
    assert sample_features_df["loan_grade_enc"].between(1, 7).all()
    assert sample_features_df["cb_default_enc"].isin([0, 1]).all()
    assert sample_features_df["loan_status"].isin([0, 1]).all()


def test_validate_rejects_missing_column(sample_features_df):
    with pytest.raises(ValueError, match="missing required columns"):
        data_cleaning.validate(sample_features_df.drop(columns=["credit_utilization"]))


def test_validate_drops_non_finite_rows(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[0, "person_income"] = 0  # -> NaN ratios

    pipeline = data_cleaning.engineer_features(
        data_cleaning.encode_categoricals(
            data_cleaning.fill_nulls(data_cleaning.remove_outliers(df))
        )
    )
    out = data_cleaning.validate(pipeline)
    assert len(out) == len(pipeline) - 1
    assert out.isnull().sum().sum() == 0


def test_validate_rejects_empty_dataset(sample_features_df):
    with pytest.raises(ValueError, match="empty dataset"):
        data_cleaning.validate(sample_features_df.iloc[0:0])


def test_run_cleaning_end_to_end(sample_raw_df, tmp_path):
    raw = tmp_path / "raw.csv"
    out_path = tmp_path / "processed" / "features.csv"
    sample_raw_df.to_csv(raw, index=False)

    result = data_cleaning.run_cleaning(raw_path=raw, processed_path=out_path)

    assert out_path.exists()
    assert len(result) == 18
    assert list(result.columns) == data_cleaning.OUTPUT_COLUMNS

    written = pd.read_csv(out_path)
    assert written.shape == (18, 15)
    assert written.isnull().sum().sum() == 0


def test_run_cleaning_writes_params_file(sample_raw_df, tmp_path):
    raw = tmp_path / "raw.csv"
    out_path = tmp_path / "processed" / "features.csv"
    sample_raw_df.to_csv(raw, index=False)
    data_cleaning.run_cleaning(raw_path=raw, processed_path=out_path)

    params = json.loads((out_path.parent / "cleaning_params.json").read_text())
    assert params["home_ownership_map"] == ENCODING_MAPS["person_home_ownership"]
    assert params["income_serviceable_share"] == 0.3
    assert params["int_rate_median"] is not None


def test_env_var_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("RAW_DATA_PATH", str(tmp_path / "r.csv"))
    monkeypatch.setenv("PROCESSED_DATA_PATH", str(tmp_path / "p.csv"))

    assert data_cleaning.get_raw_path() == tmp_path / "r.csv"
    assert data_cleaning.get_processed_path() == tmp_path / "p.csv"


def test_default_paths_when_env_unset(monkeypatch):
    monkeypatch.delenv("RAW_DATA_PATH", raising=False)
    monkeypatch.delenv("PROCESSED_DATA_PATH", raising=False)

    assert str(data_cleaning.get_raw_path()) == "data/credit_risk_dataset.csv"
    assert str(data_cleaning.get_processed_path()) == "data/processed/features.csv"
