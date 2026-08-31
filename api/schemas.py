"""Pydantic request/response models for the credit risk scoring API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

HOME_OWNERSHIP_VALUES = ("RENT", "OWN", "MORTGAGE", "OTHER")
LOAN_INTENT_VALUES = (
    "PERSONAL",
    "EDUCATION",
    "MEDICAL",
    "VENTURE",
    "HOMEIMPROVEMENT",
    "DEBTCONSOLIDATION",
)
LOAN_GRADE_VALUES = ("A", "B", "C", "D", "E", "F", "G")
DEFAULT_ON_FILE_VALUES = ("N", "Y")

RiskLabel = Literal["LOW", "MEDIUM", "HIGH"]


class PredictionRequest(BaseModel):
    """One credit application scored by the model."""

    person_age: int = Field(..., ge=18, le=100, description="Applicant age in years (18-100)")
    person_income: float = Field(..., gt=0, description="Annual income; must be > 0")
    person_emp_length: float = Field(..., ge=0, le=60, description="Employment length in years")
    loan_amnt: float = Field(..., gt=0, description="Requested loan amount; must be > 0")
    loan_int_rate: float = Field(..., gt=0, le=100, description="Interest rate (percent)")
    loan_percent_income: float = Field(..., ge=0, le=1, description="Loan as a fraction of income")
    cb_person_cred_hist_length: int = Field(..., ge=0, le=100, description="Credit history years")

    person_home_ownership: Literal[HOME_OWNERSHIP_VALUES]
    loan_intent: Literal[LOAN_INTENT_VALUES]
    loan_grade: Literal[LOAN_GRADE_VALUES]
    cb_person_default_on_file: Literal[DEFAULT_ON_FILE_VALUES]

    @field_validator(
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
        mode="before",
    )
    @classmethod
    def _normalize_categorical(cls, v):
        """Uppercase/strip before the Literal check, matching data_cleaning."""
        return v.strip().upper() if isinstance(v, str) else v

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "person_age": 30,
                "person_income": 60000,
                "person_emp_length": 5.0,
                "loan_amnt": 10000,
                "loan_int_rate": 12.5,
                "loan_percent_income": 0.17,
                "cb_person_cred_hist_length": 4,
                "person_home_ownership": "RENT",
                "loan_intent": "PERSONAL",
                "loan_grade": "C",
                "cb_person_default_on_file": "N",
            }
        },
    )


class PredictionResponse(BaseModel):
    default_probability: float = Field(..., ge=0.0, le=1.0)
    risk_label: RiskLabel
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
