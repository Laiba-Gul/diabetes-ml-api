"""Pydantic request/response models with input validation.

Ranges are deliberately generous (to allow real clinical values) but reject
values that are impossible or clearly data-entry errors. A value of 0 is
accepted for glucose, blood pressure, skin thickness, insulin and BMI because
in the PIMA convention 0 means "not measured"; the model pipeline imputes it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "pregnancies": 6,
                    "glucose": 148,
                    "blood_pressure": 72,
                    "skin_thickness": 35,
                    "insulin": 0,
                    "bmi": 33.6,
                    "diabetes_pedigree_function": 0.627,
                    "age": 50,
                }
            ]
        },
    )

    pregnancies: int = Field(..., ge=0, le=20, description="Number of times pregnant")
    glucose: float = Field(
        ..., ge=0, le=300, description="Plasma glucose, 2h OGTT (mg/dL). 0 = not measured"
    )
    blood_pressure: float = Field(
        ..., ge=0, le=200, description="Diastolic blood pressure (mm Hg). 0 = not measured"
    )
    skin_thickness: float = Field(
        ..., ge=0, le=100, description="Triceps skin fold thickness (mm). 0 = not measured"
    )
    insulin: float = Field(
        ..., ge=0, le=1000, description="2-hour serum insulin (mu U/mL). 0 = not measured"
    )
    bmi: float = Field(..., ge=0, le=80, description="Body mass index (kg/m^2). 0 = not measured")
    diabetes_pedigree_function: float = Field(
        ..., ge=0, le=3, description="Diabetes pedigree function (genetic risk score)"
    )
    age: int = Field(
        ..., ge=21, le=120, description="Age in years (the PIMA study population is aged 21+)"
    )


class PredictionResponse(BaseModel):
    predicted_class: int = Field(..., description="0 = non-diabetic, 1 = diabetic")
    label: str = Field(..., description="Human-readable class label")
    probability: float = Field(
        ..., ge=0, le=1, description="Model probability that the patient is diabetic (class 1)"
    )
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_version: str
    model_type: str
    features: list[str]
    decision_threshold: float
    trained_at: str | None = None
    metrics: dict[str, float] | None = None
