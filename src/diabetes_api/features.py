"""Single source of truth for the PIMA feature schema.

Shared by the training script and the API so the column order used at
training time is always the order used at inference time.
"""

from __future__ import annotations

# Column names exactly as used in the trained pipeline (snake_case).
FEATURE_COLUMNS: list[str] = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree_function",
    "age",
]

TARGET_COLUMN: str = "outcome"

# In the PIMA dataset a value of 0 in these columns is physiologically
# impossible and is used to encode "not measured". They are converted to NaN
# and imputed inside the pipeline.
ZERO_AS_MISSING_COLUMNS: list[str] = [
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
]

CLASS_LABELS: dict[int, str] = {0: "non-diabetic", 1: "diabetic"}
