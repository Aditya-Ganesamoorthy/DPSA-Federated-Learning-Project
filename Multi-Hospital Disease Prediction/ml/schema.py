"""Single source of truth for columns, encodings and feature order.

Used by training AND the API. Never re-implement encoding elsewhere.
Survey convention of the Kaggle file: 1 = NO, 2 = YES for symptom fields.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "lung_cancer"
TARGET_MAP = {"NO": 0, "YES": 1}
GENDER_MAP = {"F": 0, "M": 1}
AGE_SCALE = 100.0                      # fixed constant, no data-fitted scaler
AGE_BAND_WIDTH = 10

BINARY_FEATURES = [
    "smoking", "yellow_fingers", "anxiety", "peer_pressure", "chronic_disease",
    "fatigue", "allergy", "wheezing", "alcohol_consuming", "coughing",
    "shortness_of_breath", "swallowing_difficulty", "chest_pain",
]
# Model input order. Do not change without retraining.
FEATURES = ["gender", "age"] + BINARY_FEATURES
N_FEATURES = len(FEATURES)
QUASI_IDENTIFIERS = ["age_band", "gender"]
IDENTIFIER_HINTS = ("name", "id", "phone", "address", "email", "ssn")
AGE_RANGE = (1, 120)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lstrip("﻿").lower().replace(" ", "_") for c in out.columns]
    return out


def age_band(age) -> pd.Series:
    a = pd.Series(age).astype(int)
    lo = (a // AGE_BAND_WIDTH) * AGE_BAND_WIDTH
    return lo.astype(str) + "-" + (lo + AGE_BAND_WIDTH - 1).astype(str)


def encode_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw (normalised-name) Kaggle frame -> numeric frame with FEATURES (+ TARGET if present).

    Raw age and 1/2 survey codes are mapped to model units: age/100, gender F/M -> 0/1, 1/2 -> 0/1.
    """
    out = pd.DataFrame(index=raw.index)
    out["gender"] = raw["gender"].astype(str).str.strip().str.upper().map(GENDER_MAP)
    out["age"] = raw["age"].astype(float) / AGE_SCALE
    for c in BINARY_FEATURES:
        out[c] = raw[c].astype(int).map({1: 0, 2: 1})
    if TARGET in raw.columns:
        out[TARGET] = raw[TARGET].astype(str).str.strip().str.upper().map(TARGET_MAP)
    if out.isna().any().any():
        raise ValueError("Unknown category encountered while encoding")
    return out


def encode_api_record(rec: dict) -> np.ndarray:
    """API format (booleans, integer age, gender 'M'/'F') -> float32 model vector in FEATURES order."""
    row = {
        "gender": GENDER_MAP[str(rec["gender"]).upper()],
        "age": float(rec["age"]) / AGE_SCALE,
        **{c: int(bool(rec[c])) for c in BINARY_FEATURES},
    }
    return np.array([row[f] for f in FEATURES], dtype=np.float32)
