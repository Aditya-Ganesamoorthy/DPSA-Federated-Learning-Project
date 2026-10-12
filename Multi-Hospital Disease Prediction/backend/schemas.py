from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from ml import schema

MODEL_CHOICES = Literal["centralized", "federated", "federated_dp"]

# Patient fields are generated from the real feature list, never hand-copied.
_fields = {c: (bool, Field(..., description=f"{c.replace('_', ' ')} (true = yes)")) for c in schema.BINARY_FEATURES}

PatientFeatures = create_model(
    "PatientFeatures",
    __config__=ConfigDict(extra="forbid"),
    gender=(Literal["M", "F"], Field(..., description="M or F")),
    age=(int, Field(..., ge=schema.AGE_RANGE[0], le=schema.AGE_RANGE[1])),
    **_fields,
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: MODEL_CHOICES = "federated"
    patient: PatientFeatures  # type: ignore[valid-type]


class PredictResponse(BaseModel):
    probability: float
    label: int
    threshold: float
    model_used: str
    disclaimer: str
    model_note: str


DISCLAIMER = "Educational demo, not medical advice."
MODEL_NOTE = ("On this dataset every model performs close to chance (ROC-AUC ~0.5): the labels carry no "
              "learnable signal, so this output has no clinical meaning.")
