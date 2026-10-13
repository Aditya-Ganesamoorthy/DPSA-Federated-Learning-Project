"""Serves predictions with numpy only (no torch on the server). Encoding comes from ml.schema (shared with training)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ml import model_io, schema

STATIC_MODELS = {"centralized": "centralized.npz", "federated": "federated.npz", "federated_dp": "federated_dp.npz"}
MODEL_CHOICES = ("global", *STATIC_MODELS)


class ModelsMissing(Exception):
    pass


class Predictor:
    def __init__(self, models_dir: Path, federation=None):
        self.dir = Path(models_dir)
        self.federation = federation
        self._cache: dict[str, list[np.ndarray]] = {}

    def available(self) -> list[str]:
        out = ["global"] if self.federation is not None else []
        return out + [k for k, f in STATIC_MODELS.items() if (self.dir / f).is_file()]

    def _weights(self, name: str):
        if name == "global":
            if self.federation is None:
                raise ModelsMissing("Federation is not running")
            with self.federation.lock:
                return self.federation.weights, self.federation.version
        if name not in self._cache:
            f = self.dir / STATIC_MODELS[name]
            if not f.is_file():
                raise ModelsMissing(f"Model '{name}' not found. Run `python scripts/export_npz.py` first.")
            self._cache[name] = model_io.load(f)[0]
        return self._cache[name], 0

    def predict(self, name: str, record: dict) -> dict:
        weights, version = self._weights(name)
        p = float(model_io.predict_proba(weights, schema.encode_api_record(record)[None, :])[0])
        return {"probability": p, "label": int(p >= 0.5), "threshold": 0.5, "model_version": version}
