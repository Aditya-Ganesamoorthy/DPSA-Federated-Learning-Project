"""Loads trained state_dicts and serves predictions. Encoding comes from ml.schema (shared with training)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ml import model as M, schema

MODEL_FILES = {"centralized": "centralized.pt", "federated": "federated.pt", "federated_dp": "federated_dp.pt"}


class ModelsMissing(Exception):
    pass


class Predictor:
    def __init__(self, models_dir: Path):
        self.dir = Path(models_dir)
        self._models: dict[str, torch.nn.Module] = {}
        self.cfg: dict = {}

    def available(self) -> list[str]:
        return [k for k, f in MODEL_FILES.items() if (self.dir / f).is_file()]

    def _load(self, name: str):
        if name not in self._models:
            f, cfgf = self.dir / MODEL_FILES[name], self.dir / "model_config.json"
            if not f.is_file() or not cfgf.is_file():
                raise ModelsMissing(f"Model '{name}' not found. Run `python -m ml.run_all` first.")
            self.cfg = json.loads(cfgf.read_text())
            net = M.make_model(self.cfg.get("kind", "mlp"), 0)
            net.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
            net.eval()
            self._models[name] = net
        return self._models[name]

    def predict(self, name: str, record: dict) -> dict:
        net = self._load(name)
        x = schema.encode_api_record(record)[None, :]
        p = float(M.predict_proba(net, x)[0])
        thr = float(self.cfg.get("threshold", 0.5))
        return {"probability": p, "label": int(p >= thr), "threshold": thr}

    def predict_from_vector(self, name: str, x: np.ndarray) -> float:
        return float(M.predict_proba(self._load(name), x.astype(np.float32))[0])
