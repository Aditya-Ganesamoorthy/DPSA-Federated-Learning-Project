"""The live federated-learning coordinator (the 'centralized server model' hospitals download from).

Protocol (synchronous rounds, FedAvg):
  1. A hospital downloads the current global model (version k).
  2. It trains locally on its own data (raw rows never leave) and uploads the new weights, its number of
     training examples, and the base version k it started from.
  3. When MIN_CLIENTS distinct hospitals have submitted for version k (or the admin forces it), the server
     computes w_{k+1} = sum_n (n_h / N) * w_h, evaluates it on the shared test file, saves it as version k+1,
     and publishes it to Hugging Face (if configured).

Safeguards: authenticated hospitals only; updates must be built on the CURRENT version (409 otherwise);
shape / finite-value validation; a cap on the L2 norm of (update - global) as a basic poisoning guard;
at most one pending update per hospital per round (a re-submission replaces the earlier one).
The live server uses the numpy FedAvg that the test-suite verifies against Flower's FedAvg.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from backend.services import hf_sync
from backend.settings import Settings
from ml import config, model_io, schema

log = logging.getLogger("federation")


class FederationError(Exception):
    def __init__(self, message: str, status: int = 422, **extra):
        super().__init__(message)
        self.status, self.extra = status, extra


class Federation:
    def __init__(self, settings: Settings):
        self.s = settings
        self.lock = threading.RLock()
        self.dir = Path(settings.state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pending: dict[str, dict] = {}
        self.history: list[dict] = []
        self.last_publish: dict = {"status": "never" if hf_sync.configured(settings) else "not configured"}
        self.weights: list[np.ndarray]
        self.version: int
        self._xte = self._yte = None
        self._load_state()

    # ---- state ----
    def _load_state(self) -> None:
        latest, hist = self.dir / "global_latest.npz", self.dir / "history.json"
        source = "local state"
        if not latest.exists() and hf_sync.configured(self.s):
            if hf_sync.restore(self.s, self.dir):
                source = "Hugging Face"
        if latest.exists():
            self.weights, self.version = model_io.load(latest)
            if hist.exists():
                self.history = json.loads(hist.read_text(encoding="utf-8"))
        else:
            self.weights, self.version = model_io.load(self.s.models_dir / "global_v0.npz")
            self.version = 0
            self.history = [self._record(0, [], [], source="initial primary model (trained across 3 simulated hospitals)")]
            self._persist()
            source = "initial primary model"
            if hf_sync.configured(self.s):      # make version 0 downloadable from the Hub straight away
                threading.Thread(target=self._publish, daemon=True).start()
        log.info("federation: loaded version %d from %s", self.version, source)

    def _persist(self) -> None:
        model_io.save(self.dir / "global_latest.npz", self.weights, self.version)
        (self.dir / "history.json").write_text(json.dumps(self.history, indent=1), encoding="utf-8")

    def _evaluate(self) -> dict:
        if self._xte is None:
            df = _read_test()
            self._xte, self._yte = df
        return model_io.metrics(self._yte, model_io.predict_proba(self.weights, self._xte))

    def _record(self, version: int, hospitals: list[str], counts: list[int], source: str = "") -> dict:
        m = self._evaluate()
        return {"version": version, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "participants": hospitals, "examples": counts, "total_examples": int(sum(counts)),
                "test_metrics": m, "note": source}

    # ---- public API ----
    def model_bytes(self) -> tuple[bytes, int]:
        with self.lock:
            return model_io.to_bytes(self.weights, self.version), self.version

    def status(self) -> dict:
        with self.lock:
            return {"version": self.version, "min_clients": self.s.min_clients,
                    "pending_hospitals": sorted(self.pending), "n_pending": len(self.pending),
                    "n_params": int(sum(a.size for a in self.weights)), "max_update_norm": self.s.max_update_norm,
                    "hf_repo": self.s.hf_repo_id or None, "last_publish": self.last_publish,
                    "latest": self.history[-1] if self.history else None}

    def submit(self, hospital_id: str, base_version: int, num_examples: int, arrays: list[np.ndarray]) -> dict:
        with self.lock:
            if base_version != self.version:
                raise FederationError(
                    f"Stale update: built on version {base_version} but the server is at version {self.version}. "
                    "Download the current model and train again.", 409, current_version=self.version)
            if not (1 <= num_examples <= self.s.max_examples):
                raise FederationError(f"num_examples must be between 1 and {self.s.max_examples}")
            try:
                model_io.validate_arrays(arrays)
            except ValueError as e:
                raise FederationError(f"Invalid model update: {e}")
            delta = float(np.linalg.norm(model_io.flat(arrays) - model_io.flat(self.weights)))
            if delta > self.s.max_update_norm:
                raise FederationError(
                    f"Update rejected: L2 distance from the global model is {delta:.2f}, above the allowed "
                    f"{self.s.max_update_norm:g} (possible poisoned or diverged update).")
            replaced = hospital_id in self.pending
            self.pending[hospital_id] = {"n": int(num_examples), "arrays": arrays, "delta_norm": delta}
            aggregated = None
            if len(self.pending) >= self.s.min_clients:
                aggregated = self._aggregate()
            return {"accepted": True, "replaced_previous": replaced, "delta_norm": round(delta, 4),
                    "aggregated": aggregated is not None,
                    "new_version": self.version, "pending": sorted(self.pending),
                    "needed": max(0, self.s.min_clients - len(self.pending)),
                    **({"round": aggregated} if aggregated else {})}

    def force_aggregate(self) -> dict:
        with self.lock:
            if not self.pending:
                raise FederationError("No pending updates to aggregate", 409)
            return {"aggregated": True, "round": self._aggregate(), "new_version": self.version}

    # ---- internals ----
    def _aggregate(self) -> dict:
        hospitals = sorted(self.pending)
        counts = [self.pending[h]["n"] for h in hospitals]
        self.weights = model_io.weighted_average([self.pending[h]["arrays"] for h in hospitals], counts)
        self.version += 1
        rec = self._record(self.version, hospitals, counts, source=f"FedAvg of {len(hospitals)} hospital update(s)")
        self.history.append(rec)
        self.pending = {}
        self._persist()
        threading.Thread(target=self._publish, daemon=True).start()
        return rec

    def _publish(self) -> None:
        if not hf_sync.configured(self.s):
            return
        with self.lock:
            snap = (model_io.to_bytes(self.weights, self.version), json.dumps(self.history, indent=1), self.version)
        self.last_publish = hf_sync.publish(self.s, *snap)


def _read_test():
    import pandas as pd
    df = pd.read_csv(config.CLIENT_DIR / "test.csv")
    return df[schema.FEATURES].to_numpy(np.float32), df[schema.TARGET].to_numpy(np.float32)
