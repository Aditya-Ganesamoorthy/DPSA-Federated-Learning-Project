"""The live federated-learning coordinator (the 'centralized server model' hospitals download from).

Protocol (synchronous rounds, FedAvg):
  1. A hospital downloads the current global model (version k).
  2. It trains locally on its own data (raw rows never leave) and uploads the new weights, its number of
     training examples, the base version k it started from, and (optionally) the differential-privacy settings it used.
  3. When MIN_CLIENTS distinct hospitals have submitted for version k (or the administrator forces it), the server
     computes w_{k+1} = sum_h (n_h / N) * w_h, evaluates it on the shared test file, saves it as version k+1 together
     with a security record of the round, and publishes it to Hugging Face (if configured).

Safeguards: authenticated hospitals only; updates must be built on the CURRENT version (409 otherwise); shape / finite-value
validation; a cap on the L2 norm of (update - global) as a basic poisoning guard; at most one pending update per hospital
per round; every accepted or rejected update becomes a security event. DP settings are client-declared (the server cannot
verify that noise was really added) but the server recomputes epsilon from them and composes it across rounds.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

from backend.services import hf_sync
from backend.services.security import REASON_LABEL, REJECT_REASONS, SecurityLog, risk_assessment
from backend.settings import Settings
from ml import config, model_io, rdp, schema

log = logging.getLogger("federation")
DP_DELTA = 1e-5


class FederationError(Exception):
    def __init__(self, message: str, status: int = 422, reason: str = "invalid", **extra):
        super().__init__(message)
        self.status, self.reason, self.extra = status, reason, extra


class Federation:
    def __init__(self, settings: Settings, security_log: SecurityLog):
        self.s = settings
        self.sec = security_log
        self.lock = threading.RLock()
        self.dir = Path(settings.state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pending: dict[str, dict] = {}
        self.history: list[dict] = []
        self.round = self._new_round()
        self.last_publish: dict = {"status": "never" if hf_sync.configured(settings) else "not configured"}
        self.weights: list[np.ndarray]
        self.version: int
        self._xte = self._yte = None
        self._load_state()

    # ---- state ----
    @staticmethod
    def _new_round() -> dict:
        return {"opened": time.time(), "received": 0, "rejected": Counter(), "rejected_events": [], "extra_dp": {}}

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
            self.history = [self._record(0, [], [], note="initial primary model (trained across 3 simulated hospitals)")]
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
            self._xte, self._yte = _read_test()
        return model_io.metrics(self._yte, model_io.predict_proba(self.weights, self._xte))

    def _record(self, version: int, hospitals: list[str], counts: list[int], note: str = "",
                security: dict | None = None) -> dict:
        rec = {"version": version, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "participants": hospitals, "examples": counts, "total_examples": int(sum(counts)),
               "test_metrics": self._evaluate(), "note": note}
        if security is not None:
            rec["security"] = security
        return rec

    # ---- read API ----
    def model_bytes(self) -> tuple[bytes, int]:
        with self.lock:
            return model_io.to_bytes(self.weights, self.version), self.version

    def model_arrays(self) -> tuple[list[np.ndarray], int]:
        with self.lock:
            return [a.copy() for a in self.weights], self.version

    def current_risk(self) -> dict | None:
        with self.lock:
            if not self.pending and not self.round["received"]:
                return None
            dp = sum(1 for p in self.pending.values() if (p["privacy"] or {}).get("dp_enabled"))
            return risk_assessment(participants=len(self.pending), dp_enabled=dp, rejected=dict(self.round["rejected"]),
                                   auth_alerts=self.sec.count_since(["login_locked"], self.round["opened"]))

    def status(self) -> dict:
        with self.lock:
            return {"version": self.version, "min_clients": self.s.min_clients,
                    "pending_hospitals": sorted(self.pending), "n_pending": len(self.pending),
                    "n_params": int(sum(a.size for a in self.weights)), "max_update_norm": self.s.max_update_norm,
                    "aggregator": self.s.aggregator, "hf_repo": self.s.hf_repo_id or None,
                    "last_publish": self.last_publish, "latest": self.history[-1] if self.history else None,
                    "round": {"received": self.round["received"], "rejected": dict(self.round["rejected"]),
                              "opened": self.round["opened"], "risk": self.current_risk()}}

    def totals(self) -> dict:
        """Received / accepted / rejected updates across all finished rounds plus the one in progress."""
        with self.lock:
            rec = acc = 0
            rej: Counter = Counter()
            for h in self.history:
                sec = h.get("security")
                if sec:
                    rec += sec["received"]
                    acc += sec["accepted"]
                    rej.update(sec["rejected_by_reason"])
            rec += self.round["received"]
            rej.update(self.round["rejected"])
            acc += len(self.pending)
            return {"received": rec, "accepted": acc, "rejected": int(sum(rej.values())),
                    "rejected_by_reason": dict(rej)}

    @staticmethod
    def _dp_seg(p: dict | None):
        return (p["sigma"], p["sample_rate"], p["steps"]) if p and p.get("dp_enabled") else None

    def _segments(self, hospital_id: str) -> list[tuple[float, float, int]]:
        """Every DP release a hospital made: accepted, still pending, replaced or rejected (the server saw them all,
        so all of them spend privacy budget)."""
        segs = []
        for h in self.history:
            sec = h.get("security") or {}
            per = (sec.get("per_hospital") or {}).get(hospital_id, {})
            for p in [per.get("dp")] + list(per.get("extra_dp", [])) + list((sec.get("extra_dp_other") or {}).get(hospital_id, [])):
                if self._dp_seg(p):
                    segs.append(self._dp_seg(p))
        pend = (self.pending.get(hospital_id) or {}).get("privacy")
        for p in [pend] + list(self.round["extra_dp"].get(hospital_id, [])):
            if self._dp_seg(p):
                segs.append(self._dp_seg(p))
        return segs

    def preview_epsilon(self, hospital_id: str, sigma: float, sample_rate: float, steps: int) -> dict:
        with self.lock:
            new = (sigma, sample_rate, steps)
            return {"this_update": round(rdp.epsilon([new], DP_DELTA), 3),
                    "cumulative_after": round(rdp.epsilon(self._segments(hospital_id) + [new], DP_DELTA), 3),
                    "delta": DP_DELTA}

    def hospital_activity(self) -> dict[str, dict]:
        """Per hospital: accepted updates, rejected updates, last round, cumulative epsilon of declared DP."""
        with self.lock:
            out: dict[str, dict] = {}

            def slot(h):
                return out.setdefault(h, {"accepted": 0, "rejected": 0, "last_version": None, "dp_updates": 0,
                                          "epsilon": None, "pending": False})
            for rec in self.history:
                sec = rec.get("security") or {}
                for h in rec["participants"]:
                    s = slot(h)
                    s["accepted"] += 1
                    s["last_version"] = rec["version"]
                    if ((sec.get("per_hospital") or {}).get(h, {}).get("dp") or {}).get("dp_enabled"):
                        s["dp_updates"] += 1
                for ev in sec.get("rejected_events", []):
                    slot(ev["hospital"])["rejected"] += 1
            for h in self.pending:
                slot(h)["pending"] = True
            for ev in self.round["rejected_events"]:
                slot(ev["hospital"])["rejected"] += 1
            for h, s in out.items():
                segs = self._segments(h)
                if segs:
                    s["epsilon"] = round(rdp.epsilon(segs, DP_DELTA), 3)
            return out

    # ---- write API ----
    def reject(self, hospital_id: str, reason: str, detail: str, privacy: dict | None = None) -> None:
        with self.lock:
            try:
                priv = self._check_privacy(privacy)
            except FederationError:
                priv = None
            if priv and priv.get("dp_enabled"):                 # the noisy update reached the server: budget is spent
                self.round["extra_dp"].setdefault(hospital_id, []).append(priv)
            self.round["received"] += 1
            self.round["rejected"][reason] += 1
            if len(self.round["rejected_events"]) < 50:
                self.round["rejected_events"].append({"hospital": hospital_id, "reason": reason,
                                                       "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        self.sec.add(REJECT_REASONS[reason], hospital_id, detail)

    @staticmethod
    def _check_privacy(privacy: dict | None) -> dict | None:
        if not privacy or not privacy.get("dp_enabled"):
            return {"dp_enabled": False}
        try:
            sigma, q, steps = float(privacy["sigma"]), float(privacy["sample_rate"]), int(privacy["steps"])
        except (KeyError, TypeError, ValueError):
            raise FederationError("Privacy settings need sigma, sample_rate and steps")
        if not (0.3 <= sigma <= 100) or not (0 < q <= 1) or not (1 <= steps <= 1_000_000):
            raise FederationError("Privacy settings out of range (sigma 0.3-100, sample_rate in (0,1], steps 1-1e6)")
        return {"dp_enabled": True, "sigma": sigma, "sample_rate": q, "steps": steps}

    def submit(self, hospital_id: str, base_version: int, num_examples: int, arrays: list[np.ndarray],
               privacy: dict | None = None, payload_bytes: int = 0) -> dict:
        with self.lock:
            try:
                accepted = self._validate(hospital_id, base_version, num_examples, arrays, privacy)
            except FederationError as e:
                self.reject(hospital_id, e.reason, str(e), privacy)
                raise
            self.round["received"] += 1
            replaced = hospital_id in self.pending
            if replaced and self.pending[hospital_id]["privacy"].get("dp_enabled"):
                self.round["extra_dp"].setdefault(hospital_id, []).append(self.pending[hospital_id]["privacy"])
            self.pending[hospital_id] = {"n": int(num_examples), "arrays": arrays, "delta_norm": accepted["delta"],
                                         "privacy": accepted["privacy"], "payload_bytes": int(payload_bytes)}
            dp_note = "with differential privacy" if accepted["privacy"]["dp_enabled"] else "without differential privacy"
            self.sec.add("update_accepted", hospital_id, f"{num_examples} rows, {dp_note}, v{base_version}")
            aggregated = None
            if len(self.pending) >= self.s.min_clients:
                aggregated = self._aggregate()
            return {"accepted": True, "replaced_previous": replaced, "delta_norm": round(accepted["delta"], 4),
                    "aggregated": aggregated is not None, "new_version": self.version,
                    "pending": sorted(self.pending), "needed": max(0, self.s.min_clients - len(self.pending)),
                    "dp_enabled": accepted["privacy"]["dp_enabled"],
                    **({"round": aggregated} if aggregated else {})}

    def _validate(self, hospital_id, base_version, num_examples, arrays, privacy) -> dict:
        if base_version != self.version:
            raise FederationError(
                f"Stale update: built on version {base_version} but the server is at version {self.version}. "
                "Download the current model and train again.", 409, "stale", current_version=self.version)
        if not (1 <= num_examples <= self.s.max_examples):
            raise FederationError(f"num_examples must be between 1 and {self.s.max_examples}")
        try:
            model_io.validate_arrays(arrays)
        except ValueError as e:
            raise FederationError(f"Invalid model update: {e}")
        priv = self._check_privacy(privacy)
        delta = float(np.linalg.norm(model_io.flat(arrays) - model_io.flat(self.weights)))
        if delta > self.s.max_update_norm:
            raise FederationError(
                f"Update rejected: L2 distance from the global model is {delta:.2f}, above the allowed "
                f"{self.s.max_update_norm:g} (possible poisoned or diverged update).", 422, "norm")
        return {"delta": delta, "privacy": priv}

    def force_aggregate(self) -> dict:
        with self.lock:
            if not self.pending:
                raise FederationError("No pending updates to aggregate", 409)
            return {"aggregated": True, "round": self._aggregate(), "new_version": self.version}

    # ---- internals ----
    def _aggregate(self) -> dict:
        hospitals = sorted(self.pending)
        counts = [self.pending[h]["n"] for h in hospitals]
        updates = [self.pending[h]["arrays"] for h in hospitals]
        used = "fedavg"
        if self.s.aggregator == "median" and len(updates) >= 3:
            # robust to a minority of poisoned updates (measured in results/extras.json); ignores sample counts
            self.weights = [np.median(np.stack([u[i] for u in updates]), axis=0).astype(np.float32)
                            for i in range(len(updates[0]))]
            used = "coordinate-wise median"
        else:
            self.weights = model_io.weighted_average(updates, counts)
        security = self._security_block(hospitals, used)
        self.version += 1
        rec = self._record(self.version, hospitals, counts, security=security,
                           note=f"{used} of {len(hospitals)} hospital update(s)")
        self.history.append(rec)
        self.pending = {}
        self.round = self._new_round()
        self._persist()
        self.sec.add("round_aggregated", "", f"v{self.version}: {used} of {len(hospitals)} update(s), "
                                              f"risk {security['risk']['score']} ({security['risk']['level']})")
        threading.Thread(target=self._publish, daemon=True).start()
        return rec

    def _security_block(self, hospitals: list[str], aggregator: str) -> dict:
        rejected = dict(self.round["rejected"])
        dp_enabled = sum(1 for h in hospitals if (self.pending[h]["privacy"] or {}).get("dp_enabled"))
        risk = risk_assessment(participants=len(hospitals), dp_enabled=dp_enabled, rejected=rejected,
                               auth_alerts=self.sec.count_since(["login_locked"], self.round["opened"]))
        return {
            "received": self.round["received"], "accepted": len(hospitals), "rejected": int(sum(rejected.values())),
            "rejected_by_reason": rejected, "rejected_events": list(self.round["rejected_events"]),
            "dp_enabled": dp_enabled, "aggregator": aggregator,
            "per_hospital": {h: {"rows": self.pending[h]["n"], "payload_bytes": self.pending[h]["payload_bytes"],
                                 "delta_norm": round(self.pending[h]["delta_norm"], 4), "dp": self.pending[h]["privacy"],
                                 "extra_dp": list(self.round["extra_dp"].get(h, []))}
                             for h in hospitals},
            "extra_dp_other": {h: list(v) for h, v in self.round["extra_dp"].items() if h not in hospitals},
            "egress": {"raw_rows_transmitted": 0, "payload_bytes_total": int(sum(self.pending[h]["payload_bytes"] for h in hospitals)),
                       "content": "6 weight arrays + a row count + optional DP settings"},
            "risk": risk,
        }

    def _publish(self) -> None:
        if not hf_sync.configured(self.s):
            return
        with self.lock:
            snap = (model_io.to_bytes(self.weights, self.version), json.dumps(self.history, indent=1), self.version)
        self.last_publish = hf_sync.publish(self.s, *snap)
        self.sec.add("publish_ok" if self.last_publish.get("status") == "ok" else "publish_error", "",
                     f"v{snap[2]} -> Hugging Face: {self.last_publish.get('status')}")


def _read_test():
    import pandas as pd
    df = pd.read_csv(config.CLIENT_DIR / "test.csv")
    return df[schema.FEATURES].to_numpy(np.float32), df[schema.TARGET].to_numpy(np.float32)


__all__ = ["Federation", "FederationError", "REASON_LABEL"]
