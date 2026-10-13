"""Security event log and round risk scoring (the data behind the Security & Privacy dashboard)."""
from __future__ import annotations

import json
import threading
import time
from collections import Counter, deque
from pathlib import Path

# event type -> (severity, attack label or None)
EVENT_TYPES: dict[str, tuple[str, str | None]] = {
    "update_accepted": ("info", None),
    "round_aggregated": ("info", None),
    "registration": ("info", None),
    "hospital_restored": ("info", None),
    "publish_ok": ("info", None),
    "publish_error": ("medium", None),
    "login_failed": ("low", None),
    "login_blocked": ("medium", None),
    "registration_blocked": ("medium", None),
    "hospital_suspended": ("medium", None),
    "update_rejected_stale": ("medium", "Replay / stale update"),
    "update_rejected_invalid": ("medium", "Malformed update / tampering"),
    "update_rejected_norm": ("high", "Suspected model poisoning"),
    "login_locked": ("high", "Brute-force login attempt"),
}
REJECT_REASONS = {"norm": "update_rejected_norm", "stale": "update_rejected_stale", "invalid": "update_rejected_invalid"}
REASON_LABEL = {"norm": "Suspected model poisoning (update too far from the global model)",
                "stale": "Stale or replayed update (built on an old model version)",
                "invalid": "Malformed update (bad shape, NaN, bad privacy parameters or payload)"}


def mask_actor(actor: str) -> str:
    """Public views never show full account names."""
    if not actor:
        return "unknown"
    return actor[:2] + "***"


class SecurityLog:
    def __init__(self, state_dir: Path, max_events: int = 500):
        self.path = Path(state_dir) / "events.json"
        self.lock = threading.Lock()
        self.events: deque[dict] = deque(maxlen=max_events)
        self.counters: Counter = Counter()
        self._seq = 0
        self._last_write = 0.0
        if self.path.exists():
            try:
                for e in json.loads(self.path.read_text(encoding="utf-8")):
                    self.events.append(e)
                    self.counters[e["type"]] += 1
                    self._seq = max(self._seq, e["id"])
            except (ValueError, KeyError, OSError):
                pass

    def add(self, etype: str, actor: str = "", detail: str = "") -> dict:
        severity, attack = EVENT_TYPES[etype]
        with self.lock:
            self._seq += 1
            ev = {"id": self._seq, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ts": time.time(),
                  "type": etype, "severity": severity, "attack": attack, "actor": actor, "detail": detail[:240]}
            self.events.append(ev)
            self.counters[etype] += 1
            if time.time() - self._last_write > 1.0:      # throttle disk writes during floods
                self._flush()
        return ev

    def _flush(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(list(self.events)), encoding="utf-8")
            self._last_write = time.time()
        except OSError:
            pass

    def flush(self) -> None:
        with self.lock:
            self._flush()

    def count_since(self, types: list[str], since_ts: float) -> int:
        with self.lock:
            return sum(1 for e in self.events if e["type"] in types and e["ts"] >= since_ts)

    def recent(self, n: int = 40, public: bool = True) -> list[dict]:
        with self.lock:
            items = list(self.events)[-n:][::-1]
        out = []
        for e in items:
            d = {k: v for k, v in e.items() if k != "ts"}
            if public:
                d["actor"] = mask_actor(e["actor"])
            out.append(d)
        return out

    def attacks(self) -> list[dict]:
        with self.lock:
            items = list(self.events)
        grouped: dict[str, dict] = {}
        for e in items:
            if e["attack"]:
                g = grouped.setdefault(e["attack"], {"label": e["attack"], "type": e["type"], "count": 0,
                                                     "severity": e["severity"], "last_time": e["time"]})
                g["count"] += 1
                g["last_time"] = e["time"]
        return sorted(grouped.values(), key=lambda g: -g["count"])


def risk_assessment(*, participants: int, dp_enabled: int, rejected: dict | None = None, auth_alerts: int = 0,
                    secure_agg: bool = False, accepted_attack: bool = False) -> dict:
    """Transparent 0-100 residual-risk score for one training round. Every point is listed with its reason.

    visibility  +15  the server sees each hospital's plain update (secure aggregation is only a simulation here)
    no DP       +25  scaled by the share of updates that did not declare differential privacy
    attacker    +10  per kind of attack that was blocked this round (an active adversary is present)
    brute force +10  login lockouts during the round
    small cohort+10  fewer than 3 hospitals: median aggregation cannot be used and each update is easy to isolate
    accepted    +45  a malicious update got through (only known in the controlled demo)
    """
    rejected = rejected or {}
    factors: list[dict] = []

    def add(label: str, pts: int) -> None:
        if pts > 0:
            factors.append({"label": label, "points": int(pts)})

    if not secure_agg:
        add("Server sees each hospital's plain model update (no secure aggregation)", 15)
    if participants > 0:
        no_dp = participants - dp_enabled
        add(f"{no_dp} of {participants} updates without differential privacy", round(25 * no_dp / participants))
    if rejected.get("norm"):
        add(f"Poisoning attempt blocked ({rejected['norm']} update(s) rejected)", 10)
    if rejected.get("stale"):
        add(f"Stale/replayed update blocked ({rejected['stale']})", 5)
    if rejected.get("invalid"):
        add(f"Malformed update blocked ({rejected['invalid']})", 5)
    if auth_alerts:
        add(f"Brute-force login lockout(s): {auth_alerts}", 10)
    if 0 < participants < 3:
        add("Fewer than 3 hospitals: robust aggregation unavailable", 10)
    if accepted_attack:
        add("A malicious update was accepted into the global model", 45)
    score = min(100, sum(f["points"] for f in factors))
    return {"score": score, "level": "low" if score < 30 else "medium" if score < 60 else "high", "factors": factors}
