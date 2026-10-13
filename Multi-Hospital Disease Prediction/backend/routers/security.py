"""Security & Privacy Analysis Dashboard data: round security, data egress, privacy status, attacks, demo."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from backend.routers.fl import current_user, get_fed, hospital_user
from backend.services.auth import User
from backend.services.federation import Federation, _read_test
from backend.services.security import REASON_LABEL, risk_assessment
from ml import model_io

router = APIRouter(prefix="/api/v1")


@router.get("/security/overview")
def overview(request: Request):
    """Public, aggregated view: counts and scores only (account names are masked)."""
    fed: Federation = request.app.state.federation
    accounts, sec, s = request.app.state.accounts, request.app.state.security, request.app.state.settings
    status = fed.status()
    totals = fed.totals()
    act = fed.hospital_activity()

    rounds = []
    for h in fed.history:
        b = h.get("security")
        if not b:
            continue
        rounds.append({
            "version": h["version"], "time": h["time"], "hospitals": len(h["participants"]), "received": b["received"],
            "accepted": b["accepted"], "rejected": b["rejected"], "rejected_by_reason": b["rejected_by_reason"],
            "dp_enabled": b["dp_enabled"], "aggregator": b["aggregator"], "raw_rows_transmitted": b["egress"]["raw_rows_transmitted"],
            "payload_kb": round(b["egress"]["payload_bytes_total"] / 1024, 1), "risk": b["risk"],
            "test_auc": (h.get("test_metrics") or {}).get("auc")})
    last = rounds[-1] if rounds else None
    if not last:
        dp_status, dp_text = "none", "No federated round has finished yet."
    elif last["dp_enabled"] == 0:
        dp_status, dp_text = "disabled", f"Differential privacy was NOT used in round v{last['version']}."
    elif last["dp_enabled"] == last["accepted"]:
        dp_status, dp_text = "enabled", f"All {last['accepted']} updates in round v{last['version']} declared differential privacy."
    else:
        dp_status, dp_text = "partial", f"{last['dp_enabled']} of {last['accepted']} updates in round v{last['version']} declared differential privacy."
    eps = [a["epsilon"] for a in act.values() if a["epsilon"] is not None]
    payloads = [r["payload_kb"] / max(r["accepted"], 1) for r in rounds if r["accepted"]]

    robust = s.aggregator == "median"
    controls = [
        {"name": "Authentication required", "status": "on", "detail": "Signed, expiring tokens; failed logins are throttled and logged."},
        {"name": "Update-size guard (poisoning)", "status": "on", "detail": f"Updates farther than {s.max_update_norm:g} (L2) from the global model are rejected."},
        {"name": "Stale / replay rejection", "status": "on", "detail": "Every update must be built on the current model version."},
        {"name": "Payload validation", "status": "on", "detail": "Fixed weight shapes, finite numbers, no extra fields, size limits."},
        {"name": "Robust aggregation (median)", "status": "on" if robust else "off",
         "detail": "Coordinate-wise median when 3+ hospitals report." if robust else "Plain FedAvg. Set AGGREGATOR=median to enable."},
        {"name": "Differential privacy", "status": "client-declared", "detail": "Hospitals add noise on their own machine; the server recomputes epsilon but cannot verify the noise."},
        {"name": "Secure aggregation", "status": "off", "detail": "Not part of the live protocol: the server sees each hospital's update."},
        {"name": "Transport encryption", "status": "on", "detail": "HTTPS when served from Render."}]

    return {
        "network": {"registered": accounts.count(), "participated": sum(1 for a in act.values() if a["accepted"]),
                    "active_now": status["n_pending"], "needed_per_round": status["min_clients"]},
        "model": {"version": status["version"], "aggregator": s.aggregator, "hf_repo": status["hf_repo"],
                  "last_publish": status["last_publish"]},
        "totals": totals,
        "round_in_progress": status["round"],
        "data_egress": {
            "raw_rows_transmitted": 0, "leaves_hospital": "Model weights (1,057 numbers), a row count and optional DP settings.",
            "stays_local": "Patient records. Training runs in the hospital's own browser or client program.",
            "avg_update_kb": round(sum(payloads) / len(payloads), 1) if payloads else None,
            "guarantees": ["The update endpoints accept only six fixed-shape weight arrays, a row count and DP settings; anything else is rejected.",
                           "There is no endpoint that accepts patient rows."]},
        "privacy": {"status": dp_status, "text": dp_text, "dp_rounds": sum(1 for r in rounds if r["dp_enabled"]),
                    "rounds": len(rounds), "max_epsilon": round(max(eps), 2) if eps else None, "delta": 1e-5,
                    "verified": False},
        "rounds": rounds, "controls": controls, "attacks": sec.attacks(), "events": sec.recent(30, public=True),
        "reasons": REASON_LABEL}


# ---------------------------------------------------------------------------------------------
# Demo: ordinary federated learning vs a protected training session
# ---------------------------------------------------------------------------------------------
def _unflat(vec: np.ndarray, shapes) -> list[np.ndarray]:
    out, i = [], 0
    for sh in shapes:
        n = int(np.prod(sh))
        out.append(vec[i:i + n].reshape(sh).astype(np.float32))
        i += n
    return out


def _wavg(vs, counts):
    w = np.array(counts, dtype=np.float64)
    return (np.stack(vs) * (w / w.sum())[:, None]).sum(0)


@router.get("/security/demo")
def demo(request: Request, seed: int = Query(0, ge=0, le=10_000)):
    """Same hospitals, same attacker, two sessions. Attack part: controlled simulation on synthetic updates around the
    real primary model. Privacy part: measured results from the experiments (membership inference, epsilon)."""
    s, store = request.app.state.settings, request.app.state.store
    base_arrays, _ = model_io.load(s.models_dir / "global_v0.npz")
    shapes = [a.shape for a in base_arrays]
    base = model_io.flat(base_arrays)
    P, cap = len(base), s.max_update_norm
    rng = np.random.default_rng(seed)
    rows = [900, 539, 359, 420]
    honest = [base + rng.normal(0, 0.012, P) for _ in rows]
    honest_avg = _wavg(honest, rows)
    Xte, _ = _read_test()
    p_ref = model_io.predict_proba(_unflat(honest_avg, shapes), Xte)
    attacker_rows = 50_000                                   # inflated claim: tries to dominate FedAvg

    attacks = []
    for key, name, norm in [("blatant", "Blatant poisoning (update far from the global model)", 80.0),
                            ("stealthy", "Stealthy poisoning (small enough to pass the size guard)", 15.0)]:
        d = rng.normal(size=P)
        evil = base + d / np.linalg.norm(d) * norm
        ups, cnt = honest + [evil], rows + [attacker_rows]
        ordinary = _wavg(ups, cnt)
        kept = [(u, c) for u, c in zip(ups, cnt) if np.linalg.norm(u - base) <= cap]
        rejected = len(ups) - len(kept)
        protected = (np.median(np.stack([u for u, _ in kept]), axis=0) if len(kept) >= 3
                     else _wavg([u for u, _ in kept], [c for _, c in kept]))

        def shift(v):
            return {"model_shift": round(float(np.linalg.norm(v - honest_avg)), 3),
                    "prediction_shift": round(float(np.mean(np.abs(model_io.predict_proba(_unflat(v, shapes), Xte) - p_ref))), 4)}
        attacks.append({"key": key, "name": name, "update_norm": norm,
                        "ordinary": {"attacker_accepted": True, "attacker_weight": round(attacker_rows / sum(cnt), 3), **shift(ordinary)},
                        "protected": {"attacker_accepted": rejected == 0, "rejected": rejected, "stage": "size guard" if rejected else "median aggregation",
                                      **shift(protected)}})

    mia, curve = store.get("attack.json"), store.get("privacy_curve.json")
    mia_plain = next(m for m in mia["models"] if m["label"].startswith("FedAvg (non-IID)"))
    mia_dp = next(m for m in mia["models"] if m.get("sigma") == 1.0)
    eps_dp = next(p["epsilon"] for p in curve["points"] if p["sigma"] == 1.0)

    risk_ord = risk_assessment(participants=5, dp_enabled=0, accepted_attack=True)
    risk_pro = risk_assessment(participants=4, dp_enabled=4, rejected={"norm": 1})
    blatant = attacks[0]
    table = [
        {"control": "Raw patient data leaves a hospital?", "ordinary": "No (weights only)", "protected": "No (weights only)", "better": None},
        {"control": "What the server sees from each hospital", "ordinary": "Exact trained weights", "protected": "Weights with DP noise (declared)", "better": "protected"},
        {"control": "Membership-inference attack (measured, AUC; 0.5 = no leak)", "ordinary": f"{mia_plain['auc']['mean']:.3f}", "protected": f"{mia_dp['auc']['mean']:.3f}", "better": "protected"},
        {"control": "Privacy budget epsilon (delta 1e-5)", "ordinary": "unbounded", "protected": f"{eps_dp:.1f}", "better": "protected"},
        {"control": "Blatant poisoned update", "ordinary": "Accepted", "protected": "Rejected by the size guard", "better": "protected"},
        {"control": "Prediction shift caused by the blatant attacker", "ordinary": f"{blatant['ordinary']['prediction_shift']:.3f}", "protected": f"{blatant['protected']['prediction_shift']:.3f}", "better": "protected"},
        {"control": "Stealthy poisoned update (passes the size guard)", "ordinary": f"Dominates (weight {attacks[1]['ordinary']['attacker_weight']:.0%})", "protected": "Neutralised by median aggregation", "better": "protected"},
        {"control": "Attacks detected and logged", "ordinary": "0 (silent)", "protected": "1 blocked + logged", "better": "protected"},
        {"control": "Round risk score", "ordinary": f"{risk_ord['score']} ({risk_ord['level']})", "protected": f"{risk_pro['score']} ({risk_pro['level']})", "better": "protected"}]
    return {"seed": seed, "hospitals": len(rows), "attacker_claimed_rows": attacker_rows, "size_guard": cap,
            "table": table, "attacks": attacks, "risk": {"ordinary": risk_ord, "protected": risk_pro},
            "privacy": {"membership_auc_ordinary": mia_plain["auc"]["mean"], "membership_auc_protected": mia_dp["auc"]["mean"],
                        "epsilon_protected": eps_dp, "sigma": 1.0},
            "method": ("Attack rows: 4 honest hospitals + 1 attacker around the real primary model; the attacker also claims 50,000 "
                       "rows to dominate FedAvg. Protected = size guard + median aggregation. Privacy rows are measured results "
                       "from the experiments (FedAvg vs FedAvg + DP, sigma 1.0). Risk uses the same scoring as live rounds.")}


# ---------------------------------------------------------------------------------------------
class PrivacyPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sigma: float = Field(..., ge=0.3, le=100)
    sample_rate: float = Field(..., gt=0, le=1)
    steps: int = Field(..., ge=1, le=1_000_000)


@router.post("/fl/privacy-preview")
def privacy_preview(body: PrivacyPreview, user: User = Depends(hospital_user), fed: Federation = Depends(get_fed)):
    """Epsilon of a planned DP training run, alone and cumulative with everything this hospital already released."""
    return fed.preview_epsilon(user.hospital_id, body.sigma, body.sample_rate, body.steps)
