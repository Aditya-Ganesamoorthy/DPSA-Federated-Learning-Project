from __future__ import annotations

import re

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request

from backend.schemas import DISCLAIMER, MODEL_NOTE, PredictRequest, PredictResponse
from backend.services.predictor import ModelsMissing
from ml import schema, secagg

router = APIRouter(prefix="/api/v1")

SCENARIO_RE = re.compile(r"^(iid|noniid|dp_[0-9.]+)$")


def _store(request: Request):
    return request.app.state.store


@router.get("/health")
def health(request: Request):
    store, pred = _store(request), request.app.state.predictor
    return {"status": "ok", "results_loaded": store.all_loaded(), "models_loaded": pred.available()}


@router.get("/features")
def features():
    """Feature list the Predict form is generated from (single source of truth: ml/schema.py)."""
    return {"fields": [
        {"name": "gender", "type": "enum", "options": ["M", "F"]},
        {"name": "age", "type": "int", "min": schema.AGE_RANGE[0], "max": schema.AGE_RANGE[1]},
        *[{"name": c, "type": "bool"} for c in schema.BINARY_FEATURES]],
        "models": ["centralized", "federated", "federated_dp"]}


@router.get("/meta")
def meta(request: Request):
    return _store(request).get("run_meta.json")


@router.get("/dataset/summary")
def dataset_summary(request: Request):
    s = _store(request)
    pre, cl = s.get("preprocess_report.json"), s.get("client_summary.json")
    return {"preprocess": pre, "split_rows": cl["split_rows"], "k_anonymity_pooled": cl["k_anonymity_pooled"],
            "partition_config": cl["config"]}


@router.get("/clients")
def clients(request: Request):
    return _store(request).get("client_summary.json")["scenarios"]


@router.get("/results/comparison")
def comparison(request: Request):
    return _store(request).get("comparison.json")


@router.get("/results/rounds")
def rounds(request: Request, scenario: str = Query("noniid", max_length=24)):
    store = _store(request)
    if not SCENARIO_RE.match(scenario) or scenario not in store.scenarios():
        raise HTTPException(404, detail=f"Unknown scenario '{scenario}'. Available: {store.scenarios()}")
    return store.get(f"rounds_{scenario}.json")


@router.get("/results/privacy-curve")
def privacy_curve(request: Request):
    return _store(request).get("privacy_curve.json")


@router.get("/results/attack")
def attack(request: Request):
    return _store(request).get("attack.json")


@router.get("/results/communication")
def communication(request: Request):
    return _store(request).get("communication.json")


@router.get("/results/sanity")
def sanity(request: Request):
    return _store(request).get("sanity_planted_signal.json")


@router.get("/results/secagg-check")
def secagg_check(request: Request):
    return _store(request).get("secagg_check.json")


@router.get("/results/roc")
def roc(request: Request, model: str = Query("federated", pattern=r"^[a-z_]{1,24}$")):
    store = _store(request)
    if model not in store.rocs():
        raise HTTPException(404, detail=f"Unknown model '{model}'. Available: {store.rocs()}")
    return store.get(f"roc_{model}.json")


@router.get("/secagg/demo")
def secagg_demo(seed: int = Query(0, ge=0, le=10_000), drop: int | None = Query(None, ge=0, le=2)):
    """Cheap deterministic toy demo (no training): 3 hospitals, 4-number updates, pairwise masks."""
    rng = np.random.default_rng(seed)
    counts = [50, 30, 20]
    updates = [np.round(rng.normal(size=4), 2) for _ in counts]
    r = secagg.secure_weighted_average(updates, counts, base_seed=seed, drop=drop)
    weights = [n / sum(counts) for n in counts]
    return {
        "counts": counts, "weights": weights, "modulus": "2^32", "fixed_point_bits": secagg.config.SECAGG_SCALE_BITS,
        "updates": [u.tolist() for u in updates],
        "masked": [[int(v) for v in m] for m in r["masked"]],
        "dropped_client": drop,
        "server_aggregate": [round(float(v), 6) for v in r["aggregate"]],
        "plain_fedavg": [round(float(v), 6) for v in r["plain_float"]],
        "max_abs_error": r["float_max_abs_error"],
        "masks_cancelled": bool(r["int_max_abs_diff"] == 0),
        "explanation": ("All masks cancel: the server learns only the weighted average." if drop is None else
                        "A client dropped out, so its partners' masks are not cancelled and the aggregate is garbage. "
                        "Real protocols recover with secret sharing."),
    }


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    # NOTE: request bodies are deliberately never logged.
    pred = request.app.state.predictor
    try:
        out = pred.predict(req.model, req.patient.model_dump())
    except ModelsMissing as e:
        raise HTTPException(503, detail=str(e))
    return PredictResponse(**out, model_used=req.model, disclaimer=DISCLAIMER, model_note=MODEL_NOTE)
