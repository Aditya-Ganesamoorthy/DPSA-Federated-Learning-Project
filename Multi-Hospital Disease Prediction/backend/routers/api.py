from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Query, Request

from ml import config, model_io, schema

router = APIRouter(prefix="/api/v1")

SCENARIO_RE = re.compile(r"^(iid|noniid|dp_[0-9.]+)$")


def _store(request: Request):
    return request.app.state.store


@router.get("/health")
def health(request: Request):
    return {"status": "ok", "results_loaded": _store(request).all_loaded(),
            "global_model_version": request.app.state.federation.version}


@router.get("/features")
def features():
    """Everything the in-browser trainer needs, from one source of truth (ml/schema.py and ml/config.py)."""
    return {
        "features": schema.FEATURES, "binary_features": schema.BINARY_FEATURES, "target": schema.TARGET,
        "gender_map": schema.GENDER_MAP, "age_scale": schema.AGE_SCALE, "age_range": list(schema.AGE_RANGE),
        "survey_codes": {"no": 1, "yes": 2}, "target_map": schema.TARGET_MAP,
        "array_shapes": [list(s) for s in model_io.expected_shapes()],
        "training": {"lr": config.LR, "momentum": config.MOMENTUM, "batch_size": config.BATCH_SIZE,
                     "local_epochs": config.LOCAL_EPOCHS, "clip_norm": config.CLIP_NORM, "delta": 1e-5},
        "limits": {"max_rows": 50_000, "max_file_mb": 8, "min_rows": 40}}


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


@router.get("/results/eda")
def eda(request: Request):
    return _store(request).get("eda.json")


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


@router.get("/results/roc")
def roc(request: Request, model: str = Query("federated", pattern=r"^[a-z_]{1,24}$")):
    store = _store(request)
    if model not in store.rocs():
        raise HTTPException(404, detail=f"Unknown model '{model}'. Available: {store.rocs()}")
    return store.get(f"roc_{model}.json")
