import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.tests.conftest import hospital_headers, make_settings
from ml import config, model as M, schema

PATIENT = {"gender": "M", "age": 60, **{c: True for c in schema.BINARY_FEATURES}}


@pytest.fixture(scope="module", autouse=True)
def results_ready():
    need = ["comparison.json", "privacy_curve.json", "attack.json", "communication.json", "run_meta.json",
            "client_summary.json", "preprocess_report.json"]
    if not all((config.RESULTS_DIR / n).exists() for n in need):
        pytest.skip("run `python -m ml.run_all` first")


@pytest.fixture
def auth(client, app):
    return hospital_headers(client, app, "test-hospital-a")


def test_health(client):
    r = client.get("/api/v1/health").json()
    assert r["status"] == "ok" and r["results_loaded"] and "global" in r["models_loaded"]
    assert r["global_model_version"] == 0


@pytest.mark.parametrize("path", ["/api/v1/meta", "/api/v1/dataset/summary", "/api/v1/clients", "/api/v1/features",
                                  "/api/v1/results/comparison", "/api/v1/results/rounds?scenario=noniid",
                                  "/api/v1/results/rounds?scenario=iid", "/api/v1/results/privacy-curve",
                                  "/api/v1/results/attack", "/api/v1/results/communication",
                                  "/api/v1/results/secagg-check", "/api/v1/results/sanity",
                                  "/api/v1/results/roc?model=federated", "/api/v1/secagg/demo?seed=3"])
def test_public_read_endpoints_ok_and_valid_json(client, path):
    r = client.get(path)
    assert r.status_code == 200, r.text
    assert "NaN" not in r.text and "Infinity" not in r.text
    json.loads(r.text)


def test_privacy_curve_no_dp_is_null(client):
    pts = client.get("/api/v1/results/privacy-curve").json()["points"]
    assert pts[0]["epsilon"] is None and all(p["epsilon"] is not None for p in pts[1:])


def test_rounds_scenario_allowlist(client):
    assert client.get("/api/v1/results/rounds?scenario=../../etc/passwd").status_code == 404
    assert client.get("/api/v1/results/rounds?scenario=nope").json()["error"]["code"] == "not_found"


def test_features_endpoint_matches_schema(client):
    f = client.get("/api/v1/features").json()
    assert [x["name"] for x in f["fields"]] == ["gender", "age", *schema.BINARY_FEATURES]
    assert set(f["models"]) == {"global", "centralized", "federated", "federated_dp"}


def test_secagg_demo_cancels_and_dropout_fails(client):
    ok = client.get("/api/v1/secagg/demo?seed=1").json()
    assert ok["masks_cancelled"] and ok["max_abs_error"] < 1e-4
    bad = client.get("/api/v1/secagg/demo?seed=1&drop=1").json()
    assert not bad["masks_cancelled"] and bad["max_abs_error"] > 1
    assert client.get("/api/v1/secagg/demo?seed=1").json() == ok


@pytest.mark.parametrize("model", ["global", "centralized", "federated", "federated_dp"])
def test_predict_range_and_disclaimer(client, auth, model):
    r = client.post("/api/v1/predict", json={"model": model, "patient": PATIENT}, headers=auth)
    assert r.status_code == 200, r.text
    b = r.json()
    assert 0 <= b["probability"] <= 1 and b["label"] in (0, 1) and "not medical advice" in b["disclaimer"]


def test_predict_requires_login(client):
    r = client.post("/api/v1/predict", json={"model": "global", "patient": PATIENT})
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"


@pytest.mark.parametrize("bad", [
    {"model": "federated", "patient": {**PATIENT, "age": 500}},
    {"model": "federated", "patient": {**PATIENT, "extra": 1}},
    {"model": "nope", "patient": PATIENT},
    {"model": "federated", "patient": {k: v for k, v in PATIENT.items() if k != "smoking"}},
    {"model": "federated", "patient": {**PATIENT, "gender": "X"}},
])
def test_predict_validation_422(client, auth, bad):
    r = client.post("/api/v1/predict", json=bad, headers=auth)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error" and r.json()["error"]["fields"]


def test_serving_parity_with_training_pipeline(client, auth):
    """API (numpy) prediction == training-time (torch) pipeline prediction on the same raw rows."""
    from ml import preprocess
    raw = preprocess.load_raw().drop_duplicates().reset_index(drop=True)
    net = M.make_model("mlp", 0)
    net.load_state_dict(torch.load(config.MODELS_DIR / "federated.pt", weights_only=True))
    enc = schema.encode_frame(raw.head(10))
    for i in range(10):
        row = raw.iloc[i]
        patient = {"gender": row["gender"], "age": int(row["age"]),
                   **{c: int(row[c]) == 2 for c in schema.BINARY_FEATURES}}
        api = client.post("/api/v1/predict", json={"model": "federated", "patient": patient},
                          headers=auth).json()["probability"]
        train = float(M.predict_proba(net, enc.iloc[[i]][schema.FEATURES].to_numpy(np.float32))[0])
        assert abs(api - train) < 1e-5


def test_missing_results_return_503_not_stack_trace(tmp_path):
    empty = TestClient(create_app(make_settings(tmp_path, results_dir=tmp_path / "nothing")))
    r = empty.get("/api/v1/results/comparison")
    assert r.status_code == 503 and r.json()["error"]["code"] == "results_missing"
    assert "run" in r.json()["error"]["message"].lower()
    assert empty.get("/api/v1/health").json()["results_loaded"] is False


def test_cors_allows_vite_origin(client):
    r = client.get("/api/v1/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
