import json

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.tests.conftest import make_settings
from ml import config, schema


@pytest.fixture(scope="module", autouse=True)
def results_ready():
    need = ["comparison.json", "privacy_curve.json", "attack.json", "communication.json", "run_meta.json",
            "client_summary.json", "preprocess_report.json", "eda.json"]
    if not all((config.RESULTS_DIR / n).exists() for n in need):
        pytest.skip("run `python -m ml.run_all` and `python -m ml.eda` first")


def test_health(client):
    r = client.get("/api/v1/health").json()
    assert r["status"] == "ok" and r["results_loaded"] and r["global_model_version"] == 0


@pytest.mark.parametrize("path", ["/api/v1/meta", "/api/v1/dataset/summary", "/api/v1/clients", "/api/v1/features",
                                  "/api/v1/results/eda", "/api/v1/results/comparison",
                                  "/api/v1/results/rounds?scenario=noniid", "/api/v1/results/rounds?scenario=iid",
                                  "/api/v1/results/privacy-curve", "/api/v1/results/attack",
                                  "/api/v1/results/communication", "/api/v1/results/sanity",
                                  "/api/v1/results/roc?model=federated", "/api/v1/security/overview",
                                  "/api/v1/security/demo?seed=3"])
def test_public_read_endpoints_ok_and_valid_json(client, path):
    r = client.get(path)
    assert r.status_code == 200, r.text
    def reject(const):                      # a numeric NaN/Infinity would be invalid JSON for browsers
        raise ValueError(f"invalid JSON constant {const}")
    json.loads(r.text, parse_constant=reject)


def test_removed_endpoints_are_gone(client):
    for method, path in [("get", "/api/v1/secagg/demo"), ("get", "/api/v1/results/secagg-check"),
                         ("post", "/api/v1/predict")]:
        assert getattr(client, method)(path).status_code in (404, 405)


def test_privacy_curve_no_dp_is_null(client):
    pts = client.get("/api/v1/results/privacy-curve").json()["points"]
    assert pts[0]["epsilon"] is None and all(p["epsilon"] is not None for p in pts[1:])


def test_rounds_scenario_allowlist(client):
    assert client.get("/api/v1/results/rounds?scenario=../../etc/passwd").status_code == 404
    assert client.get("/api/v1/results/rounds?scenario=nope").json()["error"]["code"] == "not_found"


def test_features_endpoint_drives_the_browser_trainer(client):
    f = client.get("/api/v1/features").json()
    assert f["features"] == schema.FEATURES and f["binary_features"] == schema.BINARY_FEATURES
    assert f["array_shapes"] == [[32, 15], [32], [16, 32], [16], [1, 16], [1]]
    assert f["gender_map"] == {"F": 0, "M": 1} and f["age_scale"] == 100 and f["survey_codes"] == {"no": 1, "yes": 2}
    assert f["training"]["clip_norm"] == config.CLIP_NORM and f["training"]["batch_size"] == config.BATCH_SIZE


def test_eda_is_aggregate_only_and_consistent(client):
    e = client.get("/api/v1/results/eda").json()
    assert e["rows"] == e["positives"] + e["negatives"] and 0.4 < e["prevalence"] < 0.6
    assert len(e["features"]) == 14 and e["max_abs_correlation"] < 0.1
    assert sum(e["age"]["histogram"]["positive"]) + sum(e["age"]["histogram"]["negative"]) == e["rows"]
    names = e["correlation_matrix"]["names"]
    assert len(e["correlation_matrix"]["values"]) == len(names) == 16


def test_missing_results_return_503_not_stack_trace(tmp_path):
    empty = TestClient(create_app(make_settings(tmp_path, results_dir=tmp_path / "nothing")))
    r = empty.get("/api/v1/results/comparison")
    assert r.status_code == 503 and r.json()["error"]["code"] == "results_missing"
    assert "run" in r.json()["error"]["message"].lower()
    assert empty.get("/api/v1/health").json()["results_loaded"] is False


def test_cors_allows_vite_origin(client):
    r = client.get("/api/v1/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
