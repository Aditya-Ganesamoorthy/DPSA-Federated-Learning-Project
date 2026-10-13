"""End-to-end tests of the live federation: registration, download, local training, update, FedAvg, versioning."""
import base64
import io
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.tests.conftest import PW, login, make_settings
from client.api import ApiError, ServerAPI
from client.core import load_hospital_csv, local_update
from ml import config, model_io

HOSPITALS = ["hospital-one", "hospital-two", "hospital-three"]


def _api(app, hid):
    api = ServerAPI("http://testserver", http=TestClient(app))
    try:
        api.register(f"Hospital {hid}", hid, PW)
    except ApiError as e:
        assert e.status == 409
        api.login(hid, PW)
    return api


def _train(api, k, seed=0):
    """A hospital's full local step using its own CSV (the project's non-IID hospital files)."""
    arrays, version = api.download_model()
    X, y = load_hospital_csv(config.CLIENT_DIR / f"noniid_hospital_{k}.csv")
    res = local_update(arrays, X, y, epochs=1, seed=seed, hospital_id=HOSPITALS[k - 1])
    return version, res


def _bytes_unchecked(arrays):
    buf = io.BytesIO()
    np.savez(buf, version=np.array(0), **{f"p{i}": np.asarray(a, dtype=np.float32) for i, a in enumerate(arrays)})
    return buf.getvalue()


def test_full_round_equals_manual_weighted_average(app):
    """3 hospitals download v0, train on their own rows, upload; server FedAvg == manual n_k/N average."""
    apis = [_api(app, h) for h in HOSPITALS]
    results = []
    for k, api in enumerate(apis, 1):
        version, res = _train(api, k, seed=k)
        assert version == 0
        results.append(res)
    outs = [api.submit(0, res["num_examples"], res["arrays"], res["privacy"]) for api, res in zip(apis, results)]
    assert [o["aggregated"] for o in outs] == [False, False, True]
    assert outs[1]["needed"] == 1 and outs[2]["new_version"] == 1
    expected = model_io.weighted_average([r["arrays"] for r in results], [r["num_examples"] for r in results])
    got, version = apis[0].download_model()
    assert version == 1
    for a, b in zip(got, expected):
        np.testing.assert_allclose(a, b, atol=1e-6)
    s = apis[0].status()
    assert s["version"] == 1 and s["n_pending"] == 0
    hist = apis[0]._call("GET", "/fl/history").json()["history"]
    assert [h["version"] for h in hist] == [0, 1]
    assert hist[1]["participants"] == sorted(HOSPITALS) and hist[1]["test_metrics"]["auc"] is not None
    sec = hist[1]["security"]
    assert sec["received"] == 3 and sec["accepted"] == 3 and sec["rejected"] == 0
    assert sec["egress"]["raw_rows_transmitted"] == 0 and sec["risk"]["level"] in ("low", "medium", "high")


def test_weights_follow_sample_counts(app):
    a, b, c = [_api(app, h) for h in HOSPITALS]
    base, _ = a.download_model()
    up = [[x + d for x in base] for d in (0.0, 0.5, 0.5)]
    a.submit(0, 900, up[0])
    b.submit(0, 100, up[1])
    c.submit(0, 100, up[2])
    got, _ = a.download_model()
    np.testing.assert_allclose(got[0], base[0] + 0.5 * (200 / 1100), atol=1e-5)


def test_stale_update_rejected_with_current_version_and_logged(app):
    apis = [_api(app, h) for h in HOSPITALS]
    base, _ = apis[0].download_model()
    for api in apis:
        api.submit(0, 50, [x + 0.01 for x in base])           # completes round 0 -> v1
    with pytest.raises(ApiError) as e:
        apis[0].submit(0, 50, [x + 0.01 for x in base])        # built on v0 but server is at v1
    assert e.value.status == 409 and e.value.body["error"]["current_version"] == 1
    types = [ev["type"] for ev in app.state.security.recent(20, public=False)]
    assert "update_rejected_stale" in types
    assert app.state.federation.totals()["rejected_by_reason"] == {"stale": 1}


def test_invalid_updates_rejected_and_counted(app):
    api = _api(app, "hospital-one")
    base, _ = api.download_model()
    for arrays, msg in [
        (base[:-1], "missing arrays"),
        ([np.zeros((3, 3), np.float32)] + base[1:], "shape"),
        ([np.full_like(base[0], np.nan)] + base[1:], "NaN"),
        ([x + 1000 for x in base], "L2 distance"),             # poisoning / divergence guard
    ]:
        with pytest.raises(ApiError) as e:
            api._call("POST", "/fl/update", json={"base_version": 0, "num_examples": 10,
                      "npz_b64": base64.b64encode(_bytes_unchecked(arrays)).decode()})
        assert e.value.status == 422 and msg in str(e.value), (msg, str(e.value))
    with pytest.raises(ApiError) as e:
        api.submit(0, 5_000_000, base)
    assert e.value.status == 422
    with pytest.raises(ApiError) as e:
        api._call("POST", "/fl/update", json={"base_version": 0, "num_examples": 10, "npz_b64": "!!notbase64!!"})
    assert e.value.status == 422
    reasons = app.state.federation.totals()["rejected_by_reason"]
    assert reasons["norm"] == 1 and reasons["invalid"] >= 4 and "stale" not in reasons
    atk = {a["label"]: a["count"] for a in app.state.security.attacks()}
    assert atk["Suspected model poisoning"] == 1


def test_resubmission_replaces_previous_update(app):
    api = _api(app, "hospital-one")
    base, _ = api.download_model()
    first = api.submit(0, 10, [x + 0.01 for x in base])
    second = api.submit(0, 10, [x + 0.02 for x in base])
    assert not first["replaced_previous"] and second["replaced_previous"]
    assert second["pending"] == ["hospital-one"] and not second["aggregated"]


def test_admin_force_aggregate(app, client, admin_headers):
    api = _api(app, "hospital-one")
    base, _ = api.download_model()
    api.submit(0, 25, [x + 0.05 for x in base])
    out = client.post("/api/v1/admin/aggregate", headers=admin_headers).json()
    assert out["aggregated"] and out["new_version"] == 1
    assert client.post("/api/v1/admin/aggregate", headers=admin_headers).status_code == 409   # nothing pending


def test_state_survives_restart(tmp_path):
    s = make_settings(tmp_path, min_clients=1)
    app1 = create_app(s)
    api = _api(app1, "hospital-one")
    base, _ = api.download_model()
    api.submit(0, 40, [x + 0.03 for x in base])             # min_clients=1 -> aggregates immediately
    app2 = create_app(make_settings(tmp_path, min_clients=1))   # new process, same state dir
    assert app2.state.federation.version == 1
    np.testing.assert_allclose(app2.state.federation.weights[0], app1.state.federation.weights[0])
    assert len(app2.state.federation.history) == 2 and app2.state.federation.history[1]["security"]["accepted"] == 1


def test_concurrent_submissions_aggregate_exactly_once(tmp_path):
    app = create_app(make_settings(tmp_path, min_clients=4))
    ids = [f"hospital-{i}x" for i in range(4)]
    apis = [_api(app, h) for h in ids]
    base, _ = apis[0].download_model()
    res = []
    ts = [threading.Thread(target=lambda a=a: res.append(a.submit(0, 30, [x + 0.01 for x in base]))) for a in apis]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sum(r["aggregated"] for r in res) == 1 and app.state.federation.version == 1


def test_publish_called_when_configured(tmp_path, monkeypatch):
    from backend.services import hf_sync
    calls = []
    monkeypatch.setattr(hf_sync, "publish", lambda s, npz, hist, version: calls.append(version) or {"status": "ok"})
    monkeypatch.setattr(hf_sync, "restore", lambda s, d: False)
    monkeypatch.setattr(hf_sync, "restore_file", lambda s, name, dest: False)
    monkeypatch.setattr(hf_sync, "publish_file", lambda s, p, data, msg: {"status": "ok"})
    app = create_app(make_settings(tmp_path, min_clients=1, hf_repo_id="user/repo", hf_token="hf_x"))
    api = _api(app, "hospital-one")
    base, _ = api.download_model()
    api.submit(0, 40, [x + 0.03 for x in base])
    for _ in range(50):
        if 1 in calls:
            break
        time.sleep(0.05)
    assert 0 in calls and 1 in calls          # v0 published at startup, v1 after the round


def test_median_aggregator_ignores_a_poisoned_update(tmp_path):
    app = create_app(make_settings(tmp_path, aggregator="median"))
    apis = [_api(app, h) for h in HOSPITALS]
    base, _ = apis[0].download_model()
    apis[0].submit(0, 100, [x + 0.01 for x in base])
    apis[1].submit(0, 100, [x + 0.02 for x in base])
    apis[2].submit(0, 100, [x + 15.0 / np.sqrt(1057) for x in base])      # large (but guard-passing) outlier
    got, _ = apis[0].download_model()
    np.testing.assert_allclose(got[0], base[0] + 0.02, atol=1e-5)          # median picks the middle update
    assert apis[0].status()["aggregator"] == "median"
    assert apis[0]._call("GET", "/fl/history").json()["history"][-1]["security"]["aggregator"] == "coordinate-wise median"


# ---------------- browser (JSON) update path ----------------
def _json_body(arrays, n=40, version=0, privacy=None):
    body = {"base_version": version, "num_examples": n, "weights": [np.asarray(a).ravel().astype(float).tolist() for a in arrays]}
    if privacy:
        body["privacy"] = privacy
    return body


def test_json_update_equals_npz_update(tmp_path):
    new = None
    results = []
    for mode in ("npz", "json"):
        app = create_app(make_settings(tmp_path / mode, min_clients=1))
        api = _api(app, "hospital-one")
        base, _ = api.download_model()
        new = [x + 0.02 for x in base]
        if mode == "npz":
            api.submit(0, 40, new)
        else:
            assert api._call("POST", "/fl/update-json", json=_json_body(new)).json()["aggregated"]
        results.append(app.state.federation.weights)
    for a, b in zip(*results):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_model_json_endpoint_matches_npz(app):
    api = _api(app, "hospital-one")
    j = api._call("GET", "/fl/model-json").json()
    arrays, v = api.download_model()
    assert j["version"] == v == 0
    for flat, a in zip(j["weights"], arrays):
        np.testing.assert_allclose(np.array(flat, dtype=np.float32), a.ravel(), atol=1e-6)


def test_json_update_rejects_wrong_sizes_and_counts_it(app):
    api = _api(app, "hospital-one")
    base, _ = api.download_model()
    bad = _json_body(base)
    bad["weights"][0] = bad["weights"][0][:-1]
    with pytest.raises(ApiError) as e:
        api._call("POST", "/fl/update-json", json=bad)
    assert e.value.status == 422
    assert app.state.federation.totals()["rejected_by_reason"] == {"invalid": 1}
    with pytest.raises(ApiError) as e:
        api._call("POST", "/fl/update-json", json={**_json_body(base), "extra": 1})
    assert e.value.status == 422


# ---------------- differential-privacy ledger ----------------
DP = {"dp_enabled": True, "sigma": 1.0, "sample_rate": 0.05, "steps": 200}


def test_dp_epsilon_composes_across_rounds_and_counts_discarded_releases(tmp_path):
    from ml import rdp
    app = create_app(make_settings(tmp_path, min_clients=1))
    api = _api(app, "hospital-one")
    fed = app.state.federation

    base, v = api.download_model()
    out = api._call("POST", "/fl/update-json", json=_json_body([x + 0.01 for x in base], privacy=DP)).json()
    assert out["aggregated"] and out["dp_enabled"]
    eps1 = fed.hospital_activity()["hospital-one"]["epsilon"]
    assert eps1 == pytest.approx(rdp.epsilon([(1.0, 0.05, 200)]), rel=1e-3)

    base, v = api.download_model()
    api._call("POST", "/fl/update-json", json=_json_body([x + 0.01 for x in base], n=40, version=v, privacy=DP))
    eps2 = fed.hospital_activity()["hospital-one"]["epsilon"]
    assert eps2 > eps1                                         # composition across rounds
    assert eps2 == pytest.approx(rdp.epsilon([(1.0, 0.05, 200)] * 2), rel=1e-3)

    # a DP update that the server rejects (stale) was still released: it must still spend budget
    with pytest.raises(ApiError):
        api._call("POST", "/fl/update-json", json=_json_body([x + 0.01 for x in base], version=0, privacy=DP))
    eps3 = fed.hospital_activity()["hospital-one"]["epsilon"]
    assert eps3 == pytest.approx(rdp.epsilon([(1.0, 0.05, 200)] * 3), rel=1e-3)
    assert eps3 > eps2


def test_privacy_preview_endpoint(app):
    api = _api(app, "hospital-one")
    r = api._call("POST", "/fl/privacy-preview", json={"sigma": 1.0, "sample_rate": 0.05, "steps": 200}).json()
    assert r["this_update"] > 0 and r["cumulative_after"] == pytest.approx(r["this_update"], rel=1e-6)
    with pytest.raises(ApiError):
        api._call("POST", "/fl/privacy-preview", json={"sigma": 0.01, "sample_rate": 0.05, "steps": 200})


def test_invalid_privacy_settings_rejected(app):
    api = _api(app, "hospital-one")
    base, _ = api.download_model()
    for bad in [{"dp_enabled": True, "sigma": 1.0, "sample_rate": 0.05}, {"dp_enabled": True, "sigma": 0.01, "sample_rate": 0.05, "steps": 5}]:
        with pytest.raises(ApiError) as e:
            api._call("POST", "/fl/update-json", json=_json_body(base, privacy=bad))
        assert e.value.status == 422
