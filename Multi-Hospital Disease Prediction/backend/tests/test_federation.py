"""End-to-end tests of the live federation: login, download, local training, update, FedAvg, versioning."""
import base64
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.services.auth import Auth, AuthError, slugify
from backend.tests.conftest import ADMIN_PW, hospital_headers, login, make_settings
from client.api import ApiError, ServerAPI
from client.core import load_hospital_csv, local_update
from ml import config, model_io

HOSPITALS = ["hospital-one", "hospital-two", "hospital-three"]


def _api(app, hid):
    http = TestClient(app)
    api = ServerAPI("http://testserver", http=http)
    api.login(hid, app.state.auth.hospital_password(hid))
    return api


def _train(api, k, seed=0):
    """A hospital's full local step using its own CSV (the project's non-IID hospital files)."""
    arrays, version = api.download_model()
    X, y = load_hospital_csv(config.CLIENT_DIR / f"noniid_hospital_{k}.csv")
    res = local_update(arrays, X, y, epochs=1, seed=seed, hospital_id=HOSPITALS[k - 1])
    return version, res


# ---------------- authentication ----------------
def test_login_wrong_password_and_unknown_user(client):
    for u, p in [("hospital-one", "wrong"), ("nobody-here", "x"), ("admin", "wrong")]:
        r = client.post("/api/v1/auth/login", json={"username": u, "password": p})
        assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"


def test_login_throttle_after_repeated_failures(client):
    codes = [client.post("/api/v1/auth/login", json={"username": "hospital-one", "password": "bad"}).status_code
             for _ in range(7)]
    assert codes[:5] == [401] * 5 and codes[5] == 429


def test_token_tamper_and_expiry(app, client, tmp_path):
    h = hospital_headers(client, app, "hospital-one")
    assert client.get("/api/v1/auth/me", headers=h).json() == {"username": "hospital-one", "role": "hospital"}
    tok = h["Authorization"].split()[1]
    bad = tok[:-3] + ("AAA" if not tok.endswith("AAA") else "BBB")
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {bad}"}).status_code == 401
    short = create_app(make_settings(tmp_path / "short", token_ttl_s=-1))
    c2 = TestClient(short)
    hh = login(c2, "admin", ADMIN_PW)
    r = c2.get("/api/v1/auth/me", headers=hh)
    assert r.status_code == 401 and "expired" in r.json()["error"]["message"].lower()


def test_hospital_credentials_are_derived_and_secret_dependent(tmp_path):
    a = Auth(make_settings(tmp_path / "a", secret_key="secret-one" * 4))
    b = Auth(make_settings(tmp_path / "b", secret_key="secret-two" * 4))
    assert a.hospital_password("hospital-one") == a.hospital_password("hospital-one")
    assert a.hospital_password("hospital-one") != b.hospital_password("hospital-one")
    assert a.hospital_password("hospital-one") != a.hospital_password("hospital-two")


def test_revoked_hospital_cannot_login(tmp_path):
    app = create_app(make_settings(tmp_path, revoked_hospitals=("hospital-one",)))
    c = TestClient(app)
    r = c.post("/api/v1/auth/login", json={"username": "hospital-one",
                                           "password": app.state.auth.hospital_password("hospital-one")})
    assert r.status_code == 401


def test_admin_creates_hospital_credentials_that_work(app, client, admin_headers):
    r = client.post("/api/v1/admin/hospitals", json={"name": "St Mary's Hospital, Chennai"}, headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["hospital_id"] == "st-mary-s-hospital-chennai"
    login(client, body["username"], body["password"])


def test_admin_only_and_hospital_only_routes(app, client, admin_headers):
    h = hospital_headers(client, app, "hospital-one")
    assert client.post("/api/v1/admin/hospitals", json={"name": "Some Hospital"}, headers=h).status_code == 403
    assert client.post("/api/v1/admin/aggregate", headers=h).status_code == 403
    assert client.get("/api/v1/fl/model").status_code == 401
    # admin cannot submit hospital updates
    r = client.post("/api/v1/fl/update", json={"base_version": 0, "num_examples": 5, "npz_b64": "A" * 20},
                    headers=admin_headers)
    assert r.status_code == 403


def test_slugify_rules():
    assert slugify("A B C") == "a-b-c"
    for bad in ("", "ab", "admin", "!!!"):
        with pytest.raises(AuthError):
            slugify(bad)


# ---------------- federation ----------------
def test_full_round_equals_manual_weighted_average(app):
    """3 hospitals download v0, train on their own rows, upload; server FedAvg == manual n_k/N average."""
    apis = [_api(app, h) for h in HOSPITALS]
    results = []
    for k, api in enumerate(apis, 1):
        version, res = _train(api, k, seed=k)
        assert version == 0
        results.append(res)
    outs = [api.submit(0, res["num_examples"], res["arrays"]) for api, res in zip(apis, results)]
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


def test_weights_follow_sample_counts(app):
    """A hospital with far more examples dominates the average (FedAvg weighting)."""
    a, b, c = [_api(app, h) for h in HOSPITALS]
    base, _ = a.download_model()
    up = [[x + d for x in base] for d in (0.0, 0.5, 0.5)]
    a.submit(0, 900, up[0])
    b.submit(0, 100, up[1])
    c.submit(0, 100, up[2])
    got, _ = a.download_model()
    np.testing.assert_allclose(got[0], base[0] + 0.5 * (200 / 1100), atol=1e-5)


def test_stale_update_rejected_with_current_version(app):
    apis = [_api(app, h) for h in HOSPITALS]
    base, _ = apis[0].download_model()
    for api in apis:
        api.submit(0, 50, [x + 0.01 for x in base])           # completes round 0 -> v1
    with pytest.raises(ApiError) as e:
        apis[0].submit(0, 50, [x + 0.01 for x in base])        # built on v0 but server is at v1
    assert e.value.status == 409 and e.value.body["error"]["current_version"] == 1


def test_invalid_updates_rejected(app):
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
        api.submit(0, 0, base)
    assert e.value.status == 422
    with pytest.raises(ApiError) as e:
        api._call("POST", "/fl/update", json={"base_version": 0, "num_examples": 10, "npz_b64": "!!notbase64!!"})
    assert e.value.status == 422


def _bytes_unchecked(arrays):
    import io
    buf = io.BytesIO()
    np.savez(buf, version=np.array(0), **{f"p{i}": np.asarray(a, dtype=np.float32) for i, a in enumerate(arrays)})
    return buf.getvalue()


def test_resubmission_replaces_previous_update(app):
    api = _api(app, "hospital-one")
    base, _ = api.download_model()
    first = api.submit(0, 10, [x + 0.01 for x in base])
    second = api.submit(0, 10, [x + 0.02 for x in base])
    assert not first["replaced_previous"] and second["replaced_previous"]
    assert second["pending"] == ["hospital-one"] and not second["aggregated"]


def test_admin_force_aggregate_and_global_model_used_by_predict(app, client, admin_headers):
    api = _api(app, "hospital-one")
    base, _ = api.download_model()
    api.submit(0, 25, [x + 0.05 for x in base])
    out = client.post("/api/v1/admin/aggregate", headers=admin_headers).json()
    assert out["aggregated"] and out["new_version"] == 1
    patient = {"gender": "F", "age": 50, **{c: False for c in config_features()}}
    h = hospital_headers(client, app, "hospital-one")
    r = client.post("/api/v1/predict", json={"model": "global", "patient": patient}, headers=h).json()
    assert r["model_version"] == 1
    assert client.post("/api/v1/admin/aggregate", headers=admin_headers).status_code == 409   # nothing pending


def config_features():
    from ml import schema
    return schema.BINARY_FEATURES


def test_state_survives_restart(tmp_path):
    s = make_settings(tmp_path, min_clients=1)
    app1 = create_app(s)
    api = _api(app1, "hospital-one")
    base, _ = api.download_model()
    api.submit(0, 40, [x + 0.03 for x in base])             # min_clients=1 -> aggregates immediately
    app2 = create_app(make_settings(tmp_path, min_clients=1))   # new process, same state dir
    assert app2.state.federation.version == 1
    np.testing.assert_allclose(app2.state.federation.weights[0], app1.state.federation.weights[0])
    assert len(app2.state.federation.history) == 2


def test_concurrent_submissions_aggregate_exactly_once(tmp_path):
    import threading
    app = create_app(make_settings(tmp_path, min_clients=4))
    ids = [f"hospital-{i}" for i in range(4)]
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
    apis[2].submit(0, 100, [x + 15.0 / np.sqrt(1057) for x in base])      # large (but norm-guard-passing) outlier
    got, _ = apis[0].download_model()
    np.testing.assert_allclose(got[0], base[0] + 0.02, atol=1e-5)          # median picks the middle update
    assert apis[0].status()["aggregator"] == "median"
