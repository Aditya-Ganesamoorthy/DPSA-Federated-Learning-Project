"""Self-service registration, login, account safety, and the Security & Privacy dashboard data."""
import json

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.services.security import risk_assessment
from backend.tests.conftest import ADMIN_PW, PW, hospital_headers, login, make_settings, register


# ---------------- registration ----------------
def test_hospital_can_register_itself_and_use_the_token(client):
    r = register(client, "city-general", name="City General Hospital", parent="State Health Network")
    assert r.status_code == 201
    body = r.json()
    assert body["account"] == {"username": "city-general", "name": "City General Hospital", "parent": "State Health Network",
                               "joined": body["account"]["joined"], "status": "active"}
    h = {"Authorization": f"Bearer {body['access_token']}"}
    assert client.get("/api/v1/auth/me", headers=h).json() == {"username": "city-general", "role": "hospital"}
    login(client, "city-general", PW)                           # and can log in later with the same password


def test_registration_validation(client):
    assert register(client, "good-name").status_code == 201
    assert register(client, "good-name").status_code == 409                     # duplicate
    for bad_user in ("ab", "Has Space", "-bad", "admin", "root", "a" * 40):
        r = register(client, bad_user)
        assert r.status_code in (422, 409), (bad_user, r.status_code)
    # usernames are case-insensitive: stored lower-case, and a differently-cased duplicate conflicts
    assert register(client, "Mixed-Case").json()["username"] == "mixed-case"
    assert register(client, "MIXED-CASE").status_code == 409
    login(client, "MIXED-case", PW)
    assert register(client, "short-pass", password="short").status_code == 422
    assert register(client, "same-as-user", password="same-as-user").status_code == 422
    assert register(client, "tiny-name", name="ab").status_code == 422


def test_names_are_sanitised(client):
    r = register(client, "xss-hospital", name="<script>alert(1)</script> Clinic")
    assert r.status_code == 201
    assert "<" not in r.json()["account"]["name"] and ">" not in r.json()["account"]["name"]


def test_passwords_are_never_stored_and_state_is_safe_to_publish(app, client):
    register(client, "secret-keeper", password="My-Very-Secret-Password-9")
    blob = (app.state.settings.state_dir / "accounts.json").read_text(encoding="utf-8")
    assert "My-Very-Secret-Password-9" not in blob
    acc = json.loads(blob)["accounts"][0]
    assert set(acc) == {"username", "name", "parent", "joined", "status", "salt", "verifier"} and len(acc["verifier"]) == 64


def test_verifiers_are_peppered_with_the_server_secret(tmp_path):
    a = create_app(make_settings(tmp_path / "a", secret_key="secret-one-" * 4))
    b = create_app(make_settings(tmp_path / "b", secret_key="secret-two-" * 4))
    salt = "00" * 16
    assert a.state.accounts._verifier(PW, salt) != b.state.accounts._verifier(PW, salt)


def test_accounts_survive_restart_but_not_a_changed_secret(tmp_path):
    app1 = create_app(make_settings(tmp_path))
    register(TestClient(app1), "durable-one")
    ok = TestClient(create_app(make_settings(tmp_path)))
    login(ok, "durable-one", PW)
    rotated = TestClient(create_app(make_settings(tmp_path, secret_key="a-different-secret-key-0123456789")))
    assert rotated.post("/api/v1/auth/login", json={"username": "durable-one", "password": PW}).status_code == 401


def test_invite_code_gates_registration_when_configured(tmp_path):
    app = create_app(make_settings(tmp_path, invite_code="network-invite-42"))
    c = TestClient(app)
    assert register(c, "no-invite").status_code == 403
    assert register(c, "wrong-invite", invite="nope").status_code == 403
    assert register(c, "with-invite", invite="network-invite-42").status_code == 201
    assert app.state.security.counters["registration_blocked"] == 2


def test_registration_rate_limit(tmp_path):
    c = TestClient(create_app(make_settings(tmp_path, registrations_per_hour=3)))
    codes = [register(c, f"hospital-{i}x").status_code for i in range(5)]
    assert codes == [201, 201, 201, 429, 429]


# ---------------- login / throttling / suspension ----------------
def test_wrong_password_unknown_user_and_throttle(client):
    register(client, "careful-one")
    r = client.post("/api/v1/auth/login", json={"username": "careful-one", "password": "wrong-password"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"
    codes = [client.post("/api/v1/auth/login", json={"username": "careful-one", "password": "bad"}).status_code
             for _ in range(6)]
    assert codes == [401, 401, 401, 401, 429, 429]               # 5th failure locks the account
    # the right password is also refused while locked
    assert client.post("/api/v1/auth/login", json={"username": "careful-one", "password": PW}).status_code == 429


def test_brute_force_is_detected_and_logged_as_an_attack(app, client):
    for _ in range(5):
        client.post("/api/v1/auth/login", json={"username": "victim-user", "password": "guess"})
    types = [e["type"] for e in app.state.security.recent(20, public=False)]
    assert "login_failed" in types and "login_locked" in types
    assert any(a["label"] == "Brute-force login attempt" for a in app.state.security.attacks())


def test_token_tamper_and_expiry(app, client, tmp_path):
    h = hospital_headers(client, app, "token-user")
    tok = h["Authorization"].split()[1]
    bad = tok[:-3] + ("AAA" if not tok.endswith("AAA") else "BBB")
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {bad}"}).status_code == 401
    c2 = TestClient(create_app(make_settings(tmp_path / "short", token_ttl_s=-1)))
    hh = login(c2, "admin", ADMIN_PW)
    r = c2.get("/api/v1/auth/me", headers=hh)
    assert r.status_code == 401 and "expired" in r.json()["error"]["message"].lower()


def test_admin_can_suspend_and_restore_and_it_applies_immediately(app, client, admin_headers):
    h = hospital_headers(client, app, "to-suspend")
    assert client.get("/api/v1/fl/status", headers=h).status_code == 200
    assert client.post("/api/v1/admin/hospitals/to-suspend/suspend", headers=admin_headers).json()["status"] == "suspended"
    assert client.get("/api/v1/fl/status", headers=h).status_code == 403            # existing token stops working
    assert client.post("/api/v1/auth/login", json={"username": "to-suspend", "password": PW}).status_code == 403
    client.post("/api/v1/admin/hospitals/to-suspend/restore", headers=admin_headers)
    assert client.post("/api/v1/auth/login", json={"username": "to-suspend", "password": PW}).status_code == 200
    assert client.post("/api/v1/admin/hospitals/nobody-here/suspend", headers=admin_headers).status_code == 404


def test_admin_can_remove_a_hospital_and_its_pending_update(app, client, admin_headers):
    from client.api import ServerAPI
    api = ServerAPI("http://testserver", http=TestClient(app))
    api.register("Temp Hospital", "temp-hospital", PW)
    base, _ = api.download_model()
    api.submit(0, 20, [x + 0.01 for x in base])
    assert app.state.federation.status()["pending_hospitals"] == ["temp-hospital"]
    assert client.delete("/api/v1/admin/hospitals/temp-hospital", headers=admin_headers).json() == {"removed": "temp-hospital"}
    assert app.state.federation.status()["pending_hospitals"] == []
    assert client.post("/api/v1/auth/login", json={"username": "temp-hospital", "password": PW}).status_code == 401
    assert client.get("/api/v1/fl/status", headers={"Authorization": f"Bearer {api.token}"}).status_code == 403
    assert client.delete("/api/v1/admin/hospitals/temp-hospital", headers=admin_headers).status_code == 404
    assert register(client, "temp-hospital").status_code == 201                      # the name can be registered again
    h = hospital_headers(client, app, "someone-else")
    assert client.delete("/api/v1/admin/hospitals/someone-else", headers=h).status_code == 403


def test_role_separation(app, client, admin_headers):
    h = hospital_headers(client, app, "plain-user")
    assert client.post("/api/v1/admin/aggregate", headers=h).status_code == 403
    assert client.post("/api/v1/admin/hospitals/plain-user/suspend", headers=h).status_code == 403
    assert client.get("/api/v1/fl/model").status_code == 401
    r = client.post("/api/v1/fl/update-json", headers=admin_headers,
                    json={"base_version": 0, "num_examples": 5, "weights": [[0.0]] * 6})
    assert r.status_code == 403                                                     # admin cannot submit updates


def test_network_list_requires_login_and_shows_activity(app, client):
    assert client.get("/api/v1/fl/network").status_code == 401
    h = hospital_headers(client, app, "net-one")
    hospital_headers(client, app, "net-two")
    rows = client.get("/api/v1/fl/network", headers=h).json()["hospitals"]
    assert [r["username"] for r in rows] == ["net-one", "net-two"] and all("verifier" not in r and "salt" not in r for r in rows)


# ---------------- security dashboard data ----------------
def test_risk_scoring_is_transparent():
    clean = risk_assessment(participants=4, dp_enabled=4, rejected={})
    assert clean["score"] == 15 and clean["level"] == "low"                       # only "server sees updates"
    nodp = risk_assessment(participants=4, dp_enabled=0)
    assert nodp["score"] == 15 + 25 and nodp["level"] == "medium"
    attacked = risk_assessment(participants=2, dp_enabled=0, rejected={"norm": 1, "stale": 2}, auth_alerts=1)
    assert attacked["score"] == 15 + 25 + 10 + 5 + 10 + 10 and attacked["level"] == "high"
    assert sum(f["points"] for f in attacked["factors"]) == attacked["score"]
    assert risk_assessment(participants=5, dp_enabled=0, accepted_attack=True)["level"] == "high"
    assert risk_assessment(participants=10, dp_enabled=0, rejected={"norm": 99}, auth_alerts=9, accepted_attack=True)["score"] <= 100


def test_overview_is_public_aggregated_and_masks_names(app, client):
    from client.api import ServerAPI
    for _ in range(5):
        client.post("/api/v1/auth/login", json={"username": "private-name", "password": "guess"})
    o = client.get("/api/v1/security/overview").json()
    assert o["network"]["registered"] == 0 and o["data_egress"]["raw_rows_transmitted"] == 0
    assert o["privacy"]["status"] == "none" and o["privacy"]["verified"] is False
    assert any(c["name"] == "Secure aggregation" and c["status"] == "off" for c in o["controls"])
    assert "private-name" not in json.dumps(o) and any(e["actor"] == "pr***" for e in o["events"])
    assert any(a["label"] == "Brute-force login attempt" for a in o["attacks"])


def test_overview_after_real_rounds_tracks_received_rejected_participants_and_dp(tmp_path):
    from client.api import ApiError, ServerAPI
    app = create_app(make_settings(tmp_path, min_clients=2))
    c = TestClient(app)
    apis = []
    for hid in ("clinic-alpha", "clinic-beta"):
        a = ServerAPI("http://testserver", http=TestClient(app))
        a.register(f"Clinic {hid}", hid, PW)
        apis.append(a)
    base, _ = apis[0].download_model()
    with pytest.raises(ApiError):
        apis[0].submit(0, 30, [x + 1000 for x in base])                           # poisoning attempt
    apis[0].submit(0, 30, [x + 0.01 for x in base], {"dp_enabled": True, "sigma": 1.0, "sample_rate": 0.05, "steps": 100})
    out = apis[1].submit(0, 50, [x + 0.02 for x in base])
    assert out["aggregated"]
    o = c.get("/api/v1/security/overview").json()
    assert o["network"]["registered"] == 2 and o["network"]["participated"] == 2
    assert o["totals"] == {"received": 3, "accepted": 2, "rejected": 1, "rejected_by_reason": {"norm": 1}}
    r = o["rounds"][-1]
    assert (r["hospitals"], r["received"], r["accepted"], r["rejected"], r["dp_enabled"]) == (2, 3, 2, 1, 1)
    assert o["privacy"]["status"] == "partial" and o["privacy"]["max_epsilon"] is not None
    assert r["risk"]["level"] in ("medium", "high") and r["raw_rows_transmitted"] == 0
    assert any(a["label"] == "Suspected model poisoning" for a in o["attacks"])
    assert all(e["actor"] in ("", "unknown") or e["actor"].endswith("***") for e in o["events"])


def test_demo_shows_the_security_difference(client):
    d = client.get("/api/v1/security/demo?seed=1").json()
    blatant, stealthy = d["attacks"]
    assert blatant["ordinary"]["attacker_accepted"] and not blatant["protected"]["attacker_accepted"]
    assert blatant["ordinary"]["model_shift"] > 10 * blatant["protected"]["model_shift"]
    assert blatant["ordinary"]["prediction_shift"] > blatant["protected"]["prediction_shift"]
    assert stealthy["protected"]["attacker_accepted"] and stealthy["protected"]["stage"] == "median aggregation"
    assert stealthy["ordinary"]["model_shift"] > 3 * stealthy["protected"]["model_shift"]
    assert d["risk"]["ordinary"]["level"] == "high" and d["risk"]["protected"]["level"] == "low"
    assert d["privacy"]["membership_auc_protected"] < d["privacy"]["membership_auc_ordinary"]
    assert client.get("/api/v1/security/demo?seed=1").json() == d                 # deterministic
    assert len(d["table"]) >= 8
