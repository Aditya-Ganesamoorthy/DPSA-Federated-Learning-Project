"""Black-box smoke test of a running server (local or Render). Needs only numpy + the standard library.

  python scripts/smoke_server.py --url https://<app>.onrender.com --admin-password <pw>
  python scripts/smoke_server.py --url http://localhost:8000 --admin-password <pw> --mutating

Default (safe for a live server): public dashboards, authentication, self-registration of two throw-away hospitals,
download, validation and rejection paths. The throw-away hospitals are DELETED at the end.
--mutating additionally submits two fake updates and completes a round (adds a synthetic version to the history):
use it only on a disposable server.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

import numpy as np


def call(base, method, path, token=None, body=None, raw=False, timeout=120, api=True):
    req = urllib.request.Request(base + ("/api/v1" if api else "") + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            return r.status, (data if raw else json.loads(data)), dict(r.headers)
    except urllib.error.HTTPError as e:
        data = e.read()
        try:
            return e.code, json.loads(data), dict(e.headers)
        except ValueError:
            return e.code, data, dict(e.headers)


def npz_to_arrays(data: bytes):
    z = np.load(io.BytesIO(data), allow_pickle=False)
    return [z[f"p{i}"] for i in range(6)], int(z["version"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--admin-password", required=True)
    ap.add_argument("--admin-username", default="admin")
    ap.add_argument("--invite-code", default="", help="needed only if the server requires an invite code")
    ap.add_argument("--mutating", action="store_true")
    a = ap.parse_args()
    base, ok = a.url.rstrip("/"), True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))

    t0 = time.time()
    s, h, _ = call(base, "GET", "/health", timeout=180)       # first call may wake a sleeping free instance
    check("health reachable", s == 200 and h.get("status") == "ok", s)
    print(f"     (responded in {time.time() - t0:.1f}s; global model v{h.get('global_model_version')}, registration {h.get('registration')})")
    s, page, _ = call(base, "GET", "/", raw=True, api=False)
    check("dashboard served at /", s == 200 and b'<div id="root">' in page, s)

    # ---- public dashboards
    s, o, _ = call(base, "GET", "/security/overview")
    check("security overview (public)", s == 200 and o["data_egress"]["raw_rows_transmitted"] == 0 and "network" in o and "rounds" in o, s)
    s, d, _ = call(base, "GET", "/security/demo?seed=1")
    check("ordinary-vs-protected demo", s == 200 and not d["attacks"][0]["protected"]["attacker_accepted"] and d["attacks"][0]["ordinary"]["attacker_accepted"], s)
    s, e, _ = call(base, "GET", "/results/eda")
    check("EDA results", s == 200 and e["rows"] > 0, s)
    s, f, _ = call(base, "GET", "/features")
    check("features spec for the browser trainer", s == 200 and len(f["features"]) == 15 and len(f["array_shapes"]) == 6, s)
    for gone in ("/secagg/demo", "/predict"):
        s, _, _ = call(base, "GET", gone)
        check(f"removed endpoint {gone} is gone", s in (404, 405), s)

    # ---- authentication
    s, _, _ = call(base, "POST", "/auth/login", body={"username": a.admin_username, "password": "wrong-password-xyz"})
    check("wrong admin password rejected", s == 401, s)
    s, b, _ = call(base, "POST", "/auth/login", body={"username": a.admin_username, "password": a.admin_password})
    check("admin login", s == 200 and b.get("role") == "admin", s)
    admin = b.get("access_token")
    for path in ("/fl/status", "/fl/model", "/fl/network", "/fl/history"):
        s, _, _ = call(base, "GET", path)
        check(f"{path} requires login", s == 401, s)

    # ---- hospitals register themselves
    names = [f"smoke-{uuid.uuid4().hex[:6]}" for _ in range(2)]
    pw = "smoke-pass-" + uuid.uuid4().hex[:6]
    toks = []
    for n in names:
        s, r, _ = call(base, "POST", "/auth/register", body={"hospital_name": f"Smoke Test {n}", "parent_network": "", "username": n,
                                                              "password": pw, "invite_code": a.invite_code})
        check(f"hospital {n} registers itself", s == 201 and r.get("access_token"), (s, r))
        toks.append(r.get("access_token"))
    s, r, _ = call(base, "POST", "/auth/register", body={"hospital_name": "Dup", "username": names[0], "password": pw, "invite_code": a.invite_code})
    check("duplicate username refused (409)", s == 409, s)
    s, r, _ = call(base, "POST", "/auth/register", body={"hospital_name": "Weak", "username": "weak-" + names[0][6:], "password": "short", "invite_code": a.invite_code})
    check("weak password refused (422)", s == 422, s)
    s, r, _ = call(base, "POST", "/auth/login", body={"username": names[0], "password": pw})
    check("login with the chosen password", s == 200 and r.get("role") == "hospital", s)
    s, r, _ = call(base, "POST", "/auth/login", body={"username": names[1], "password": "not-the-password"})
    check("wrong hospital password refused", s == 401, s)

    # ---- federation access
    s, st, _ = call(base, "GET", "/fl/status", token=toks[0])
    check("hospital sees federation status", s == 200 and "version" in st, s)
    version = st.get("version", 0)
    s, data, hdr = call(base, "GET", "/fl/model", token=toks[0], raw=True)
    arrays, v = npz_to_arrays(data) if s == 200 else (None, None)
    check("hospital downloads the global model (npz)", s == 200 and v == version and len(arrays) == 6, s)
    s, mj, _ = call(base, "GET", "/fl/model-json", token=toks[0])
    check("browser model download (json) matches", s == 200 and mj["version"] == version and np.allclose(np.array(mj["weights"][0], dtype=np.float32), arrays[0].ravel(), atol=1e-6), s)
    s, net, _ = call(base, "GET", "/fl/network", token=toks[0])
    check("network list shows the new hospitals", s == 200 and all(any(h_["username"] == n for h_ in net["hospitals"]) for n in names), s)
    s, _, _ = call(base, "POST", "/admin/aggregate", token=toks[0])
    check("hospital cannot use admin routes", s == 403, s)
    s, _, _ = call(base, "DELETE", f"/admin/hospitals/{names[0]}", token=toks[1])
    check("hospital cannot delete accounts", s == 403, s)
    s, pr, _ = call(base, "POST", "/fl/privacy-preview", token=toks[0], body={"sigma": 1.0, "sample_rate": 0.05, "steps": 200})
    check("privacy budget preview", s == 200 and pr["this_update"] > 0, s)

    flat = lambda arrs: [np.asarray(x).ravel().astype(float).tolist() for x in arrs]
    s, r, _ = call(base, "POST", "/fl/update-json", token=toks[0], body={"base_version": version + 99, "num_examples": 5, "weights": flat(arrays)})
    check("stale update rejected (409)", s == 409, s)
    s, r, _ = call(base, "POST", "/fl/update-json", token=toks[0], body={"base_version": version, "num_examples": 5, "weights": flat([x + 1000 for x in arrays])})
    check("oversized (poisoned) update rejected (422)", s == 422, s)
    s, r, _ = call(base, "POST", "/fl/update-json", token=toks[0], body={"base_version": version, "num_examples": 5, "weights": [[0.0]] * 6})
    check("wrongly shaped update rejected (422)", s == 422, s)
    s, o2, _ = call(base, "GET", "/security/overview")
    check("rejections show up on the security dashboard", s == 200 and o2["totals"]["rejected"] >= 3 and any(a_["label"] == "Suspected model poisoning" for a_ in o2["attacks"]), o2.get("totals"))

    # ---- admin controls
    s, r, _ = call(base, "POST", f"/admin/hospitals/{names[1]}/suspend", token=admin)
    check("admin suspends a hospital", s == 200 and r["status"] == "suspended", s)
    s, _, _ = call(base, "GET", "/fl/status", token=toks[1])
    check("suspension takes effect immediately", s == 403, s)
    s, _, _ = call(base, "POST", f"/admin/hospitals/{names[1]}/restore", token=admin)
    check("admin restores it", s == 200, s)

    if a.mutating:
        need = st["min_clients"]
        if need > 2:
            print(f"SKIP fake round: server needs {need} hospitals, smoke created 2")
        else:
            ups = [[x + d_ for x in arrays] for d_ in (0.01, 0.03)]
            counts = [100, 300]
            outs = [call(base, "POST", "/fl/update-json", token=t, body={"base_version": version, "num_examples": n, "weights": flat(u),
                         "privacy": {"dp_enabled": i == 0, "sigma": 1.0, "sample_rate": 0.05, "steps": 100} if i == 0 else None})[:2]
                    for i, (t, u, n) in enumerate(zip(toks, ups, counts))]
            check("both updates accepted", all(s_ == 200 for s_, _ in outs), [o[0] for o in outs])
            check("round aggregated", outs[-1][1].get("aggregated") and outs[-1][1].get("new_version") == version + 1)
            s, data, _ = call(base, "GET", "/fl/model", token=toks[0], raw=True)
            new, nv = npz_to_arrays(data)
            expected = (ups[0][0] * 100 + ups[1][0] * 300) / 400
            check("new model == weighted average", nv == version + 1 and np.allclose(new[0], expected, atol=1e-5))
            s, hist, _ = call(base, "GET", "/fl/history", token=toks[0])
            sec = hist["history"][-1]["security"]
            check("history records the round with its security block", s == 200 and sec["accepted"] == 2 and sec["dp_enabled"] == 1 and sec["egress"]["raw_rows_transmitted"] == 0)

    # ---- cleanup: remove the throw-away hospitals
    for n in names:
        s, _, _ = call(base, "DELETE", f"/admin/hospitals/{n}", token=admin)
        check(f"cleanup: removed {n}", s == 200, s)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
