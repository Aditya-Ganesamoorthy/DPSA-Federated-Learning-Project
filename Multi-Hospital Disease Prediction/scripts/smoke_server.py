"""Black-box smoke test of a running server (local or Render). Needs only numpy + the standard library.

  python scripts/smoke_server.py --url http://localhost:8000 --admin-password <pw>
  python scripts/smoke_server.py --url https://<app>.onrender.com --admin-password <pw>            (read-only checks)
  python scripts/smoke_server.py --url http://localhost:8000 --admin-password <pw> --mutating      (also runs a fake round;
        NEVER use --mutating on a server whose global model matters: it adds a synthetic version to the history)
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


def arrays_to_b64(arrays, version):
    buf = io.BytesIO()
    np.savez(buf, version=np.array(version, dtype=np.int64), **{f"p{i}": a.astype(np.float32) for i, a in enumerate(arrays)})
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--admin-password", required=True)
    ap.add_argument("--admin-username", default="admin")
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
    print(f"     (responded in {time.time() - t0:.1f}s; global model v{h.get('global_model_version')}, models {h.get('models_loaded')})")
    s, page, _ = call(base, "GET", "/", raw=True, api=False)
    check("dashboard served at /", s == 200 and b"<div id=\"root\">" in page, s)
    s, b, _ = call(base, "POST", "/auth/login", body={"username": a.admin_username, "password": "wrong-password"})
    check("wrong admin password rejected", s == 401, s)
    s, b, _ = call(base, "POST", "/auth/login", body={"username": a.admin_username, "password": a.admin_password})
    check("admin login", s == 200 and b.get("role") == "admin", s)
    admin = b.get("access_token")
    s, _, _ = call(base, "GET", "/fl/status")
    check("status requires login", s == 401, s)
    s, _, _ = call(base, "POST", "/predict", body={})
    check("predict requires login", s == 401, s)

    names = [f"smoke-{uuid.uuid4().hex[:6]}" for _ in range(2)]
    creds = []
    for n in names:
        s, c, _ = call(base, "POST", "/admin/hospitals", token=admin, body={"name": n})
        check(f"admin creates hospital {n}", s == 200 and c.get("password"), s)
        creds.append(c)
    toks = []
    for c in creds:
        s, b, _ = call(base, "POST", "/auth/login", body={"username": c["username"], "password": c["password"]})
        check(f"hospital {c['username']} login", s == 200 and b.get("role") == "hospital", s)
        toks.append(b.get("access_token"))
    s, st, _ = call(base, "GET", "/fl/status", token=toks[0])
    check("hospital sees status", s == 200 and "version" in st, s)
    version = st["version"]
    s, data, hdr = call(base, "GET", "/fl/model", token=toks[0], raw=True)
    arrays, v = npz_to_arrays(data) if s == 200 else (None, None)
    check("hospital downloads global model", s == 200 and v == version and len(arrays) == 6, s)
    s, b, _ = call(base, "POST", "/admin/aggregate", token=toks[0])
    check("hospital cannot use admin routes", s == 403, s)
    patient = {"gender": "M", "age": 60, **{k: True for k in
               "smoking yellow_fingers anxiety peer_pressure chronic_disease fatigue allergy wheezing alcohol_consuming "
               "coughing shortness_of_breath swallowing_difficulty chest_pain".split()}}
    s, p, _ = call(base, "POST", "/predict", token=toks[0], body={"model": "global", "patient": patient})
    check("global-model prediction", s == 200 and 0 <= p.get("probability", -1) <= 1, s)
    s, _, _ = call(base, "POST", "/fl/update", token=toks[0], body={"base_version": version + 99, "num_examples": 5,
                   "npz_b64": arrays_to_b64(arrays, version)})
    check("stale update rejected (409)", s == 409, s)
    s, _, _ = call(base, "POST", "/fl/update", token=toks[0], body={"base_version": version, "num_examples": 5,
                   "npz_b64": arrays_to_b64([x + 1000 for x in arrays], version)})
    check("oversized update rejected (422)", s == 422, s)

    if a.mutating:
        need = st["min_clients"]
        if need > len(toks):
            print(f"SKIP fake round: server needs {need} hospitals, smoke created {len(toks)}")
        else:
            ups = [[x + d for x in arrays] for d in (0.01, 0.03)]
            counts = [100, 300]
            outs = [call(base, "POST", "/fl/update", token=t, body={"base_version": version, "num_examples": n,
                         "npz_b64": arrays_to_b64(u, version)})[:2] for t, u, n in zip(toks, ups, counts)]
            check("both updates accepted", all(s == 200 for s, _ in outs), [o[0] for o in outs])
            check("round aggregated", outs[-1][1].get("aggregated") and outs[-1][1].get("new_version") == version + 1)
            s, data, _ = call(base, "GET", "/fl/model", token=toks[0], raw=True)
            new, nv = npz_to_arrays(data)
            expected = (ups[0][0] * 100 + ups[1][0] * 300) / 400
            check("new model == weighted average", nv == version + 1 and np.allclose(new[0], expected, atol=1e-5))
            s, hist, _ = call(base, "GET", "/fl/history", token=toks[0])
            check("history records the round with test metrics",
                  s == 200 and hist["history"][-1]["version"] == version + 1 and hist["history"][-1]["test_metrics"])
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
