"""The in-browser trainer (frontend/src/lib/trainer.js) must agree with the Python/torch reference."""
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
import torch

from backend.routers.api import features
from ml import config, model as M, model_io, preprocess, sanity, schema

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


@pytest.fixture(scope="module")
def js(tmp_path_factory):
    d = tmp_path_factory.mktemp("parity")
    weights, _ = model_io.load(config.MODELS_DIR / "global_v0.npz")
    test = pd.read_csv(config.CLIENT_DIR / "test.csv")
    X = test[schema.FEATURES].to_numpy(np.float32)[:200]
    y = test[schema.TARGET].to_numpy(np.float32)[:200]

    raw = preprocess.load_raw()
    raw_head = pd.concat([raw.head(40), raw.head(3)])                       # 3 deliberate duplicates
    raw_head.columns = [c.upper() for c in raw_head.columns]                # as in the Kaggle file
    enc_head = pd.read_csv(config.PROCESSED_CSV).head(40)
    bad_rows = raw.head(6).copy()
    bad_rows.columns = [c.upper() for c in bad_rows.columns]
    bad_rows.loc[bad_rows.index[1], "SMOKING"] = 7
    bad_rows.loc[bad_rows.index[3], "GENDER"] = "Q"

    planted = pd.read_csv(config.CLIENT_DIR / "train_pool.csv")
    labels = sanity.plant_labels(planted)
    Xp, yp = planted[schema.FEATURES].to_numpy(np.float32), labels.to_numpy(np.float32)
    payload = {
        "weights": [w.ravel().tolist() for w in weights], "nIn": schema.N_FEATURES, "X": X.tolist(), "y": y.tolist(),
        "gradRows": [0, 3, 17, 60, 150], "spec": features(),
        "rawCsv": raw_head.to_csv(index=False), "encodedCsv": enc_head.to_csv(index=False),
        "badCsv": "a,b\n1,2\n", "badRowsCsv": bad_rows.to_csv(index=False),
        "learn": {"trainX": Xp[:600].tolist(), "trainY": yp[:600].tolist(), "holdX": Xp[600:1000].tolist(), "holdY": yp[600:1000].tolist()},
    }
    f = d / "in.json"
    f.write_text(json.dumps(payload))
    r = subprocess.run([NODE, str(config.ROOT / "frontend" / "tests" / "parity.mjs"), str(f)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout), payload, weights, raw, X, y


def test_forward_pass_matches_numpy(js):
    out, _, weights, _, X, _ = js
    ref = model_io.predict_proba(weights, X)
    np.testing.assert_allclose(out["proba"], ref, atol=1e-6)


def test_auc_matches_sklearn(js):
    from sklearn.metrics import roc_auc_score
    out, _, _, _, _, y = js
    assert out["auc"] == pytest.approx(roc_auc_score(y, out["proba"]), abs=1e-12)


def test_per_sample_gradients_match_torch_autograd(js):
    out, payload, weights, _, X, y = js
    net = M.make_model("mlp", 0)
    M.set_parameters(net, weights)
    for row, got in zip(payload["gradRows"], out["grads"]):
        net.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(torch.from_numpy(X[row:row + 1])), torch.from_numpy(y[row:row + 1]))
        loss.backward()
        assert got["loss"] == pytest.approx(loss.item(), abs=1e-6)
        for p, g in zip(net.parameters(), got["grad"]):
            np.testing.assert_allclose(np.array(g), p.grad.numpy().ravel(), atol=1e-6)


def test_raw_csv_encoding_matches_python_schema(js):
    out, _, _, raw, _, _ = js
    p = out["prep_raw"]
    expected = schema.encode_frame(raw.head(40).drop_duplicates())
    assert p["layout"] == "raw" and p["dups"] == 3 and p["n"] == len(expected) and not p["issues"]
    np.testing.assert_allclose(np.array(p["X"]).reshape(p["n"], -1), expected[schema.FEATURES].to_numpy(), atol=1e-9)
    np.testing.assert_array_equal(p["y"], expected[schema.TARGET].to_numpy())


def test_encoded_csv_is_accepted_as_is(js):
    out, *_ = js
    enc = pd.read_csv(config.PROCESSED_CSV).head(40)
    p = out["prep_encoded"]
    assert p["layout"] == "encoded" and p["n"] == 40 and not p["issues"]
    np.testing.assert_allclose(np.array(p["X"]).reshape(40, -1), enc[schema.FEATURES].to_numpy(), atol=1e-9)


def test_bad_files_are_reported_not_trained(js):
    out, *_ = js
    assert "Missing columns" in out["prep_bad"]["error"]
    rows = out["prep_bad_rows"]
    assert rows["n"] == 4 and len(rows["issues"]) == 2 and any("smoking" in i for i in rows["issues"]) and any("gender" in i for i in rows["issues"])


def test_training_learns_planted_signal_and_dp_training_runs(js):
    out, *_ = js
    L = out["learn"]
    assert L["all_finite"]
    assert L["after"]["auc"] > L["before"]["auc"] + 0.1 and L["after"]["auc"] > 0.7        # plain training learns
    assert L["dp_after"]["auc"] > 0.6                                                       # DP training still learns something
    assert L["dp_steps"] == 6 * int(np.ceil(600 / 32)) and L["dp_q"] == pytest.approx(32 / 600)
    assert 0 < L["update_norm"] < 20 and 0 < L["dp_update_norm"] < 20                       # passes the server's size guard
