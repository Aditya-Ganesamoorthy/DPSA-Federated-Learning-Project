"""E1 centralized baseline (pooled train data) and E8 local-only baseline (each hospital alone).

Matched budget: epochs = FL rounds x local epochs, so every sample is visited the same number of
times as in federated training.
"""
from __future__ import annotations

from ml import config, metrics, model as M

MATCHED_EPOCHS = config.FL_ROUNDS * config.LOCAL_EPOCHS


def eval_split(net, name: str = "test") -> dict:
    X, y = M.load_xy(config.CLIENT_DIR / f"{name}.csv")
    return metrics.compute(y, M.predict_proba(net, X))


def run_centralized(seed: int, kind: str = "mlp", epochs: int = MATCHED_EPOCHS):
    X, y = M.load_xy(config.CLIENT_DIR / "train_pool.csv")
    net = M.make_model(kind, seed)
    M.train_local(net, X, y, epochs, seed)
    return net, eval_split(net)


def run_local_only(scenario: str, client_idx: int, seed: int, epochs: int = MATCHED_EPOCHS):
    X, y = M.load_xy(M.client_path(scenario, client_idx))
    net = M.make_model("mlp", seed)
    M.train_local(net, X, y, epochs, seed)
    return net, eval_split(net)
