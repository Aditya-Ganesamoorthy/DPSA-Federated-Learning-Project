"""Hand-written FedAvg (about 40 lines). Used to cross-check Flower, to explain aggregation in the
viva, and as a fallback if Flower cannot be installed.

    w_global = sum_k (n_k / N) * w_k
"""
from __future__ import annotations

import numpy as np

from ml import config, model as M


def weighted_average(updates: list[list[np.ndarray]], counts: list[int]) -> list[np.ndarray]:
    total = float(sum(counts))
    return [sum(u[layer].astype(np.float64) * (n / total) for u, n in zip(updates, counts)).astype(np.float32)
            for layer in range(len(updates[0]))]


def client_update(global_params, scenario: str, idx: int, epochs: int, seed: int):
    """One hospital's local training. Reads ONLY that hospital's CSV."""
    X, y = M.load_xy(M.client_path(scenario, idx))
    net = M.make_model("mlp", 0)
    M.set_parameters(net, global_params)
    M.train_local(net, X, y, epochs, seed)
    return M.get_parameters(net), len(X)


def run(scenario: str, seed: int, rounds: int = config.FL_ROUNDS, epochs: int = config.LOCAL_EPOCHS):
    params = M.get_parameters(M.make_model("mlp", seed))
    for r in range(1, rounds + 1):
        results = [client_update(params, scenario, k, epochs, seed * 10_000 + r * 100 + k)
                   for k in range(1, config.N_CLIENTS + 1)]
        params = weighted_average([p for p, _ in results], [n for _, n in results])
    return params
