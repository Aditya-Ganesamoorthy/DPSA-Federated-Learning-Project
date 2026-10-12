"""Communication efficiency analysis (E6). Counts BOTH directions, per client and in total."""
from __future__ import annotations

import numpy as np

from ml import config

FLOAT32 = 4


def analytic(n_params: int, k: int, rounds: int) -> dict:
    per_client_round = n_params * FLOAT32 * 2          # upload + download
    per_round = per_client_round * k
    return {"n_params": n_params, "bytes_per_client_per_round": per_client_round,
            "bytes_per_round_total": per_round, "rounds": rounds, "total_bytes": per_round * rounds,
            "total_mb": per_round * rounds / 1e6}


def raw_data_bytes(scenario: str) -> dict:
    """Cost of the centralized alternative: every hospital uploads its data once."""
    csv_bytes = sum((config.CLIENT_DIR / f"{scenario}_hospital_{k}.csv").stat().st_size
                    for k in range(1, config.N_CLIENTS + 1))
    import pandas as pd
    rows = sum(len(pd.read_csv(config.CLIENT_DIR / f"{scenario}_hospital_{k}.csv"))
               for k in range(1, config.N_CLIENTS + 1))
    from ml import schema
    matrix = rows * (schema.N_FEATURES + 1) * FLOAT32
    return {"rows": rows, "csv_bytes": csv_bytes, "float32_matrix_bytes": matrix,
            "csv_mb": csv_bytes / 1e6, "float32_matrix_mb": matrix / 1e6}


def rounds_to_fraction_of_final_loss(train_loss: list[float], tol: float = 0.10) -> int | None:
    """Optimisation-convergence proxy: first round whose weighted client training loss is within
    `tol` of its value at the last round. (AUC-based targets are meaningless when AUC ~ 0.5.)"""
    final = train_loss[-1]
    for r, v in enumerate(train_loss, 1):
        if abs(v - final) <= tol * abs(final):
            return r
    return None


def rounds_to_target_auc(test_auc: list[float], target: float) -> int | None:
    for r, v in enumerate(test_auc):
        if v is not None and v >= target:
            return r
    return None
