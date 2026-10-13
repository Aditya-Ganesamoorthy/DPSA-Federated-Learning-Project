"""Hospital-side logic: load local data, train locally from the downloaded global model, return the update."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml import config, dp, metrics as M_metrics, model as M, schema


def load_hospital_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Accepts the raw Kaggle layout (GENDER, AGE, 1/2 survey codes, LUNG_CANCER YES/NO) or the already-encoded
    layout produced by this project. Returns float32 X (schema.FEATURES order) and y in {0,1}."""
    raw = schema.normalize_columns(pd.read_csv(path, encoding="utf-8-sig"))
    if "gender" in raw and pd.api.types.is_numeric_dtype(raw["gender"]):
        df = raw
    else:
        df = schema.encode_frame(raw.drop_duplicates())
    missing = [c for c in schema.FEATURES + [schema.TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    if df[schema.FEATURES + [schema.TARGET]].isna().any().any():
        raise ValueError("CSV contains missing values; clean them locally first")
    y = df[schema.TARGET].to_numpy(np.float32)
    if set(np.unique(y)) - {0.0, 1.0}:
        raise ValueError("target must be 0/1 (or NO/YES in the raw layout)")
    return df[schema.FEATURES].to_numpy(np.float32), y


def local_update(global_arrays: list[np.ndarray], X: np.ndarray, y: np.ndarray, *, epochs: int = config.LOCAL_EPOCHS,
                 seed: int = 0, sigma: float = 0.0, holdout: float = 0.0, hospital_id: str = "hospital",
                 accountant_path: str | Path | None = None) -> dict:
    """Train locally starting from the global model. Optionally record-level DP-SGD (sigma > 0) with an accountant
    persisted on disk, so epsilon composes across rounds. Returns the new weights, example count and local metrics."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_hold = int(len(X) * holdout)
    hold, train = idx[:n_hold], idx[n_hold:]
    Xtr, ytr = X[train], y[train]

    def eval_hold(net):
        if n_hold < 20 or len(np.unique(y[hold])) < 2:
            return None
        m = M_metrics.compute(y[hold], M.predict_proba(net, X[hold]))
        return {k: m[k] for k in ("accuracy", "recall", "auc")}

    net = M.make_model("mlp", 0)
    M.set_parameters(net, global_arrays)
    before = eval_hold(net)
    out = {"epsilon": None}
    if sigma > 0:
        store = dp.AccountantStore.load(accountant_path) if accountant_path else dp.AccountantStore()
        dp.train_local_dp(net, Xtr, ytr, epochs, seed, sigma, store, hospital_id)
        out["epsilon"] = store.epsilon(hospital_id)
        if accountant_path:
            store.save(accountant_path)
    else:
        M.train_local(net, Xtr, ytr, epochs, seed)
    out.update(arrays=M.get_parameters(net), num_examples=int(len(Xtr)), holdout_before=before,
               holdout_after=eval_hold(net), n_holdout=int(n_hold))
    return out
