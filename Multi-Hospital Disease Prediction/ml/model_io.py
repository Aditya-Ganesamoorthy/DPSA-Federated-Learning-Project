"""Numpy-only model I/O, forward pass and metrics.

The live federation server and the hospital clients exchange the model as a small `.npz` file.
Array order is the torch `state_dict` order of `ml.model.MLP`:
    net.0.weight (32,15), net.0.bias (32), net.2.weight (16,32), net.2.bias (16), net.4.weight (1,16), net.4.bias (1)
Keeping this module free of torch/sklearn lets the server run on a 512 MB free instance.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from ml import config, schema

N_ARRAYS = 6


def expected_shapes() -> list[tuple[int, ...]]:
    h1, h2 = config.HIDDEN
    return [(h1, schema.N_FEATURES), (h1,), (h2, h1), (h2,), (1, h2), (1,)]


def validate_arrays(arrays: list[np.ndarray]) -> None:
    """Raise ValueError unless arrays match the architecture, are float, and finite."""
    shapes = expected_shapes()
    if len(arrays) != len(shapes):
        raise ValueError(f"expected {len(shapes)} arrays, got {len(arrays)}")
    for i, (a, s) in enumerate(zip(arrays, shapes)):
        if tuple(a.shape) != s:
            raise ValueError(f"array {i}: expected shape {s}, got {tuple(a.shape)}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"array {i} contains NaN or infinity")


def to_bytes(arrays: list[np.ndarray], version: int = -1) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, version=np.array(version, dtype=np.int64),
             **{f"p{i}": np.asarray(a, dtype=np.float32) for i, a in enumerate(arrays)})
    return buf.getvalue()


def from_bytes(data: bytes) -> tuple[list[np.ndarray], int]:
    """Parse an npz payload. allow_pickle=False: never unpickle data from the network."""
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        names = [f"p{i}" for i in range(N_ARRAYS)]
        missing = [n for n in names if n not in z.files]
        if missing:
            raise ValueError(f"missing arrays: {missing}")
        arrays = [z[n].astype(np.float32) for n in names]
        version = int(z["version"]) if "version" in z.files else -1
    validate_arrays(arrays)
    return arrays, version


def save(path: Path, arrays: list[np.ndarray], version: int = -1) -> None:
    Path(path).write_bytes(to_bytes(arrays, version))


def load(path: Path) -> tuple[list[np.ndarray], int]:
    return from_bytes(Path(path).read_bytes())


def predict_proba(arrays: list[np.ndarray], X: np.ndarray) -> np.ndarray:
    x = np.asarray(X, dtype=np.float32)
    h = np.maximum(x @ arrays[0].T + arrays[1], 0)
    h = np.maximum(h @ arrays[2].T + arrays[3], 0)
    z = (h @ arrays[4].T + arrays[5])[:, 0]
    return (1.0 / (1.0 + np.exp(-z.astype(np.float64)))).astype(np.float64)


def weighted_average(updates: list[list[np.ndarray]], counts: list[int]) -> list[np.ndarray]:
    """FedAvg: w = sum_k (n_k / N) * w_k (verified equal to Flower's FedAvg in the test-suite)."""
    total = float(sum(counts))
    return [sum(u[i].astype(np.float64) * (n / total) for u, n in zip(updates, counts)).astype(np.float32)
            for i in range(len(updates[0]))]


def flat(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based ROC-AUC with tie handling; NaN when only one class is present."""
    y = np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ps = np.asarray(p)[order]
    ranks = np.empty(len(p), dtype=np.float64)
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def metrics(y: np.ndarray, p: np.ndarray, threshold: float = config.THRESHOLD) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: float(a / b) if b else 0.0
    eps = 1e-7
    pc = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return {"accuracy": div(tp + tn, len(y)), "recall": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "precision": div(tp, tp + fp), "auc": auc(y, p),
            "loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))}
