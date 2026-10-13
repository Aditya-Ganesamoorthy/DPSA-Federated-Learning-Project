"""Evaluation metrics. Recall and AUC lead the discussion (a missed cancer is the costly error)."""
from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from ml import config

log = logging.getLogger(__name__)


def _div(a, b):
    return float(a / b) if b else 0.0


def compute(y_true: np.ndarray, proba: np.ndarray, threshold: float = config.THRESHOLD) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    if len(np.unique(y)) < 2:
        log.warning("single-class evaluation set: AUC undefined")
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y, p))
    eps = 1e-7
    loss = float(-np.mean(y * np.log(np.clip(p, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - p, eps, 1 - eps))))
    prec = _div(tp, tp + fp)
    rec = _div(tp, tp + fn)
    return {
        "accuracy": _div(tp + tn, len(y)), "precision": prec, "recall": rec,
        "specificity": _div(tn, tn + fp), "f1": _div(2 * prec * rec, prec + rec), "auc": auc,
        "loss": loss, "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def roc_points(y_true, proba, max_points: int = 60) -> dict:
    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), proba)
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).astype(int))
    return {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}


SCALAR_KEYS = ["accuracy", "precision", "recall", "specificity", "f1", "auc", "loss"]


def mean_std(runs: list[dict]) -> dict:
    """Aggregate a list of metric dicts over seeds. NaN-safe (NaN AUC -> skipped, null if all NaN)."""
    out = {}
    for k in SCALAR_KEYS:
        v = np.array([r[k] for r in runs], dtype=float)
        v = v[~np.isnan(v)]
        out[k] = {"mean": float(v.mean()) if len(v) else None,
                  "std": float(v.std(ddof=0)) if len(v) else None}
    out["n_seeds"] = len(runs)
    return out


from ml.jsonutil import clean_json  # noqa: E402,F401  (re-exported for backwards compatibility)
