"""Dependency-free JSON helpers (kept separate so the lightweight server never imports sklearn/torch)."""
from __future__ import annotations

import math

import numpy as np


def clean_json(o):
    """Recursively convert NaN/inf to None so output is valid JSON; numpy scalars to Python."""
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o
