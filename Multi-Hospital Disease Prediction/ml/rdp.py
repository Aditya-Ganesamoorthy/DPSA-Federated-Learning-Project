"""Numpy-only Renyi-DP accountant for the (Poisson-subsampled) Gaussian mechanism.

The live server cannot import torch/Opacus (512 MB free instance), yet it must turn a hospital's declared
(noise multiplier, sample rate, steps) into epsilon and compose it across rounds. This implements the exact
RDP of the sampled Gaussian mechanism for INTEGER orders (Mironov, Talwar, Zhang 2019) and the same RDP->(eps, delta)
conversion that Opacus uses. Opacus also searches fractional orders, so its epsilon can be slightly smaller; ours is a
valid, marginally conservative upper bound (verified against Opacus in the test-suite).
"""
from __future__ import annotations

import math

import numpy as np

ORDERS = list(range(2, 257))


def _log_add(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = max(a, b), min(a, b)
    return hi + math.log1p(math.exp(lo - hi))


def _rdp_one_step(q: float, sigma: float, alpha: int) -> float:
    """RDP of order alpha for ONE step of the Poisson-subsampled Gaussian (noise multiplier sigma, rate q)."""
    if q == 0:
        return 0.0
    if sigma == 0:
        return math.inf
    if q >= 1.0:
        return alpha / (2.0 * sigma ** 2)
    log_a = -math.inf
    for k in range(alpha + 1):
        log_coef = math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
        term = log_coef + (k * math.log(q) if k else 0.0) + ((alpha - k) * math.log1p(-q) if alpha - k else 0.0)
        term += (k * k - k) / (2.0 * sigma ** 2)
        log_a = _log_add(log_a, term)
    return log_a / (alpha - 1)


def rdp_curve(history: list[tuple[float, float, int]]) -> np.ndarray:
    """Total RDP over `ORDERS` for a composition of (noise_multiplier, sample_rate, steps) segments."""
    total = np.zeros(len(ORDERS))
    for sigma, q, steps in history:
        total += np.array([_rdp_one_step(q, sigma, a) for a in ORDERS]) * steps
    return total


def epsilon(history: list[tuple[float, float, int]], delta: float = 1e-5) -> float:
    """Smallest epsilon over orders, with the conversion used by Opacus (Balle et al. 2020 style)."""
    if not history:
        return 0.0
    rdp = rdp_curve(history)
    orders = np.array(ORDERS, dtype=float)
    eps = rdp - (math.log(delta) + np.log(orders)) / (orders - 1) + np.log((orders - 1) / orders)
    best = float(np.nanmin(eps))
    return best if math.isfinite(best) else math.inf
