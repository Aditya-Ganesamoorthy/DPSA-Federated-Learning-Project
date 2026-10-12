"""Secure aggregation by pairwise masking (SIMULATION, not a cryptographic protocol).

Each pair (i, j) shares a seed s_ij. Client i adds +PRG(s_ij) if i < j and -PRG(s_ij) if i > j to
its pre-weighted, fixed-point-quantised update. All arithmetic is integer modulo 2^32, so the
masks cancel EXACTLY in the server's sum.
"""
from __future__ import annotations

import numpy as np

from ml import config

MOD = 1 << 32


def quantize(vec: np.ndarray, bits: int = config.SECAGG_SCALE_BITS) -> np.ndarray:
    """float vector -> unsigned fixed-point residues mod 2^32 (two's complement)."""
    q = np.round(np.asarray(vec, dtype=np.float64) * (1 << bits)).astype(np.int64)
    return (q % MOD).astype(np.uint64)


def dequantize(res: np.ndarray, bits: int = config.SECAGG_SCALE_BITS) -> np.ndarray:
    r = res.astype(np.int64) % MOD
    signed = np.where(r >= MOD // 2, r - MOD, r)
    return signed.astype(np.float64) / (1 << bits)


def pair_seed(base_seed: int, i: int, j: int) -> int:
    a, b = sorted((i, j))
    return base_seed * 1_000_003 + a * 1009 + b


def _mask(seed: int, size: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, MOD, size=size, dtype=np.uint64)


def mask_update(i: int, n_clients: int, update_int: np.ndarray, base_seed: int) -> np.ndarray:
    """Client i's masked (uint32-range) update."""
    out = update_int.astype(np.uint64) % MOD
    for j in range(n_clients):
        if j == i:
            continue
        m = _mask(pair_seed(base_seed, i, j), len(update_int))
        out = (out + m) % MOD if i < j else (out + MOD - m) % MOD
    return out


def aggregate_masked(masked: list[np.ndarray]) -> np.ndarray:
    total = np.zeros_like(masked[0], dtype=np.uint64)
    for m in masked:
        total = (total + m.astype(np.uint64)) % MOD
    return total


def secure_weighted_average(flat_updates: list[np.ndarray], counts: list[int], base_seed: int = 0,
                            drop: int | None = None) -> dict:
    """Full round: clients pre-weight by n_k/N, quantise, mask; server sums.

    `drop` = index of a client whose masked update never arrives (dropout failure demo).
    Returns the decoded aggregate plus diagnostics.
    """
    k = len(flat_updates)
    total = float(sum(counts))
    weighted = [u.astype(np.float64) * (n / total) for u, n in zip(flat_updates, counts)]
    q = [quantize(w) for w in weighted]
    masked = [mask_update(i, k, q[i], base_seed) for i in range(k)]
    plain_int_sum = aggregate_masked(q)
    received = [m for i, m in enumerate(masked) if i != drop]
    masked_sum = aggregate_masked(received)
    decoded = dequantize(masked_sum)
    plain_float = sum(weighted)
    return {
        "aggregate": decoded, "masked": masked, "plain_float": plain_float,
        "int_max_abs_diff": int(np.max(np.abs(masked_sum.astype(np.int64) - plain_int_sum.astype(np.int64)))),
        "float_max_abs_error": float(np.max(np.abs(decoded - plain_float))),
    }


def correlation_with_true(masked_one: np.ndarray, true_update: np.ndarray) -> float:
    """Pearson correlation between a single masked update (decoded) and the true update."""
    d = dequantize(masked_one)
    if np.std(d) == 0 or np.std(true_update) == 0:
        return 0.0
    return float(np.corrcoef(d, true_update)[0, 1])
