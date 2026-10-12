"""Populate results/runs/ cache entries in parallel worker processes (same code path as run_all).

    python -m ml.prefill dp 3.0      # DP sweep for one sigma, all seeds
    python -m ml.prefill sec         # secure-aggregation runs
    python -m ml.prefill comm        # communication runs (E in 1, 2, 5)
run_all then finds the cache and skips the work; results are identical.
"""
from __future__ import annotations

import sys

import torch

from ml import config
from ml.run_all import _fl, cached

torch.set_num_threads(1)


def main(kind: str, arg: str | None):
    seeds, rounds = config.SEEDS, config.FL_ROUNDS
    if kind == "dp":
        sg = float(arg)
        for s in seeds:
            cached(f"dp_{sg}_s{s}", lambda s=s: _fl("noniid", s, rounds, sigma=sg), False)
    elif kind == "sec":
        for s in seeds:
            cached(f"sec_noniid_s{s}", lambda s=s: _fl("noniid", s, rounds, secure=True), False)
    elif kind == "comm":
        for e in (1, 2, 5):
            for s in seeds[:3]:
                cached(f"comm_E{e}_s{s}", lambda s=s, e=e: _fl("noniid", s, rounds, epochs=e), False)
    print("prefill done", kind, arg)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
