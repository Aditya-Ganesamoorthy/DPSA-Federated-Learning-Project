"""Optional extras (plan section 2): FedProx, 8/4-bit update quantisation, label-flip poisoning + median defence.

Each experiment runs on the REAL data (where every method is at chance, so only the byte/cost effects are
informative) and on the planted-signal data from ml/sanity.py (where accuracy effects are visible).
Writes results/extras.json.   python -m ml.extras
"""
from __future__ import annotations

import json
import shutil

import numpy as np

from ml import config, federated, metrics, sanity

SEEDS = [0, 1, 2]
ROUNDS = config.FL_ROUNDS


def _summ(runs: list[dict]) -> dict:
    tests = [r["rounds"][-1]["test"] for r in runs]
    out = metrics.mean_std(tests)
    out["bytes_up_total_mb"] = float(np.mean([sum(x.get("bytes_up", 0) for x in r["rounds"]) for r in runs]) / 1e6)
    return out


def _run(label: str, **kw) -> dict:
    runs = [federated.run_fl("noniid", s, ROUNDS, **kw) for s in SEEDS]
    print(label, round(metrics.mean_std([r["rounds"][-1]["test"] for r in runs])["auc"]["mean"], 3), flush=True)
    return _summ(runs)


def suite(include_poison: bool) -> dict:
    out = {"fedavg": _run("fedavg")}
    out["quantization"] = {f"{b}bit": _run(f"quant{b}", quant_bits=b) for b in (8, 4)}
    out["fedprox"] = {f"mu={m}": _run(f"prox{m}", prox_mu=m) for m in (0.01, 0.1, 1.0)}
    if include_poison:
        out["poisoning"] = {
            "clean_fedavg": out["fedavg"],
            "labelflip_h1_fedavg": _run("flip1-fedavg", poisoned_client=1),
            "labelflip_h1_median": _run("flip1-median", poisoned_client=1, aggregator="median"),
            "labelflip_h3_fedavg": _run("flip3-fedavg", poisoned_client=3),
            "labelflip_h3_median": _run("flip3-median", poisoned_client=3, aggregator="median"),
            "clean_median": _run("clean-median", aggregator="median"),
        }
    return out


def main() -> None:
    real = suite(include_poison=False)
    real_dir = config.CLIENT_DIR
    tmp = sanity.build_synthetic_dir()
    try:
        config.CLIENT_DIR = tmp
        planted = suite(include_poison=True)
    finally:
        config.CLIENT_DIR = real_dir
        shutil.rmtree(tmp, ignore_errors=True)
    res = {"seeds": SEEDS, "rounds": ROUNDS, "real_data": real, "planted_signal_data": planted,
           "notes": {
               "quantization": "update delta quantised per tensor to 8/4 bits; upload bytes = bits/8 per parameter + 4 B scale per tensor; download stays float32",
               "fedprox": "proximal term mu/2*||w - w_global||^2 in local training",
               "poisoning": "malicious hospital flips ALL its training labels; planted-signal labels only (real labels have no signal to poison); median = coordinate-wise median of the 3 client models",
               "planted_signal_data": "synthetic labels on real feature rows; see ml/sanity.py"}}
    (config.RESULTS_DIR / "extras.json").write_text(json.dumps(metrics.clean_json(res), indent=1))
    print("extras done")


if __name__ == "__main__":
    main()
