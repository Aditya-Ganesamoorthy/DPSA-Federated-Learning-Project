"""Pipeline sanity check on SYNTHETIC labels (NOT the real dataset).

The real Kaggle labels are nearly unpredictable from the features, so real-data results cannot show
whether the pipeline learns. Here we keep the real feature rows and hospital partitions but replace
the label with a planted-signal rule. If centralized, FL, FL+DP and FL+SecAgg all learn here, a
near-chance result on the real data reflects the data, not a pipeline bug.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ml import centralized, config, federated, metrics, schema


def plant_labels(df: pd.DataFrame, seed: int = 123) -> pd.Series:
    rng = np.random.default_rng(seed)
    z = (1.6 * (df["smoking"] + df["wheezing"] + df["coughing"] + df["chest_pain"] - 2)
         + 4.0 * (df["age"] - 0.55))
    return pd.Series((rng.random(len(df)) < 1 / (1 + np.exp(-z))).astype(int), index=df.index)


def build_synthetic_dir() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="fl_sanity_"))
    for f in config.CLIENT_DIR.glob("*.csv"):
        d = pd.read_csv(f)
        d[schema.TARGET] = plant_labels(d)
        d.to_csv(tmp / f.name, index=False)
    return tmp


def run(seeds=(0, 1, 2), rounds: int = config.FL_ROUNDS) -> dict:
    real_dir = config.CLIENT_DIR
    tmp = build_synthetic_dir()
    out = {"note": "SYNTHETIC planted-signal labels on the real feature rows; pipeline validation only.",
           "seeds": list(seeds)}
    try:
        config.CLIENT_DIR = tmp
        rows = {"centralized": [], "fl_iid": [], "fl_noniid": [], "fl_noniid_dp_sigma1.0": [],
                "fl_noniid_secagg": []}
        for s in seeds:
            rows["centralized"].append(centralized.run_centralized(s)[1])
            rows["fl_iid"].append(federated.run_fl("iid", s, rounds)["rounds"][-1]["test"])
            rows["fl_noniid"].append(federated.run_fl("noniid", s, rounds)["rounds"][-1]["test"])
            r = federated.run_fl("noniid", s, rounds, sigma=1.0)
            rows["fl_noniid_dp_sigma1.0"].append(r["rounds"][-1]["test"])
            out["dp_epsilon_max"] = r["epsilon_max"]
            rows["fl_noniid_secagg"].append(federated.run_fl("noniid", s, rounds, secure=True)["rounds"][-1]["test"])
        out["methods"] = {k: metrics.mean_std(v) for k, v in rows.items()}
    finally:
        config.CLIENT_DIR = real_dir
        shutil.rmtree(tmp, ignore_errors=True)
    return out
