"""Exploratory data analysis of the cleaned dataset -> results/eda.json (served to the Overview page).

    python -m ml.eda
Counts only: no row-level data is written.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ml import config, schema
from ml.jsonutil import clean_json


def run() -> dict:
    df = pd.read_csv(config.PROCESSED_CSV)
    y = df[schema.TARGET].astype(int)
    pos, neg = df[y == 1], df[y == 0]

    ages = df["age"] * schema.AGE_SCALE
    edges = list(range(30, 90, 10))
    labels = [f"{a}-{a + 9}" for a in edges[:-1]] + ["80+"]
    bins = np.digitize(ages, edges[1:])          # 0..5
    age_hist = {"bins": labels,
                "positive": [int(((bins == i) & (y == 1)).sum()) for i in range(len(labels))],
                "negative": [int(((bins == i) & (y == 0)).sum()) for i in range(len(labels))]}

    feats = []
    for c in schema.FEATURES:
        if c == "age":
            continue
        corr = float(np.corrcoef(df[c], y)[0, 1])
        feats.append({"name": c, "prevalence_positive": float(pos[c].mean()), "prevalence_negative": float(neg[c].mean()),
                      "correlation_with_cancer": corr})
    cols = [c for c in schema.FEATURES]
    cm = np.corrcoef(df[cols + [schema.TARGET]].to_numpy().T)
    out = {
        "rows": int(len(df)), "positives": int(y.sum()), "negatives": int((1 - y).sum()),
        "prevalence": float(y.mean()), "n_features": len(schema.FEATURES),
        "gender": {"female": int((df["gender"] == 0).sum()), "male": int((df["gender"] == 1).sum()),
                   "female_prevalence": float(y[df["gender"] == 0].mean()), "male_prevalence": float(y[df["gender"] == 1].mean())},
        "age": {"min": float(ages.min()), "max": float(ages.max()), "mean": float(ages.mean()),
                "mean_positive": float(ages[y == 1].mean()), "mean_negative": float(ages[y == 0].mean()),
                "histogram": age_hist},
        "features": feats,
        "max_abs_correlation": float(max(abs(f["correlation_with_cancer"]) for f in feats + [{"correlation_with_cancer": 0}])),
        "correlation_matrix": {"names": cols + ["lung_cancer"], "values": np.round(cm, 4).tolist()},
        "note": "Aggregate counts and correlations only. Every feature is nearly uncorrelated with the label.",
    }
    out = clean_json(out)
    (config.RESULTS_DIR / "eda.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    r = run()
    print(f"rows {r['rows']}, positives {r['positives']}, max |corr(feature, label)| = {r['max_abs_correlation']:.3f}")
