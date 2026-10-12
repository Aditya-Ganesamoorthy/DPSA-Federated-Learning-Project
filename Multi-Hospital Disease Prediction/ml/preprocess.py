"""Cleaning + privacy transforms. Writes data/processed/clean.csv and results/preprocess_report.json."""
from __future__ import annotations

import json

import pandas as pd

from ml import config, schema


def load_raw() -> pd.DataFrame:
    return schema.normalize_columns(pd.read_csv(config.RAW_CSV, encoding="utf-8-sig"))


def run() -> pd.DataFrame:
    raw = load_raw()
    report = {"raw_rows": len(raw), "columns": raw.columns.tolist(),
              "missing_values": int(raw.isna().sum().sum())}

    # direct identifiers: drop if present (expected: none) and assert none remain
    ids = [c for c in raw.columns if c in schema.IDENTIFIER_HINTS]
    report["identifier_columns_dropped"] = ids
    raw = raw.drop(columns=ids)

    # remove exact duplicates BEFORE any split
    before = len(raw)
    raw = raw.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = before - len(raw)

    clean = schema.encode_frame(raw)
    clean["age_band"] = schema.age_band(raw["age"])   # generalised quasi-identifier, NOT a model feature

    assert not [c for c in clean.columns if c in schema.IDENTIFIER_HINTS]
    assert list(clean.columns[: schema.N_FEATURES]) == schema.FEATURES
    assert set(clean[schema.TARGET].unique()) <= {0, 1}
    assert not clean.isna().any().any()

    report.update(clean_rows=len(clean), positives=int(clean[schema.TARGET].sum()),
                  prevalence=float(clean[schema.TARGET].mean()))
    k = clean.groupby(schema.QUASI_IDENTIFIERS).size()
    report["k_anonymity_pooled"] = int(k.min())
    report["quasi_identifier_groups"] = int(len(k))
    report["age_min"], report["age_max"] = int(raw["age"].min()), int(raw["age"].max())

    config.PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(config.PROCESSED_CSV, index=False)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "preprocess_report.json").write_text(json.dumps(report, indent=2))
    return clean


if __name__ == "__main__":
    run()
    print(json.loads((config.RESULTS_DIR / "preprocess_report.json").read_text()))
