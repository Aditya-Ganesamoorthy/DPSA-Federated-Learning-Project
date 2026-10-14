"""Generate four SYNTHETIC hospital CSVs (about 300 rows each) for demos, in the same layout as the Kaggle file:

    GENDER,AGE,SMOKING,...,CHEST_PAIN,LUNG_CANCER      (symptom fields coded 1 = no / 2 = yes, label YES / NO)

Every row is invented: values are drawn from the overall statistics of the original dataset (about 50% yes for each
symptom, ages 30-80 with mean 55, about 50% cancer) with small per-hospital differences so the sites are slightly
non-IID, like real hospitals. As in the original data, the label is independent of the features, so models trained on
these files also score near chance (AUC about 0.5). No row is copied from the original file.

    python scripts/make_demo_data.py            # writes demo_data/*.csv (deterministic, seed 2026)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo_data"
ORIGINAL = ROOT / "Full Lung Cancer dataset.csv"

SYMPTOMS = ["SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC_DISEASE", "FATIGUE", "ALLERGY",
            "WHEEZING", "ALCOHOL_CONSUMING", "COUGHING", "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN"]
COLUMNS = ["GENDER", "AGE", *SYMPTOMS, "LUNG_CANCER"]

# name, rows, cancer rate, mean age, share of male patients, symptom-rate shift (added to the overall ~0.50 rate)
HOSPITALS = [
    ("hospital_A_city_general",    300, 0.46, 52.0, 0.52, +0.03),
    ("hospital_B_district_medical", 310, 0.57, 58.5, 0.48, -0.02),
    ("hospital_C_regional_clinic",  290, 0.52, 55.0, 0.55, +0.00),
    ("hospital_D_community_care",   300, 0.49, 60.5, 0.46, -0.03),
]


def make_one(rng: np.random.Generator, n: int, cancer: float, mean_age: float, male: float, shift: float,
             base_rates: dict[str, float]) -> pd.DataFrame:
    age = np.clip(np.round(rng.normal(mean_age, 14.5, n)), 30, 80).astype(int)
    df = pd.DataFrame({"GENDER": np.where(rng.random(n) < male, "M", "F"), "AGE": age})
    for c in SYMPTOMS:
        p = float(np.clip(base_rates[c] + shift + rng.normal(0, 0.02), 0.35, 0.65))
        df[c] = np.where(rng.random(n) < p, 2, 1)
    df["LUNG_CANCER"] = np.where(rng.random(n) < cancer, "YES", "NO")
    return df[COLUMNS]


def main() -> int:
    orig = pd.read_csv(ORIGINAL)
    base = {c: float((orig[c] == 2).mean()) for c in SYMPTOMS}
    orig_rows = set(map(tuple, orig[COLUMNS].astype(str).to_numpy().tolist()))
    rng = np.random.default_rng(2026)
    OUT.mkdir(exist_ok=True)
    seen: set[tuple] = set(orig_rows)
    for name, n, cancer, mean_age, male, shift in HOSPITALS:
        while True:                                    # resample until every row is unique and not in the original
            df = make_one(rng, n, cancer, mean_age, male, shift, base)
            keys = list(map(tuple, df.astype(str).to_numpy().tolist()))
            dup_inside = len(keys) - len(set(keys))
            clash = sum(k in seen for k in keys)
            if dup_inside == 0 and clash == 0:
                break
            df = df.loc[[k not in seen for k in keys]].drop_duplicates()
            if len(df) >= n - 3:                       # tiny losses: top up instead of redrawing everything
                extra = make_one(rng, n - len(df) + 10, cancer, mean_age, male, shift, base)
                df = pd.concat([df, extra]).drop_duplicates().head(n)
                keys = list(map(tuple, df.astype(str).to_numpy().tolist()))
                if len(df) == n and not any(k in seen for k in keys) and len(set(keys)) == n:
                    break
        seen.update(map(tuple, df.astype(str).to_numpy().tolist()))
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(df)} rows, cancer {(df.LUNG_CANCER == 'YES').mean():.1%}, mean age {df.AGE.mean():.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
