"""Global split + hospital partitions (IID and non-IID) + k-anonymity per client."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml import config, schema

SCENARIOS = ("iid", "noniid")


def global_split(df: pd.DataFrame, seed: int = config.SPLIT_SEED) -> dict[str, pd.DataFrame]:
    """Stratified train/val/test/attack split. `attack` rows are only ever non-members for the MIA."""
    f = config.SPLIT_FRACTIONS
    train, rest = train_test_split(df, train_size=f["train"], stratify=df[schema.TARGET], random_state=seed)
    rest_frac = 1 - f["train"]
    val, rest2 = train_test_split(rest, train_size=f["val"] / rest_frac,
                                  stratify=rest[schema.TARGET], random_state=seed)
    test, attack = train_test_split(rest2, train_size=f["test"] / (rest_frac - f["val"]),
                                    stratify=rest2[schema.TARGET], random_state=seed)
    return {k: v.reset_index(drop=True) for k, v in
            dict(train=train, val=val, test=test, attack=attack).items()}


def iid_partition(train: pd.DataFrame, n_clients: int, seed: int) -> list[pd.DataFrame]:
    parts = [[] for _ in range(n_clients)]
    rng = np.random.default_rng(seed)
    for _, g in train.groupby(schema.TARGET):
        idx = rng.permutation(g.index.to_numpy())
        for k, chunk in enumerate(np.array_split(idx, n_clients)):
            parts[k].extend(chunk.tolist())
    return [train.loc[sorted(p)].reset_index(drop=True) for p in parts]


def noniid_partition(train: pd.DataFrame, seed: int, n_clients: int = config.N_CLIENTS,
                     quantity=config.QUANTITY_SKEW, alpha: float = config.DIRICHLET_ALPHA
                     ) -> list[pd.DataFrame]:
    """Quantity skew (fixed proportions) + label skew (Dirichlet(alpha) prevalence per client).

    Resamples until every client has both classes, >= MIN_CLIENT_ROWS and prevalence in bounds.
    """
    rng = np.random.default_rng(seed)
    n = len(train)
    sizes = np.floor(np.array(quantity) * n).astype(int)
    sizes[0] += n - sizes.sum()
    pos_total = int(train[schema.TARGET].sum())
    lo, hi = config.PREVALENCE_BOUNDS
    for _ in range(10_000):
        prev = rng.dirichlet([alpha, alpha], size=n_clients)[:, 1]
        w = sizes * prev
        pos = np.round(pos_total * w / w.sum()).astype(int)
        pos[-1] = pos_total - pos[:-1].sum()
        neg = sizes - pos
        if (neg < 0).any() or (pos < 0).any() or sizes.min() < config.MIN_CLIENT_ROWS:
            continue
        p = pos / sizes
        if p.min() >= lo and p.max() <= hi:
            break
    else:
        raise RuntimeError("Could not find a feasible non-IID partition")
    pos_idx = rng.permutation(train.index[train[schema.TARGET] == 1].to_numpy())
    neg_idx = rng.permutation(train.index[train[schema.TARGET] == 0].to_numpy())
    parts, pi, ni = [], 0, 0
    for k in range(n_clients):
        idx = np.concatenate([pos_idx[pi:pi + pos[k]], neg_idx[ni:ni + neg[k]]])
        pi, ni = pi + pos[k], ni + neg[k]
        parts.append(train.loc[np.sort(idx)].reset_index(drop=True))
    return parts


def check_partition(parts: list[pd.DataFrame], train: pd.DataFrame, others: list[pd.DataFrame]) -> None:
    """Asserts: disjoint, union == train, both classes, min rows, no overlap with val/test/attack.

    Row identity is the `_row` column (original row id), not feature values, because
    survey rows can legitimately share identical feature vectors.
    """
    ids = [set(p["_row"]) for p in parts]
    assert sum(len(i) for i in ids) == len(set().union(*ids)), "clients overlap"
    assert set().union(*ids) == set(train["_row"]), "union != train"
    for p in parts:
        assert p[schema.TARGET].nunique() == 2, "client missing a class"
        assert len(p) >= config.MIN_CLIENT_ROWS, "client too small"
    for o in others:
        assert not (set().union(*ids) & set(o["_row"])), "client row leaked into val/test/attack"


def k_anonymity(df: pd.DataFrame) -> int:
    return int(df.groupby(schema.QUASI_IDENTIFIERS).size().min())


def summarize(parts: list[pd.DataFrame]) -> list[dict]:
    out = []
    for i, p in enumerate(parts, 1):
        out.append({
            "hospital": f"hospital_{i}", "rows": len(p), "positives": int(p[schema.TARGET].sum()),
            "prevalence": float(p[schema.TARGET].mean()),
            "age_bands": p["age_band"].value_counts().sort_index().to_dict(),
            "k_anonymity": k_anonymity(p),
        })
    return out


def build_all() -> dict:
    clean = pd.read_csv(config.PROCESSED_CSV)
    clean["_row"] = np.arange(len(clean))
    s = global_split(clean)
    train = s["train"]
    out = {"iid": iid_partition(train, config.N_CLIENTS, config.SPLIT_SEED),
           "noniid": noniid_partition(train, config.SPLIT_SEED)}
    others = [s["val"], s["test"], s["attack"]]
    for parts in out.values():
        check_partition(parts, train, others)

    config.CLIENT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("val", "test", "attack"):
        s[name].to_csv(config.CLIENT_DIR / f"{name}.csv", index=False)
    train.to_csv(config.CLIENT_DIR / "train_pool.csv", index=False)   # centralized baseline ONLY
    for scen, parts in out.items():
        for i, p in enumerate(parts, 1):
            p.to_csv(config.CLIENT_DIR / f"{scen}_hospital_{i}.csv", index=False)

    summary = {
        "split_rows": {k: len(v) for k, v in s.items()},
        "k_anonymity_pooled": k_anonymity(clean),
        "scenarios": {scen: summarize(parts) for scen, parts in out.items()},
        "config": {"quantity_skew": config.QUANTITY_SKEW, "dirichlet_alpha": config.DIRICHLET_ALPHA,
                   "split_seed": config.SPLIT_SEED},
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "client_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    s = build_all()
    print(s["split_rows"], "pooled k =", s["k_anonymity_pooled"])
    for scen, rows in s["scenarios"].items():
        for r in rows:
            print(scen, r["hospital"], r["rows"], round(r["prevalence"], 3), "k =", r["k_anonymity"])
