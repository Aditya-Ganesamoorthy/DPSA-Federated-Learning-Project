import numpy as np
import pandas as pd
import pytest

from ml import config, partition, preprocess, schema


@pytest.fixture(scope="module")
def clean():
    return preprocess.run()


def test_no_identifiers_no_nan_and_feature_order(clean):
    assert not set(clean.columns) & set(schema.IDENTIFIER_HINTS)
    assert not clean.isna().any().any()
    assert list(clean.columns[: schema.N_FEATURES]) == schema.FEATURES
    assert set(clean[schema.TARGET].unique()) <= {0, 1}


def test_duplicates_removed(clean):
    raw = preprocess.load_raw()
    assert len(clean) == len(raw.drop_duplicates())


def test_encoding_maps():
    raw = pd.DataFrame([{"gender": "M", "age": 50, **{c: 2 for c in schema.BINARY_FEATURES},
                         "lung_cancer": "YES"}])
    enc = schema.encode_frame(raw)
    assert enc.loc[0, "gender"] == 1 and enc.loc[0, "age"] == 0.5
    assert enc.loc[0, schema.BINARY_FEATURES].sum() == len(schema.BINARY_FEATURES)
    assert enc.loc[0, schema.TARGET] == 1


def test_api_encoding_matches_training_encoding(clean):
    raw = preprocess.load_raw().drop_duplicates().reset_index(drop=True)
    row = raw.iloc[0]
    rec = {"gender": row["gender"], "age": int(row["age"]),
           **{c: int(row[c]) == 2 for c in schema.BINARY_FEATURES}}
    np.testing.assert_allclose(schema.encode_api_record(rec),
                               clean.loc[0, schema.FEATURES].to_numpy(dtype=np.float32))


def _split():
    c = pd.read_csv(config.PROCESSED_CSV)
    c["_row"] = np.arange(len(c))
    return c, partition.global_split(c)


@pytest.mark.parametrize("scen", ["iid", "noniid"])
def test_partition_guarantees(clean, scen):
    c, s = _split()
    fn = partition.iid_partition if scen == "iid" else partition.noniid_partition
    parts = fn(s["train"], config.N_CLIENTS, config.SPLIT_SEED) if scen == "iid" \
        else fn(s["train"], config.SPLIT_SEED)
    partition.check_partition(parts, s["train"], [s["val"], s["test"], s["attack"]])
    assert sum(len(p) for p in parts) == len(s["train"])


def test_split_sets_are_disjoint_and_stratified(clean):
    c, s = _split()
    rows = [set(v["_row"]) for v in s.values()]
    assert sum(map(len, rows)) == len(set().union(*rows)) == len(c)
    for v in s.values():
        assert abs(v[schema.TARGET].mean() - c[schema.TARGET].mean()) < 0.02


def test_noniid_is_actually_skewed(clean):
    c, s = _split()
    parts = partition.noniid_partition(s["train"], config.SPLIT_SEED)
    prev = [p[schema.TARGET].mean() for p in parts]
    assert max(prev) - min(prev) > 0.05
    assert len({len(p) for p in parts}) == 3


def test_checker_detects_leak(clean):
    c, s = _split()
    parts = partition.iid_partition(s["train"], 3, 0)
    with pytest.raises(AssertionError):
        partition.check_partition(parts, s["train"], [s["train"].head(5)])
