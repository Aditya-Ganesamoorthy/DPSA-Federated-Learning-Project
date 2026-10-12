import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from ml import attack, centralized, config, dp, federated, fedavg_reference, metrics, model as M, secagg


def test_parameter_roundtrip():
    a = M.make_model("mlp", 1)
    b = M.make_model("mlp", 2)
    M.set_parameters(b, M.get_parameters(a))
    for x, y in zip(M.get_parameters(a), M.get_parameters(b)):
        np.testing.assert_array_equal(x, y)
    assert all(p.dtype == np.float32 for p in M.get_parameters(a))


def test_weighted_average_uses_sample_counts():
    u = [[np.array([1.0, 1.0], np.float32)], [np.array([5.0, 9.0], np.float32)]]
    out = fedavg_reference.weighted_average(u, [1, 3])
    np.testing.assert_allclose(out[0], [4.0, 7.0])


def test_flower_fedavg_equals_manual_weighted_average():
    class P(federated._Proxy):
        pass
    w = [[np.full((2, 2), 1, np.float32), np.zeros(2, np.float32)],
         [np.full((2, 2), 4, np.float32), np.ones(2, np.float32)]]
    counts = [1, 3]
    res = [(P("a"), FitRes(status=Status(Code.OK, ""), parameters=ndarrays_to_parameters([x.copy() for x in w[i]]),
                           num_examples=counts[i], metrics={})) for i in range(2)]
    agg, _ = FedAvg(min_fit_clients=2, min_available_clients=2).aggregate_fit(1, res, [])
    ref = fedavg_reference.weighted_average(w, counts)
    for a, b in zip(parameters_to_ndarrays(agg), ref):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_flower_round_matches_reference_loop():
    """One full federated round (real client training) equals the hand-written FedAvg."""
    fl = federated.run_fl("noniid", 3, rounds=1)["final_params"]
    ref = fedavg_reference.run("noniid", 3, rounds=1)
    for a, b in zip(fl, ref):
        np.testing.assert_allclose(a, b, atol=1e-5)


# ---- secure aggregation ----
def _updates(k=3, size=50, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=size) for _ in range(k)], [100, 60, 40]


def test_secagg_integer_sum_exact_and_matches_fedavg():
    ups, counts = _updates()
    r = secagg.secure_weighted_average(ups, counts, base_seed=7)
    assert r["int_max_abs_diff"] == 0
    assert r["float_max_abs_error"] < 1e-5


def test_secagg_masked_update_hides_true_update():
    ups, counts = _updates(size=2000)
    r = secagg.secure_weighted_average(ups, counts, base_seed=1)
    true = ups[0] * counts[0] / sum(counts)
    assert abs(secagg.correlation_with_true(r["masked"][0], true)) < 0.1


def test_secagg_dropout_breaks_cancellation():
    ups, counts = _updates()
    r = secagg.secure_weighted_average(ups, counts, base_seed=7, drop=1)
    assert r["int_max_abs_diff"] > 0 and r["float_max_abs_error"] > 1


def test_secagg_federated_strategy_aborts_on_missing_client():
    s = federated.SecAggFedAvg([(4,)], 3, min_fit_clients=3, min_available_clients=3)
    p, m = s.aggregate_fit(1, [], [])
    assert p is None and "secagg_error" in m


def test_secagg_fl_round_matches_plain_fedavg():
    plain = federated.run_fl("noniid", 0, rounds=1)["final_params"]
    sec = federated.run_fl("noniid", 0, rounds=1, secure=True)["final_params"]
    for a, b in zip(plain, sec):
        np.testing.assert_allclose(a, b, atol=1e-5)


# ---- differential privacy ----
def test_epsilon_increases_as_sigma_decreases():
    eps = []
    for sigma in (2.0, 1.0, 0.5):
        r = federated.run_fl("noniid", 0, rounds=2, sigma=sigma)
        eps.append(r["epsilon_max"])
    assert eps[0] < eps[1] < eps[2]


def test_accountant_persists_across_rounds():
    one = federated.run_fl("noniid", 0, rounds=1, sigma=1.0)["epsilon_max"]
    three = federated.run_fl("noniid", 0, rounds=3, sigma=1.0)["epsilon_max"]
    assert three > one


def test_accountant_store_composes_history():
    s = dp.AccountantStore()
    s.add("c", [(1.0, 0.05, 100)])
    e1 = s.epsilon("c")
    s.add("c", [(1.0, 0.05, 100)])
    assert s.epsilon("c") > e1


def test_dp_noise_scale_controls_distance_from_non_dp():
    """Poisson-sampled DP-SGD batches differ from shuffled SGD, so exact equality is not expected.
    With negligible noise the model must land much closer to non-DP training than with heavy noise."""
    X, y = M.load_xy(M.client_path("iid", 1))
    flat = lambda m: np.concatenate([a.ravel() for a in M.get_parameters(m)])
    base = M.make_model("mlp", 0)
    M.train_local(base, X, y, 1, 0)

    def dist(sigma):
        net = M.make_model("mlp", 0)
        dp.train_local_dp(net, X, y, 1, 0, sigma=sigma, store=dp.AccountantStore(), client_id="t")
        return np.linalg.norm(flat(net) - flat(base))

    assert dist(1e-6) < 0.5 * dist(10.0)


# ---- metrics / attack ----
def test_auc_guard_single_class():
    m = metrics.compute(np.ones(10), np.linspace(0, 1, 10))
    assert np.isnan(m["auc"])
    assert metrics.clean_json(m)["auc"] is None


def test_attack_positive_control_beats_chance_and_untrained_near_chance():
    pc = attack.positive_control(0)
    assert pc["auc"] > 0.65
    aucs = [attack.untrained_control(s)["auc"] for s in range(5)]
    assert abs(np.mean(aucs) - 0.5) < 0.08


# ---- pipeline sanity: the pipeline CAN learn when signal exists ----
def test_pipeline_learns_planted_signal(monkeypatch):
    from ml import sanity
    tmp = sanity.build_synthetic_dir()
    monkeypatch.setattr(config, "CLIENT_DIR", tmp)
    _, cen = centralized.run_centralized(0)
    fl = federated.run_fl("noniid", 0, rounds=8)["rounds"][-1]["test"]
    # the planted rule's oracle AUC on the test set is ~0.83 (Bernoulli label noise), so >0.74 is near-ceiling
    assert cen["auc"] > 0.74 and fl["auc"] > 0.74
