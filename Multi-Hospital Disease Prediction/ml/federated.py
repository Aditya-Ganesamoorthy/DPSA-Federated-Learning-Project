"""Federated learning with Flower (flwr 1.39): FedAvg, FedAvg + DP-SGD, FedAvg + SecAgg.

Flower pieces actually used: `NumPyClient` subclasses (one per hospital), the real
`flwr.server.strategy.FedAvg` aggregation (`aggregate_fit` + `evaluate`), Flower `Parameters`/`FitRes`
types. Ray/gRPC simulation is NOT used (it does not install reliably on Windows, see plan edge case
#20): clients are called in-process, which is what the simulation engine does anyway.

Key principle: hospital CSVs are opened ONLY inside `HospitalClient`. Server code touches only the
shared test/val files.
"""
from __future__ import annotations

import logging
import time
import warnings

import numpy as np
from flwr.client import NumPyClient
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from ml import config, dp, metrics, model as M, secagg

warnings.filterwarnings("ignore")
logging.getLogger("flwr").setLevel(logging.ERROR)


class _Proxy(ClientProxy):
    """Minimal ClientProxy so Flower's strategy can aggregate in-process results."""

    def get_properties(self, *a, **k): ...
    def get_parameters(self, *a, **k): ...
    def fit(self, *a, **k): ...
    def evaluate(self, *a, **k): ...
    def reconnect(self, *a, **k): ...


class HospitalClient(NumPyClient):
    """One hospital. Loads ONLY its own CSV."""

    def __init__(self, scenario: str, idx: int, store: dp.AccountantStore | None = None, label_flip: bool = False):
        self.idx = idx
        self.cid = f"hospital_{idx}"
        self.X, self.y = M.load_xy(M.client_path(scenario, idx))
        if label_flip:                       # malicious hospital: flips every training label
            self.y = 1.0 - self.y
        self.store = store

    def get_parameters(self, config_):
        return M.get_parameters(M.make_model("mlp", 0))

    def fit(self, parameters, cfg):
        net = M.make_model("mlp", 0)
        M.set_parameters(net, parameters)
        seed, epochs, sigma = int(cfg["seed"]), int(cfg["epochs"]), float(cfg.get("sigma", 0.0))
        if sigma > 0:
            loss = dp.train_local_dp(net, self.X, self.y, epochs, seed, sigma, self.store, self.cid)
        else:
            loss = M.train_local(net, self.X, self.y, epochs, seed, prox_mu=float(cfg.get("prox_mu", 0.0)),
                                 prox_ref=parameters)
        n = len(self.X)
        out = M.get_parameters(net)
        qb = int(cfg.get("quant_bits", 0))
        if qb:                               # lossy compression of the UPDATE (delta), per-tensor symmetric
            out = [g + quantize_delta(o - g, qb) for o, g in zip(out, parameters)]
        if cfg.get("secagg", False):
            flat = np.concatenate([a.ravel() for a in out]).astype(np.float64)
            w = flat * (n / float(cfg["total_n"]))                       # pre-weight: plain sum == FedAvg
            masked = secagg.mask_update(self.idx - 1, int(cfg["n_clients"]), secagg.quantize(w),
                                        int(cfg["secagg_seed"]))
            out = [masked.astype(np.uint32)]                              # only the masked vector leaves
        return out, n, {"train_loss": float(loss)}

    def evaluate(self, parameters, cfg):
        net = M.make_model("mlp", 0)
        M.set_parameters(net, parameters)
        m = metrics.compute(self.y, M.predict_proba(net, self.X))
        return float(m["loss"]), len(self.X), {k: m[k] for k in ("accuracy", "recall", "auc")}


def quantize_delta(delta: np.ndarray, bits: int) -> np.ndarray:
    """Uniform symmetric quantisation of an update tensor to `bits` bits (returned dequantised)."""
    m = float(np.max(np.abs(delta)))
    if m == 0:
        return delta
    levels = 2 ** (bits - 1) - 1
    return (np.round(delta / m * levels) / levels * m).astype(np.float32)


class MedianFedAvg(FedAvg):
    """Robust aggregation: coordinate-wise median of client models (ignores sample counts)."""

    def aggregate_fit(self, server_round, results, failures):
        arrs = [parameters_to_ndarrays(r.parameters) for _, r in results]
        med = [np.median(np.stack([a[i] for a in arrs]), axis=0).astype(np.float32) for i in range(len(arrs[0]))]
        return ndarrays_to_parameters(med), {}


class SecAggFedAvg(FedAvg):
    """FedAvg whose server only ever sees masked uint32 vectors; masks cancel in the modular sum."""

    def __init__(self, shapes, n_clients, **kw):
        super().__init__(**kw)
        self.shapes, self.n_clients = shapes, n_clients

    def aggregate_fit(self, server_round, results, failures):
        if failures or len(results) != self.n_clients:
            # masks of the missing client(s) are not cancelled -> result would be garbage. Abort the round.
            return None, {"secagg_error": 1.0}
        masked = [parameters_to_ndarrays(r.parameters)[0].astype(np.uint64) for _, r in results]
        flat = secagg.dequantize(secagg.aggregate_masked(masked)).astype(np.float32)
        out, pos = [], 0
        for s in self.shapes:
            size = int(np.prod(s)) if len(s) else 1
            out.append(flat[pos:pos + size].reshape(s))
            pos += size
        return ndarrays_to_parameters(out), {}


def _bytes(arrs) -> int:
    return int(sum(a.nbytes for a in arrs))


def run_fl(scenario: str, seed: int, rounds: int = config.FL_ROUNDS, local_epochs: int = config.LOCAL_EPOCHS,
           sigma: float = 0.0, secure: bool = False, verbose: bool = False, prox_mu: float = 0.0,
           quant_bits: int = 0, poisoned_client: int | None = None, aggregator: str = "fedavg") -> dict:
    """Run one federated experiment. Returns per-round log, final params and (if DP) epsilon."""
    Xte, yte = M.load_xy(config.CLIENT_DIR / "test.csv")
    Xva, yva = M.load_xy(config.CLIENT_DIR / "val.csv")
    net = M.make_model("mlp", seed)
    init = M.get_parameters(net)
    shapes = [a.shape for a in init]
    K = config.N_CLIENTS
    counts = [len(M.load_xy(M.client_path(scenario, k))[1]) for k in range(1, K + 1)]  # public sizes only
    store = dp.AccountantStore()
    log: list[dict] = []

    def evaluate_fn(rnd, ndarrays, _cfg):
        e = M.make_model("mlp", 0)
        M.set_parameters(e, ndarrays)
        t = metrics.compute(yte, M.predict_proba(e, Xte))
        v = metrics.compute(yva, M.predict_proba(e, Xva))
        log[-1]["test"], log[-1]["val"] = t, v
        return t["loss"], {}

    common = dict(fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=K, min_available_clients=K,
                  evaluate_fn=evaluate_fn, accept_failures=False)
    strategy = (SecAggFedAvg(shapes, K, **common) if secure
                else MedianFedAvg(**common) if aggregator == "median" else FedAvg(**common))
    params = ndarrays_to_parameters(init)

    log.append({"round": 0})
    strategy.evaluate(0, params)
    for rnd in range(1, rounds + 1):
        t0 = time.time()
        log.append({"round": rnd})
        cur = parameters_to_ndarrays(params)
        results, up, down = [], 0, 0
        for k in range(1, K + 1):
            # Flower re-creates clients every round: the accountant store lives outside them.
            client = HospitalClient(scenario, k, store, label_flip=(k == poisoned_client))
            cfg = {"seed": seed * 10_000 + rnd * 100 + k, "epochs": local_epochs, "sigma": sigma,
                   "prox_mu": prox_mu, "quant_bits": quant_bits,
                   "secagg": secure, "total_n": sum(counts), "n_clients": K,
                   "secagg_seed": seed * 100_000 + rnd}
            new, n, fm = client.fit([a.copy() for a in cur], cfg)
            down += _bytes(cur)
            up += (sum(int(np.ceil(a.size * quant_bits / 8)) + 4 for a in new) if quant_bits else _bytes(new))
            res = FitRes(status=Status(Code.OK, "ok"), parameters=ndarrays_to_parameters(new),
                         num_examples=n, metrics=fm)
            results.append((_Proxy(str(k)), res))
        params, agg_m = strategy.aggregate_fit(rnd, results, [])
        if params is None:
            raise RuntimeError(f"aggregation aborted: {agg_m}")
        strategy.evaluate(rnd, params)
        glob = parameters_to_ndarrays(params)
        per_client = {}
        for k in range(1, K + 1):
            _, _, em = HospitalClient(scenario, k).evaluate(glob, {})
            per_client[f"hospital_{k}"] = em
        total_n = sum(r.num_examples for _, r in results)
        log[-1].update({
            "bytes_up": up, "bytes_down": down, "wall_s": time.time() - t0, "per_client": per_client,
            "train_loss": sum(r.metrics["train_loss"] * r.num_examples for _, r in results) / total_n
            if not secure or True else None,
            "eps_max": store.max_epsilon() if sigma > 0 else None,
        })
        if verbose:
            t = log[-1]["test"]
            print(f"[{scenario} s{seed} sigma={sigma} secure={secure}] round {rnd}: "
                  f"acc={t['accuracy']:.3f} auc={t['auc']:.3f}")
    final = parameters_to_ndarrays(params)
    return {"rounds": log, "final_params": final, "epsilon_per_client": store.per_client() if sigma > 0 else None,
            "epsilon_max": store.max_epsilon() if sigma > 0 else None, "n_params": int(sum(a.size for a in final)),
            "client_counts": counts}
