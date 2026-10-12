"""Runs every experiment (E1-E8 + sanity) and writes results/, models/.

    python -m ml.run_all            # full run (resumes from results/runs cache)
    python -m ml.run_all --force    # recompute everything
    python -m ml.run_all --quick    # 1 seed, 3 rounds, written to a temp dir (smoke test)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from ml import attack, centralized, comm_analysis, config, federated, fedavg_reference, metrics, model as M
from ml import partition, plots, preprocess, sanity, secagg

C = metrics.clean_json


def write(name: str, obj) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / name).write_text(json.dumps(C(obj), indent=1))


def cached(name: str, fn, force: bool):
    p = config.RESULTS_DIR / "runs" / f"{name}.json"
    if p.exists() and not force:
        return json.loads(p.read_text())
    out = C(fn())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    return out


def _net(params):
    n = M.make_model("mlp", 0)
    M.set_parameters(n, [np.asarray(a, dtype=np.float32) for a in params])
    return n


def _fl(scenario, seed, rounds, epochs=config.LOCAL_EPOCHS, sigma=0.0, secure=False):
    r = federated.run_fl(scenario, seed, rounds, epochs, sigma, secure)
    r["final_params"] = [a.tolist() for a in r["final_params"]]
    return r


def agg_rounds(runs: list[dict]) -> dict:
    n = len(runs[0]["rounds"])
    out = {"rounds": list(range(n)), "test": {}, "val": {}, "n_seeds": len(runs)}
    for split in ("test", "val"):
        for k in ("accuracy", "recall", "auc", "loss"):
            a = np.array([[r["rounds"][i][split][k] if r["rounds"][i][split][k] is not None else np.nan
                           for i in range(n)] for r in runs], dtype=float)
            out[split][k] = {"mean": np.nanmean(a, 0).tolist(), "std": np.nanstd(a, 0).tolist()}
    out["train_loss"] = [float(np.mean([r["rounds"][i].get("train_loss", np.nan) for r in runs])) if i else None
                         for i in range(n)]
    out["per_client"] = {}
    for h in runs[0]["rounds"][1]["per_client"]:
        out["per_client"][h] = {k: [None] + [float(np.mean([r["rounds"][i]["per_client"][h][k] for r in runs]))
                                              for i in range(1, n)] for k in ("accuracy", "recall", "auc")}
    return out


def method_row(mid, label, group, runs_test, **extra):
    return {"id": mid, "label": label, "group": group, "metrics": metrics.mean_std(runs_test), **extra}


def main(force=False, quick=False):
    seeds = [0] if quick else config.SEEDS
    rounds = 3 if quick else config.FL_ROUNDS
    if quick:
        config.RESULTS_DIR = Path(tempfile.mkdtemp(prefix="fl_quick_")) / "results"
        config.PLOTS_DIR = config.RESULTS_DIR / "plots"
        config.MODELS_DIR = config.RESULTS_DIR.parent / "models"
        force = True
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("results ->", config.RESULTS_DIR)

    # ---- data ----
    preprocess.run()
    partition.build_all()
    test_rows = len(M.load_xy(config.CLIENT_DIR / "test.csv")[1])
    Xte, yte = M.load_xy(config.CLIENT_DIR / "test.csv")
    methods, rocs = [], {}

    # ---- E1 centralized ----
    cen_attack, cen_runs = [], {}
    for kind in ("mlp", "logreg"):
        runs = []
        for s in seeds:
            net, m = centralized.run_centralized(s, kind)
            runs.append(m)
            if kind == "mlp":
                if s == seeds[0]:
                    M.save_model(net, "centralized")
                    rocs["centralized"] = metrics.roc_points(yte, M.predict_proba(net, Xte))
                cen_attack.append(attack.membership_inference(net, s))
        cen_runs[kind] = runs
        methods.append(method_row(f"centralized_{kind}", f"Centralized {'MLP' if kind == 'mlp' else 'LogReg'}",
                                  "centralized", runs))
        print("E1", kind, metrics.mean_std(runs)["auc"])

    # ---- E8 local-only ----
    for k in range(1, config.N_CLIENTS + 1):
        runs = [centralized.run_local_only("noniid", k, s)[1] for s in seeds]
        methods.append(method_row(f"local_h{k}", f"Local-only hospital {k}", "local", runs))
    print("E8 done")

    # ---- E2/E3 federated ----
    fl = {}
    fl_attack = {}
    for scen in ("iid", "noniid"):
        runs = [cached(f"fl_{scen}_s{s}", lambda s=s, scen=scen: _fl(scen, s, rounds), force) for s in seeds]
        fl[scen] = runs
        write(f"rounds_{scen}.json", agg_rounds(runs))
        methods.append(method_row(f"fl_{scen}", f"FedAvg ({scen.upper() if scen == 'iid' else 'non-IID'})", "fl",
                                  [r["rounds"][-1]["test"] for r in runs], n_params=runs[0]["n_params"]))
        if scen == "noniid":
            fl_attack["noniid"] = [attack.membership_inference(_net(r["final_params"]), s)
                                   for r, s in zip(runs, seeds)]
            net0 = _net(runs[0]["final_params"])
            M.save_model(net0, "federated")
            rocs["federated"] = metrics.roc_points(yte, M.predict_proba(net0, Xte))
        print("E2/E3", scen, metrics.mean_std([r["rounds"][-1]["test"] for r in runs])["auc"])

    # ---- E4 DP sweep ----
    curve = [{"sigma": 0.0, "epsilon": None, "epsilon_per_client": None,
              "metrics": metrics.mean_std([r["rounds"][-1]["test"] for r in fl["noniid"]])}]
    dp_attack = {}
    for sg in config.SIGMAS:
        runs = [cached(f"dp_{sg}_s{s}", lambda s=s, sg=sg: _fl("noniid", s, rounds, sigma=sg), force) for s in seeds]
        eps = runs[0]["epsilon_max"]
        tests = [r["rounds"][-1]["test"] for r in runs]
        curve.append({"sigma": sg, "epsilon": eps, "epsilon_per_client": runs[0]["epsilon_per_client"],
                      "metrics": metrics.mean_std(tests)})
        methods.append(method_row(f"fl_dp_{sg}", f"FedAvg + DP (σ={sg}, ε={eps:.1f})", "dp", tests, epsilon=eps, sigma=sg))
        dp_attack[sg] = [attack.membership_inference(_net(r["final_params"]), s) for r, s in zip(runs, seeds)]
        if sg == 1.0:
            net = _net(runs[0]["final_params"])
            M.save_model(net, "federated_dp")
            rocs["federated_dp"] = metrics.roc_points(yte, M.predict_proba(net, Xte))
        write(f"rounds_dp_{sg}.json", agg_rounds(runs))
        print("E4 sigma", sg, "eps", round(eps, 2), metrics.mean_std(tests)["auc"])
    write("privacy_curve.json", {"delta": config.DELTA, "clip_norm": config.CLIP_NORM, "rounds": rounds,
                                 "local_epochs": config.LOCAL_EPOCHS, "scope": "record-level DP vs the server",
                                 "points": curve})

    # ---- E5 secure aggregation ----
    sec_runs = [cached(f"sec_noniid_s{s}", lambda s=s: _fl("noniid", s, rounds, secure=True), force) for s in seeds]
    diffs = [max(float(np.abs(np.array(a) - np.array(b)).max()) for a, b in zip(p["final_params"], q["final_params"]))
             for p, q in zip(fl["noniid"], sec_runs)]
    auc_diff = [abs(p["rounds"][-1]["test"]["auc"] - q["rounds"][-1]["test"]["auc"])
                for p, q in zip(fl["noniid"], sec_runs)]
    methods.append(method_row("fl_secagg", "FedAvg + SecAgg (non-IID)", "secagg",
                              [r["rounds"][-1]["test"] for r in sec_runs]))
    # one real round of updates: exactness, hiding, dropout failure
    gp = M.get_parameters(M.make_model("mlp", 0))
    ups = [fedavg_reference.client_update(gp, "noniid", k, config.LOCAL_EPOCHS, 100 + k)
           for k in range(1, config.N_CLIENTS + 1)]
    flat = [np.concatenate([a.ravel() for a in u]).astype(np.float64) for u, _ in ups]
    cnt = [n for _, n in ups]
    ok = secagg.secure_weighted_average(flat, cnt, base_seed=1)
    bad = secagg.secure_weighted_average(flat, cnt, base_seed=1, drop=1)
    write("secagg_check.json", {
        "n_params": len(flat[0]), "client_counts": cnt, "modulus": "2^32", "fixed_point_bits": config.SECAGG_SCALE_BITS,
        "integer_masked_sum_vs_plain_sum_max_abs_diff": ok["int_max_abs_diff"],
        "float_error_vs_plain_fedavg_one_round": ok["float_max_abs_error"],
        "single_masked_update_correlation_with_true": [
            secagg.correlation_with_true(ok["masked"][i], flat[i] * cnt[i] / sum(cnt)) for i in range(len(flat))],
        "dropout_float_error": bad["float_max_abs_error"],
        "dropout_cancellation_failed": bool(bad["int_max_abs_diff"] > 0),
        "full_run": {"max_param_diff_vs_plain_fedavg": diffs, "test_auc_abs_diff": auc_diff, "rounds": rounds,
                     "note": "tiny fixed-point differences are amplified by SGD over rounds; metrics agree"},
    })
    print("E5 secagg int diff", ok["int_max_abs_diff"], "dropout err", bad["float_max_abs_error"])

    # ---- E6 communication ----
    k = config.N_CLIENTS
    n_params = fl["noniid"][0]["n_params"]
    cen_auc = metrics.mean_std(cen_runs["mlp"])["auc"]["mean"]
    comm_seeds = seeds[:3]
    by_e = []
    for e in ([1, 2] if quick else [1, 2, 5]):
        runs = [cached(f"comm_E{e}_s{s}", lambda s=s, e=e: _fl("noniid", s, rounds, epochs=e), force)
                for s in comm_seeds]
        tl = [float(np.mean([r["rounds"][i]["train_loss"] for r in runs])) for i in range(1, rounds + 1)]
        aucs = [float(np.mean([r["rounds"][i]["test"]["auc"] for r in runs])) for i in range(rounds + 1)]
        measured_up_down = float(np.mean([r["rounds"][1]["bytes_up"] + r["rounds"][1]["bytes_down"] for r in runs]))
        by_e.append({"local_epochs": e, **comm_analysis.analytic(n_params, k, rounds),
                     "measured_bytes_per_round_total": measured_up_down,
                     "mean_wall_s_per_round": float(np.mean([r["rounds"][i]["wall_s"] for r in runs
                                                             for i in range(1, rounds + 1)])),
                     "rounds_to_loss_convergence_10pct": comm_analysis.rounds_to_fraction_of_final_loss(tl),
                     "rounds_to_95pct_of_centralized_auc": comm_analysis.rounds_to_target_auc(aucs, 0.95 * cen_auc),
                     "final_test_auc": aucs[-1]})
    write("communication.json", {
        "model_params": n_params, "clients": k, "bytes_per_param": 4, "target_auc": 0.95 * cen_auc,
        "by_local_epochs": by_e, "raw_data": comm_analysis.raw_data_bytes("noniid"),
        "note": "Counts upload + download for all clients. AUC-based rounds-to-target is not meaningful when "
                "AUC ~ 0.5, so a training-loss convergence proxy is also reported."})

    # ---- E7 membership inference ----
    def row(label, lst, **kw):
        return {"label": label, "auc": metrics.mean_std([{**{k_: 0 for k_ in metrics.SCALAR_KEYS}, "auc": a["auc"]}
                                                         for a in lst])["auc"],
                "best_accuracy": float(np.mean([a["best_accuracy"] for a in lst])), **kw}
    models = [row("Centralized MLP", cen_attack), row("FedAvg (non-IID)", fl_attack["noniid"])]
    models += [row(f"FedAvg + DP (σ={sg})", dp_attack[sg], sigma=sg) for sg in config.SIGMAS]
    write("attack.json", {
        "attack": "loss-threshold membership inference", "non_members": "dedicated hold-out (attack.csv)",
        "models": models,
        "positive_control": row("Positive control (overfit 256-256 MLP, 150 rows)",
                                [attack.positive_control(s) for s in seeds], control="positive"),
        "untrained_control": row("Negative control (untrained model)",
                                 [attack.untrained_control(s) for s in seeds], control="negative")})
    print("E7 done")

    # ---- comparison ----
    n1 = int(yte.sum()); n0 = len(yte) - n1
    se = float(np.sqrt((n0 + n1 + 1) / (12 * n0 * n1)))
    write("comparison.json", {"test_rows": test_rows, "chance_band_auc": 1.96 * se, "methods": methods,
                              "matched_budget_epochs": centralized.MATCHED_EPOCHS})
    with open(config.RESULTS_DIR / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", *[f"{k_}_{s}" for k_ in metrics.SCALAR_KEYS for s in ("mean", "std")]])
        for m in methods:
            w.writerow([m["label"]] + [None if m["metrics"][k_][s] is None else round(m["metrics"][k_][s], 4)
                                       for k_ in metrics.SCALAR_KEYS for s in ("mean", "std")])
    for name, r in rocs.items():
        write(f"roc_{name}.json", r)

    # ---- sanity (synthetic planted signal) ----
    write("sanity_planted_signal.json", sanity.run(seeds[:3], rounds))

    # ---- meta + plots ----
    import flwr, opacus
    write("run_meta.json", {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
        "platform": platform.platform(), "torch": torch.__version__, "flwr": flwr.__version__,
        "opacus": opacus.__version__, "numpy": np.__version__, "seeds": seeds, "rounds": rounds,
        "hyperparameters": {"lr": config.LR, "momentum": config.MOMENTUM, "batch_size": config.BATCH_SIZE,
                            "local_epochs": config.LOCAL_EPOCHS, "hidden": list(config.HIDDEN),
                            "clip_norm": config.CLIP_NORM, "sigmas": config.SIGMAS, "delta": config.DELTA,
                            "split_fractions": config.SPLIT_FRACTIONS, "quantity_skew": config.QUANTITY_SKEW,
                            "dirichlet_alpha": config.DIRICHLET_ALPHA},
        "flower_mode": "in-process NumPyClient + flwr.server.strategy.FedAvg (no ray/gRPC)"})
    plots.make_all()
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(a.force, a.quick)
