"""matplotlib figures built from results/*.json. Palette from the validated default (light surface);
identity is never colour-only: every series also has a marker and a direct label."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml import config

BLUE, ORANGE, AQUA, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "figure.dpi": 130})


def _load(name):
    return json.loads((config.RESULTS_DIR / name).read_text())


def _save(fig, name):
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / name)
    plt.close(fig)


def comparison():
    d = _load("comparison.json")
    rows = [m for m in d["methods"] if m["group"] != "local"] + [m for m in d["methods"] if m["group"] == "local"]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.4))
    lo, hi = 0.5 - d["chance_band_auc"], 0.5 + d["chance_band_auc"]
    ax.axvspan(lo, hi, color=GRID, alpha=0.7, label=f"chance band (95%, n={d['test_rows']})")
    ax.axvline(0.5, color=MUTED, lw=1)
    for i, m in enumerate(rows):
        a = m["metrics"]["auc"]
        ax.errorbar(a["mean"], i, xerr=a["std"], fmt="o", color=BLUE, ms=6, capsize=3, lw=2)
    ax.set_yticks(range(len(rows)), [m["label"] for m in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Test ROC-AUC (mean ± std over seeds)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("All methods sit inside the chance band: the label carries no learnable signal")
    _save(fig, "comparison_auc.png")


def training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for scen, c, mk in (("iid", BLUE, "o"), ("noniid", ORANGE, "s")):
        d = _load(f"rounds_{scen}.json")
        r = d["rounds"]
        for ax, key in zip(axes, ("auc", "recall")):
            ax.plot(r, d["test"][key]["mean"], color=c, marker=mk, ms=4, lw=2, label=f"FL {scen}")
            lo = [m - s for m, s in zip(d["test"][key]["mean"], d["test"][key]["std"])]
            hi = [m + s for m, s in zip(d["test"][key]["mean"], d["test"][key]["std"])]
            ax.fill_between(r, lo, hi, color=c, alpha=0.15)
    for ax, key in zip(axes, ("auc", "recall")):
        ax.axhline(0.5, color=MUTED, ls="--", lw=1)
        ax.set_xlabel("Round")
        ax.set_ylabel(f"Global model test {key.upper() if key == 'auc' else key}")
    axes[0].legend(frameon=False)
    fig.suptitle("Global-model test metrics per round (dashed = chance)")
    _save(fig, "training_curves.png")


def privacy_curve():
    d = _load("privacy_curve.json")["points"]
    pts = [p for p in d if p["epsilon"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, key in zip(axes, ("auc", "recall")):
        ax.errorbar([p["epsilon"] for p in pts], [p["metrics"][key]["mean"] for p in pts],
                    yerr=[p["metrics"][key]["std"] for p in pts], color=BLUE, marker="o", lw=2, capsize=3)
        for p in pts:
            ax.annotate(f"σ={p['sigma']}", (p["epsilon"], p["metrics"][key]["mean"]),
                        textcoords="offset points", xytext=(4, 6), fontsize=8, color=MUTED)
        nodp = [p for p in d if p["epsilon"] is None]
        if nodp:
            ax.axhline(nodp[0]["metrics"][key]["mean"], color=ORANGE, ls="--", lw=1.5, label="no DP")
        ax.set_xlabel("ε (max over hospitals, δ=1e-5, 20 rounds)")
        ax.set_ylabel(f"Test {key}")
    axes[0].legend(frameon=False)
    fig.suptitle("Privacy-utility trade-off (smaller ε = stronger privacy)")
    _save(fig, "privacy_curve.png")


def attack():
    d = _load("attack.json")
    rows = d["models"] + [d["positive_control"], d["untrained_control"]]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(rows) + 1.4))
    for i, r in enumerate(rows):
        col = ORANGE if r.get("control") == "positive" else BLUE
        ax.errorbar(r["auc"]["mean"], i, xerr=r["auc"]["std"], fmt="s" if r.get("control") else "o",
                    color=col, ms=6, capsize=3, lw=2)
    ax.axvline(0.5, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Membership-inference AUC (0.5 = chance)")
    ax.set_title("Attack succeeds on the overfit positive control, not on real models")
    _save(fig, "attack.png")


def communication():
    d = _load("communication.json")
    labels = [f"FL, E={e['local_epochs']}" for e in d["by_local_epochs"]] + ["Raw data (CSV, once)"]
    vals = [e["total_mb"] for e in d["by_local_epochs"]] + [d["raw_data"]["csv_mb"]]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.barh(range(len(vals)), vals, color=[BLUE] * (len(vals) - 1) + [ORANGE], height=0.55)
    ax.set_yticks(range(len(vals)), labels)
    ax.invert_yaxis()
    for b, v in zip(bars, vals):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:.2f} MB", va="center", color=INK)
    ax.set_xlabel("Total data transferred (MB, both directions, 3 hospitals, 20 rounds)")
    ax.set_title("Here FL moves MORE bytes than uploading the raw data once")
    ax.set_xlim(0, max(vals) * 1.2)
    _save(fig, "communication.png")


def clients():
    d = _load("client_summary.json")["scenarios"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    w = 0.38
    for off, (scen, c) in zip((-w / 2, w / 2), (("iid", BLUE), ("noniid", ORANGE))):
        rows = d[scen]
        xs = [i + off for i in range(len(rows))]
        axes[0].bar(xs, [r["rows"] for r in rows], w, color=c, label=scen)
        axes[1].bar(xs, [r["prevalence"] for r in rows], w, color=c, label=scen)
    for ax, t in zip(axes, ("Training rows per hospital", "Cancer prevalence per hospital")):
        ax.set_xticks(range(3), ["Hospital 1", "Hospital 2", "Hospital 3"])
        ax.set_title(t)
    axes[0].legend(frameon=False)
    _save(fig, "clients.png")


def make_all():
    for f in (clients, comparison, training_curves, privacy_curve, attack, communication):
        f()
