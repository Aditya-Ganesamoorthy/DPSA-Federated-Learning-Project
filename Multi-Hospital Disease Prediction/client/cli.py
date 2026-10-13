"""Hospital command line.

  set FL_SERVER=https://<your-render-app>.onrender.com   FL_USERNAME=<hospital-id>   FL_PASSWORD=<password>
  python -m client.cli status
  python -m client.cli run --data my_hospital.csv [--epochs 2] [--sigma 0] [--holdout 0.2] [--source server|hf] [--dry-run]

`run` = download the current global model, train on YOUR local CSV (it never leaves this machine), and upload only the
new weights plus the number of training examples. Run from the project root so the `ml` package is importable.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from client.api import ApiError, ServerAPI, download_from_hf
from client.core import load_hospital_csv, local_update


def _fmt(m):
    return "n/a" if not m else f"acc {m['accuracy']:.3f} recall {m['recall']:.3f} auc {m['auc']:.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=os.environ.get("FL_SERVER", "http://localhost:8000"))
    ap.add_argument("--username", default=os.environ.get("FL_USERNAME"))
    ap.add_argument("--password", default=os.environ.get("FL_PASSWORD"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("run")
    r.add_argument("--data", required=True, help="your hospital's local CSV")
    r.add_argument("--epochs", type=int, default=2)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--sigma", type=float, default=0.0, help="DP noise multiplier (0 = no local DP)")
    r.add_argument("--holdout", type=float, default=0.2,
                   help="fraction of local rows kept aside to report before/after metrics")
    r.add_argument("--source", choices=["server", "hf"], default="server",
                   help="where to download the global model from")
    r.add_argument("--hf-repo", default=os.environ.get("HF_REPO_ID", "Adhi-gg029/lung-cancer-federated-learning"))
    r.add_argument("--dry-run", action="store_true", help="train locally but do not upload")
    a = ap.parse_args(argv)
    if not (a.username and a.password):
        print("Set FL_USERNAME and FL_PASSWORD (or pass --username/--password).", file=sys.stderr)
        return 2
    api = ServerAPI(a.server)
    try:
        who = api.login(a.username, a.password)
        print(f"Logged in as {who['username']} ({who['role']})")
        if a.cmd == "status":
            s = api.status()
            print(f"global model v{s['version']} | pending {s['pending_hospitals']} | "
                  f"{s['min_clients']} hospitals needed per round")
            return 0
        if a.source == "hf":
            arrays, version = download_from_hf(a.hf_repo)
            print(f"Downloaded global model v{version} from Hugging Face ({a.hf_repo})")
        else:
            arrays, version = api.download_model()
            print(f"Downloaded global model v{version} from the server")
        X, y = load_hospital_csv(a.data)
        print(f"Local data: {len(X)} rows, {y.mean():.1%} positive (stays on this machine)")
        acc_path = Path.home() / ".fl_hospital" / f"{a.username}_accountant.json"
        acc_path.parent.mkdir(exist_ok=True)
        res = local_update(arrays, X, y, epochs=a.epochs, seed=a.seed, sigma=a.sigma, holdout=a.holdout,
                           hospital_id=a.username, accountant_path=acc_path if a.sigma > 0 else None)
        print(f"Trained on {res['num_examples']} rows. Hold-out ({res['n_holdout']} rows) before: "
              f"{_fmt(res['holdout_before'])} | after: {_fmt(res['holdout_after'])}")
        if res["epsilon"] is not None:
            print(f"Local DP: epsilon = {res['epsilon']:.2f} at delta 1e-5 (cumulative across your runs)")
        if a.dry_run:
            print("Dry run: nothing uploaded.")
            return 0
        out = api.submit(version, res["num_examples"], res["arrays"])
        if out["aggregated"]:
            print(f"Update accepted. Round complete: new global model v{out['new_version']}.")
        else:
            print(f"Update accepted. Waiting for {out['needed']} more hospital(s) (pending: {out['pending']}).")
        return 0
    except ApiError as e:
        print(f"Server said: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
