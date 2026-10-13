"""Publish the primary global model (v0), model card and config to the Hugging Face model repo.

    set HF_TOKEN=hf_xxx      (a WRITE token for the repo owner: https://huggingface.co/settings/tokens)
    python scripts/publish_to_hf.py [--repo Adhi-gg029/lung-cancer-federated-learning]

After this, the deployed server (HF_TOKEN + HF_REPO_ID set) publishes every new global version itself.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("HF_REPO_ID", "Adhi-gg029/lung-cancer-federated-learning"))
    ap.add_argument("--dry-run", action="store_true", help="validate files and show what would be uploaded")
    a = ap.parse_args()

    from ml import config, model_io
    arrays, version = model_io.load(config.MODELS_DIR / "global_v0.npz")
    model_io.validate_arrays(arrays)
    files = {
        "README.md": (ROOT / "huggingface" / "README.md").read_bytes(),
        "global_model.npz": (config.MODELS_DIR / "global_v0.npz").read_bytes(),
        "versions/global_v0.npz": (config.MODELS_DIR / "global_v0.npz").read_bytes(),
        "model_config.json": (config.MODELS_DIR / "model_config.json").read_bytes(),
    }
    hist_path = ROOT / "state" / "history.json"
    files["history.json"] = (hist_path.read_bytes() if hist_path.exists() else
                             json.dumps([{"version": 0, "note": "initial primary model (FedAvg over 3 simulated hospitals)"}]).encode())
    print(f"Model v{version}: {sum(a.size for a in arrays)} parameters. Files to publish to {a.repo}:")
    for k, v in files.items():
        print(f"  {k:28s} {len(v):>8d} bytes")
    if a.dry_run:
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\nHF_TOKEN is not set. Create a WRITE token at https://huggingface.co/settings/tokens, then:\n"
              "  set HF_TOKEN=hf_xxx   &   python scripts/publish_to_hf.py", file=sys.stderr)
        return 2
    import io

    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=token)
    api.create_repo(a.repo, repo_type="model", exist_ok=True)
    info = api.create_commit(repo_id=a.repo, repo_type="model", commit_message="Publish primary global model v0",
                             operations=[CommitOperationAdd(k, io.BytesIO(v)) for k, v in files.items()])
    print("Published:", info.commit_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
