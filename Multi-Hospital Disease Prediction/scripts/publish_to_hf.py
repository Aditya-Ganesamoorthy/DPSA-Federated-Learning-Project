"""Publish the primary global model (v0), model card and config to the Hugging Face model repo.

    $env:HF_TOKEN="hf_xxx"   (PowerShell; a WRITE token: https://huggingface.co/settings/tokens)
    (or keep the token in the git-ignored "Hugging face access token.txt" in the project folder)
    python scripts/publish_to_hf.py [--repo Adhi-gg029/lung-cancer-federated-learning]

After this, the deployed server (HF_TOKEN + HF_REPO_ID set) publishes every new global version itself.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # progress bars print as noise in PowerShell 5.1
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _token_from_file() -> str | None:
    """Convenience: read an `hf_...` token from the git-ignored 'Hugging face access token.txt' next to the project."""
    import re
    for f in ROOT.glob("*ccess token*.txt"):
        m = re.search(r"hf_[A-Za-z0-9]{20,}", f.read_text(encoding="utf-8", errors="ignore"))
        if m:
            return m.group(0)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("HF_REPO_ID", "Adhi-gg029/lung-cancer-federated-learning"))
    ap.add_argument("--dry-run", action="store_true", help="validate files and show what would be uploaded")
    ap.add_argument("--readme-only", action="store_true", help="refresh only the model card; leave the model, history and accounts untouched")
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
    if a.readme_only:
        files = {"README.md": files["README.md"]}
    print(f"Model v{version}: {sum(a.size for a in arrays)} parameters. Files to publish to {a.repo}:")
    for k, v in files.items():
        print(f"  {k:28s} {len(v):>8d} bytes")
    if a.dry_run:
        return 0
    token = os.environ.get("HF_TOKEN") or _token_from_file()
    if not token:
        print("\nHF_TOKEN is not set. Create a WRITE token at https://huggingface.co/settings/tokens, then:\n"
              '  $env:HF_TOKEN="hf_xxx"; python scripts/publish_to_hf.py   (PowerShell)', file=sys.stderr)
        return 2
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        print("huggingface_hub is not installed. Run:  pip install huggingface_hub", file=sys.stderr)
        return 3
    api = HfApi(token=token)
    try:
        who = api.whoami()["name"]
        print(f"Authenticated as: {who}")
        api.create_repo(a.repo, repo_type="model", exist_ok=True)
        info = api.create_commit(repo_id=a.repo, repo_type="model", commit_message="Update model card" if a.readme_only else "Publish primary global model v0",
                                 operations=[CommitOperationAdd(k, v) for k, v in files.items()])
    except HfHubHTTPError as e:
        code = getattr(e.response, "status_code", None)
        hint = {401: "The token is invalid or expired. Create a new one at https://huggingface.co/settings/tokens.",
                403: "The token has no WRITE permission on this repo. For a fine-grained token, tick 'Write access to "
                     "contents/settings of all repos under your personal namespace' (or select this repo with write access).",
                404: "The repo was not found. Check the name (owner/name) and that you own it."}.get(code, "")
        print(f"Hugging Face refused the upload (HTTP {code}). {hint}\n{str(e)[:300]}", file=sys.stderr)
        return 1
    print("Published:", info.commit_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
