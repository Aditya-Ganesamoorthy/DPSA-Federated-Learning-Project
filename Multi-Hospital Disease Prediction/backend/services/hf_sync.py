"""Optional Hugging Face Hub sync for the global model (needs HF_TOKEN and HF_REPO_ID).

The model repo is both the public download point for hospitals and the persistence layer that lets a
free Render instance (ephemeral disk, spins down when idle) recover the latest global model after a restart.
Only the model weights, a small config and the aggregation history (hospital ids, counts, test metrics) are
published; never patient data.
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

from backend.settings import Settings
from ml import config

log = logging.getLogger("hf_sync")


def configured(s: Settings) -> bool:
    return bool(s.hf_repo_id and s.hf_token)


def publish(s: Settings, npz_bytes: bytes, history_json: str, version: int) -> dict:
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
        api = HfApi(token=s.hf_token)
        cfg = (config.MODELS_DIR / "model_config.json").read_bytes()
        ops = [
            CommitOperationAdd("global_model.npz", io.BytesIO(npz_bytes)),
            CommitOperationAdd(f"versions/global_v{version}.npz", io.BytesIO(npz_bytes)),
            CommitOperationAdd("history.json", io.BytesIO(history_json.encode())),
            CommitOperationAdd("model_config.json", io.BytesIO(cfg)),
        ]
        api.create_commit(repo_id=s.hf_repo_id, repo_type="model", operations=ops,
                          commit_message=f"Global model v{version} (FedAvg round)")
        out = {"status": "ok", "version": version, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    except Exception as e:  # never let a publish failure break aggregation
        log.warning("HF publish failed: %s", type(e).__name__)
        out = {"status": "error", "version": version, "error": type(e).__name__ + ": " + str(e)[:200]}
    return out


def restore(s: Settings, state_dir: Path) -> bool:
    """Download the latest published model + history into state_dir. False if nothing is there yet."""
    try:
        from huggingface_hub import hf_hub_download
        for name, dest in (("global_model.npz", "global_latest.npz"), ("history.json", "history.json")):
            p = hf_hub_download(s.hf_repo_id, name, repo_type="model", token=s.hf_token or None)
            (state_dir / dest).write_bytes(Path(p).read_bytes())
        return True
    except Exception as e:
        log.info("HF restore skipped: %s", type(e).__name__)
        return False
