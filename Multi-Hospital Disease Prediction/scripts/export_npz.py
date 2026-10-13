"""Convert the trained torch state_dicts in models/*.pt to the numpy .npz exchange format
(used by the live federation server and hospital clients). Also writes models/global_v0.npz,
the primary global model (version 0) that is published to Hugging Face.   python scripts/export_npz.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ml import config, model_io  # noqa: E402


def main() -> None:
    for name in ("centralized", "federated", "federated_dp"):
        sd = torch.load(config.MODELS_DIR / f"{name}.pt", map_location="cpu", weights_only=True)
        arrays = [v.numpy().astype(np.float32) for v in sd.values()]
        model_io.validate_arrays(arrays)
        model_io.save(config.MODELS_DIR / f"{name}.npz", arrays, version=0)
        print("wrote", f"{name}.npz")
    # primary global model v0 = FedAvg model trained across the three simulated hospitals (seed 0)
    arrays, _ = model_io.load(config.MODELS_DIR / "federated.npz")
    model_io.save(config.MODELS_DIR / "global_v0.npz", arrays, version=0)
    print("wrote global_v0.npz (primary global model, version 0)")


if __name__ == "__main__":
    main()
