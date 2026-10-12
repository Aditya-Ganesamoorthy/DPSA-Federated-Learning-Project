"""Record-level DP-SGD with Opacus + a PERSISTENT per-client privacy accountant.

Flower simulation re-creates clients every round. A fresh PrivacyEngine each round would reset
the accountant and under-report epsilon. `AccountantStore` keeps every client's full history
(noise multiplier, sample rate, steps) across rounds and epsilon is computed from all of it.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml import config

warnings.filterwarnings("ignore", module="opacus")
warnings.filterwarnings("ignore", message=".*Full backward hook.*")


class AccountantStore:
    """client_id -> list of (noise_multiplier, sample_rate, steps) accumulated over ALL rounds."""

    def __init__(self):
        self.history: dict[str, list[tuple[float, float, int]]] = {}

    def add(self, client_id: str, entries) -> None:
        self.history.setdefault(client_id, []).extend((float(s), float(q), int(n)) for s, q, n in entries)

    def epsilon(self, client_id: str, delta: float = config.DELTA) -> float:
        hist = self.history.get(client_id, [])
        if not hist:
            return 0.0
        acc = RDPAccountant()
        acc.history = list(hist)
        return float(acc.get_epsilon(delta=delta))

    def max_epsilon(self, delta: float = config.DELTA) -> float:
        return max((self.epsilon(c, delta) for c in self.history), default=0.0)

    def per_client(self, delta: float = config.DELTA) -> dict[str, float]:
        return {c: self.epsilon(c, delta) for c in self.history}


def train_local_dp(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int, seed: int, sigma: float,
                   store: AccountantStore, client_id: str, clip: float = config.CLIP_NORM,
                   lr=config.LR, momentum=config.MOMENTUM, batch_size=config.BATCH_SIZE) -> float:
    """DP-SGD local training (per-sample clipping + Gaussian noise). Mutates `model` in place."""
    torch.manual_seed(seed)
    model.train()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    engine = PrivacyEngine(accountant="rdp")
    # make_private mutates `model`'s parameters' hooks and returns a wrapper sharing the same tensors.
    dp_model, dp_opt, dp_loader = engine.make_private(
        module=model, optimizer=opt, data_loader=loader, noise_multiplier=sigma,
        max_grad_norm=clip, poisson_sampling=True, noise_generator=torch.Generator().manual_seed(seed + 1))
    loss_fn = nn.BCEWithLogitsLoss()
    last = 0.0
    for _ in range(epochs):
        tot, cnt = 0.0, 0
        for xb, yb in dp_loader:
            if len(xb) == 0:          # Poisson sampling can yield an empty batch: skip, do not crash
                dp_opt.zero_grad()
                continue
            dp_opt.zero_grad()
            loss = loss_fn(dp_model(xb), yb)
            loss.backward()
            dp_opt.step()
            tot += loss.item() * len(xb)
            cnt += len(xb)
        last = tot / max(cnt, 1)
    store.add(client_id, engine.accountant.history)
    # parameters are shared with the wrapper, so `model` is already updated; remove Opacus hooks
    dp_model._close()
    return last
