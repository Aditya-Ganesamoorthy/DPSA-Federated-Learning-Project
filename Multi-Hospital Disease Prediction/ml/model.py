"""Model definitions, parameter helpers, plain local training and data loading."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ml import config, schema


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    """input -> 32 -> 16 -> 1 (logit). No BatchNorm, so it is Opacus-compatible."""

    def __init__(self, n_in: int = schema.N_FEATURES, hidden=config.HIDDEN):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(nn.Linear(n_in, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(),
                                 nn.Linear(h2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LogReg(nn.Module):
    def __init__(self, n_in: int = schema.N_FEATURES):
        super().__init__()
        self.fc = nn.Linear(n_in, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


def make_model(kind: str = "mlp", seed: int = 0) -> nn.Module:
    set_seed(seed)
    if kind == "mlp":
        return MLP().float()
    if kind == "logreg":
        return LogReg().float()
    raise ValueError(kind)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---- parameter get/set: the ONLY place parameter ordering is defined ----
def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [v.detach().cpu().numpy().astype(np.float32).copy() for v in model.state_dict().values()]


def set_parameters(model: nn.Module, params: list[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    assert len(keys) == len(params), "parameter count mismatch"
    model.load_state_dict({k: torch.tensor(np.asarray(p, dtype=np.float32)) for k, p in zip(keys, params)})


# ---- data ----
def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return df[schema.FEATURES].to_numpy(np.float32), df[schema.TARGET].to_numpy(np.float32)


def client_path(scenario: str, idx: int) -> Path:
    return config.CLIENT_DIR / f"{scenario}_hospital_{idx}.csv"


# ---- training ----
def make_optimizer(model: nn.Module, lr=config.LR, momentum=config.MOMENTUM):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def train_local(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int, seed: int,
                lr=config.LR, momentum=config.MOMENTUM, batch_size=config.BATCH_SIZE,
                prox_mu: float = 0.0, prox_ref: list | None = None) -> float:
    """Plain mini-batch SGD. Returns mean loss of the last epoch.

    FedProx: if prox_mu > 0, adds mu * (w - w_global) to each gradient (the proximal term's gradient)."""
    model.train()
    opt = make_optimizer(model, lr, momentum)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    g = torch.Generator().manual_seed(seed)
    ref = [torch.tensor(a) for a in prox_ref] if (prox_mu > 0 and prox_ref is not None) else None
    n, last = len(Xt), 0.0
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for s in range(0, n, batch_size):
            b = perm[s:s + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward()
            if ref is not None:
                for p_, r_ in zip(model.parameters(), ref):
                    p_.grad.add_(prox_mu * (p_.detach() - r_))
            opt.step()
            tot += loss.item() * len(b)
        last = tot / n
    return last


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(X))).numpy()


def save_model(model: nn.Module, name: str, kind: str = "mlp") -> None:
    config.MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), config.MODELS_DIR / f"{name}.pt")
    (config.MODELS_DIR / "model_config.json").write_text(json.dumps({
        "kind": kind, "n_features": schema.N_FEATURES, "hidden": list(config.HIDDEN),
        "features": schema.FEATURES, "threshold": config.THRESHOLD}, indent=2))
