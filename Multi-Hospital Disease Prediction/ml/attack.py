"""Membership inference (loss-threshold attack) with a positive control.

Members: records used in training. Non-members: the dedicated `attack.csv` hold-out, which is never
used for training, validation or testing. Member and non-member counts are balanced, so AUC 0.5 = chance.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from ml import config, model as M


@torch.no_grad()
def per_sample_loss(net: nn.Module, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    net.eval()
    return nn.functional.binary_cross_entropy_with_logits(
        net(torch.from_numpy(X)), torch.from_numpy(y), reduction="none").numpy()


def attack_scores(member_loss: np.ndarray, nonmember_loss: np.ndarray) -> dict:
    """Guess 'member' when loss is low. AUC is threshold-free; accuracy uses the best threshold
    (an optimistic upper bound for the attacker, stated as such)."""
    n = min(len(member_loss), len(nonmember_loss))
    m, nm = member_loss[:n], nonmember_loss[:n]                          # balance
    score = -np.concatenate([m, nm])
    label = np.concatenate([np.ones(n), np.zeros(n)])
    auc = float(roc_auc_score(label, score))
    thr = np.unique(score)
    best = max(((score >= t) == label).mean() for t in thr)
    return {"auc": auc, "best_accuracy": float(best), "n_per_class": int(n)}


def sample_members(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = M.load_xy(config.CLIENT_DIR / "train_pool.csv")
    idx = np.random.default_rng(seed).choice(len(X), size=min(n, len(X)), replace=False)
    return X[idx], y[idx]


def membership_inference(net: nn.Module, seed: int) -> dict:
    Xn, yn = M.load_xy(config.CLIENT_DIR / "attack.csv")
    Xm, ym = sample_members(seed, len(Xn))
    return attack_scores(per_sample_loss(net, Xm, ym), per_sample_loss(net, Xn, yn))


def positive_control(seed: int, n_train: int = 150, epochs: int = 400) -> dict:
    """Deliberately overfit a large MLP on a tiny subset: the attack MUST succeed here, otherwise the
    attack code (not the model) would be the reason for a null result elsewhere."""
    Xn, yn = M.load_xy(config.CLIENT_DIR / "attack.csv")
    Xm, ym = sample_members(seed, n_train)
    net = nn.Sequential(nn.Linear(Xm.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, 1), nn.Flatten(0))
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(Xm), torch.from_numpy(ym)
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(net(Xt), yt).backward()
        opt.step()
    r = attack_scores(per_sample_loss(net, Xm, ym), per_sample_loss(net, Xn[:n_train], yn[:n_train]))
    r["train_accuracy"] = float(((net(Xt) > 0).float() == yt).float().mean())
    return r


def untrained_control(seed: int) -> dict:
    """Randomly initialised model: attack AUC should be near 0.5."""
    return membership_inference(M.make_model("mlp", seed), seed)
