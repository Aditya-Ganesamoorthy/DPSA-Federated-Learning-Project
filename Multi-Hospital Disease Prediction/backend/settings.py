"""Runtime configuration from environment variables (nothing secret is ever hard-coded)."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from ml import config


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    results_dir: Path = config.RESULTS_DIR
    models_dir: Path = config.MODELS_DIR
    state_dir: Path = field(default_factory=lambda: Path(os.environ.get("STATE_DIR", config.ROOT / "state")))
    frontend_dist: Path | None = field(default_factory=lambda: (
        Path(os.environ["FRONTEND_DIST"]) if os.environ.get("FRONTEND_DIST") else config.ROOT / "frontend" / "dist"))
    # auth
    secret_key: str = field(default_factory=lambda: os.environ.get("SECRET_KEY", ""))
    admin_username: str = field(default_factory=lambda: os.environ.get("ADMIN_USERNAME", "admin"))
    admin_password: str = field(default_factory=lambda: os.environ.get("ADMIN_PASSWORD", ""))
    token_ttl_s: int = field(default_factory=lambda: _int("TOKEN_TTL_SECONDS", 8 * 3600))
    # self-service hospital registration
    invite_code: str = field(default_factory=lambda: os.environ.get("INVITE_CODE", ""))   # empty = open registration
    max_accounts: int = field(default_factory=lambda: _int("MAX_ACCOUNTS", 200))
    registrations_per_hour: int = field(default_factory=lambda: _int("REGISTRATIONS_PER_HOUR", 30))
    pbkdf2_iterations: int = field(default_factory=lambda: _int("PBKDF2_ITERATIONS", 150_000))
    # federation
    min_clients: int = field(default_factory=lambda: _int("MIN_CLIENTS", 2))
    max_update_norm: float = field(default_factory=lambda: float(os.environ.get("MAX_UPDATE_NORM", "20")))
    max_examples: int = 1_000_000
    aggregator: str = field(default_factory=lambda: os.environ.get("AGGREGATOR", "fedavg").lower())  # fedavg | median
    # hugging face (optional)
    hf_repo_id: str = field(default_factory=lambda: os.environ.get("HF_REPO_ID", ""))
    hf_token: str = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")))

    def __post_init__(self):
        if not self.secret_key:
            # Per-process random key: tokens and stored password verifiers stop working on restart.
            self.secret_key = secrets.token_hex(32)
            self.ephemeral_secret = True
        else:
            self.ephemeral_secret = False
