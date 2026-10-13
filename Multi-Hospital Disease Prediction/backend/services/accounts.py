"""Self-service hospital accounts.

Hospitals register themselves (no administrator has to create them) with a hospital name, optional parent
network, a username and their own password.

Storage: `accounts.json` in the state folder, backed up to / restored from the Hugging Face model repo so accounts
survive a restart of the free Render instance. The backup is PUBLIC, so passwords are never stored: each account keeps
a random salt and a verifier = HMAC-SHA256(SECRET_KEY, PBKDF2-HMAC-SHA256(password, salt)). Without the server's
SECRET_KEY ("pepper") the published verifiers cannot even be attacked offline. Rotating SECRET_KEY invalidates all
passwords (hospitals must register again).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import threading
import time
from pathlib import Path

from backend.services import hf_sync
from backend.settings import Settings

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,30}[a-z0-9]$")
RESERVED = {"admin", "root", "server", "system", "administrator", "null", "undefined"}
CTRL = re.compile(r"[\x00-\x1f\x7f<>]")


class AccountError(Exception):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def _clean(text: str, lo: int, hi: int, label: str) -> str:
    t = CTRL.sub("", (text or "")).strip()
    if not (lo <= len(t) <= hi):
        raise AccountError(f"{label} must be {lo}-{hi} characters")
    return t


class AccountStore:
    def __init__(self, settings: Settings):
        self.s = settings
        self.lock = threading.RLock()
        self.path = Path(settings.state_dir) / "accounts.json"
        self.accounts: dict[str, dict] = {}
        self._reg_times: list[float] = []
        self.last_backup: dict = {"status": "never" if hf_sync.configured(settings) else "not configured"}
        if not self.path.exists() and hf_sync.configured(settings):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            hf_sync.restore_file(settings, "accounts.json", self.path)
        if self.path.exists():
            try:
                self.accounts = {a["username"]: a for a in json.loads(self.path.read_text(encoding="utf-8"))["accounts"]}
            except (ValueError, KeyError, OSError):
                self.accounts = {}

    # ---- hashing ----
    def _verifier(self, password: str, salt_hex: str) -> str:
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), self.s.pbkdf2_iterations)
        return hmac.new(self.s.secret_key.encode(), dk, hashlib.sha256).hexdigest()

    # ---- persistence ----
    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps({"accounts": list(self.accounts.values())}, indent=1)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(blob, encoding="utf-8")
        tmp.replace(self.path)
        if hf_sync.configured(self.s):
            threading.Thread(target=self._backup, args=(blob.encode(),), daemon=True).start()

    def _backup(self, blob: bytes) -> None:
        self.last_backup = hf_sync.publish_file(self.s, "accounts.json", blob, "Update hospital accounts (peppered verifiers only)")

    # ---- operations ----
    def register(self, hospital_name: str, parent: str, username: str, password: str, invite_code: str = "") -> dict:
        with self.lock:
            if self.s.invite_code and not hmac.compare_digest(invite_code.encode(), self.s.invite_code.encode()):
                raise AccountError("A valid invite code is required to register", 403)
            now = time.time()
            self._reg_times = [t for t in self._reg_times if now - t < 3600]
            if len(self._reg_times) >= self.s.registrations_per_hour:
                raise AccountError("Too many registrations right now. Please try again later.", 429)
            if len(self.accounts) >= self.s.max_accounts:
                raise AccountError("The network is full", 429)
            name = _clean(hospital_name, 3, 80, "Hospital name")
            parent_name = _clean(parent, 0, 80, "Parent network") if parent and parent.strip() else ""
            uid = (username or "").strip().lower()
            if not ID_RE.match(uid):
                raise AccountError("Username must be 4-32 characters: lowercase letters, digits and hyphens")
            if uid in RESERVED or uid == self.s.admin_username.lower():
                raise AccountError("That username is reserved")
            if uid in self.accounts:
                raise AccountError("That username is already taken", 409)
            if not (8 <= len(password) <= 128) or password.lower() == uid:
                raise AccountError("Password must be 8-128 characters and different from the username")
            salt = secrets.token_hex(16)
            self.accounts[uid] = {"username": uid, "name": name, "parent": parent_name,
                                  "joined": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "active",
                                  "salt": salt, "verifier": self._verifier(password, salt)}
            self._reg_times.append(now)
            self._persist()
            return self.public(uid)

    def check(self, username: str, password: str) -> str:
        """'ok' | 'bad' | 'suspended'. Runs a hash even for unknown users to keep timing uniform."""
        acc = self.accounts.get(username)
        salt = acc["salt"] if acc else "00" * 16
        good = hmac.compare_digest(self._verifier(password, salt), acc["verifier"] if acc else "x" * 64)
        if not acc or not good:
            return "bad"
        return "ok" if acc["status"] == "active" else "suspended"

    def is_active(self, username: str) -> bool:
        a = self.accounts.get(username)
        return bool(a and a["status"] == "active")

    def set_status(self, username: str, status: str) -> dict:
        with self.lock:
            if username not in self.accounts:
                raise AccountError("Unknown hospital", 404)
            self.accounts[username]["status"] = status
            self._persist()
            return self.public(username)

    def public(self, username: str) -> dict:
        a = self.accounts[username]
        return {k: a[k] for k in ("username", "name", "parent", "joined", "status")}

    def public_list(self) -> list[dict]:
        with self.lock:
            return [self.public(u) for u in sorted(self.accounts)]

    def count(self) -> int:
        return len(self.accounts)
