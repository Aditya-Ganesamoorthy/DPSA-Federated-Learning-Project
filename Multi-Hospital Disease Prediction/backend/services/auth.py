"""Authentication (stdlib only).

* Administrator: ADMIN_USERNAME / ADMIN_PASSWORD from the environment (can suspend hospitals and force aggregation).
* Hospitals: self-registered accounts (see accounts.py). They log in with their own username and password.
* Access tokens: base64url(JSON payload) + "." + HMAC-SHA256 signature, with expiry. Suspending a hospital takes effect
  immediately because every request re-checks the account status.
* Failed logins are throttled per username (5 failures lock it for 60 s) and recorded as security events.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

from backend.services.accounts import AccountStore
from backend.services.security import SecurityLog
from backend.settings import Settings

MAX_FAILS, LOCK_SECONDS = 5, 60


class AuthError(Exception):
    def __init__(self, message: str, status: int = 401):
        super().__init__(message)
        self.status = status


@dataclass
class User:
    username: str
    role: str                    # "admin" | "hospital"

    @property
    def hospital_id(self) -> str | None:
        return self.username if self.role == "hospital" else None


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class Auth:
    def __init__(self, settings: Settings, accounts: AccountStore, log: SecurityLog):
        self.s, self.accounts, self.log = settings, accounts, log
        self._fails: dict[str, list[float]] = {}

    def _recent_fails(self, username: str) -> list[float]:
        now = time.time()
        recent = [t for t in self._fails.get(username, []) if now - t < LOCK_SECONDS]
        self._fails[username] = recent
        return recent

    def authenticate(self, username: str, password: str) -> User:
        username = (username or "").strip().lower()
        if len(self._recent_fails(username)) >= MAX_FAILS:
            self.log.add("login_blocked", username, "attempt while the account is temporarily locked")
            raise AuthError(f"Too many failed attempts. Try again in {LOCK_SECONDS} seconds.", 429)
        user: User | None = None
        suspended = False
        if username == self.s.admin_username.lower():
            if self.s.admin_password and hmac.compare_digest(password.encode(), self.s.admin_password.encode()):
                user = User(username, "admin")
        else:
            result = self.accounts.check(username, password)
            if result == "ok":
                user = User(username, "hospital")
            suspended = result == "suspended"
        if suspended:                       # correct password, but the administrator suspended this hospital
            raise AuthError("This hospital account is suspended. Contact the network administrator.", 403)
        if user is None:
            fails = self._fails.setdefault(username, [])
            fails.append(time.time())
            if len(fails) == MAX_FAILS:
                self.log.add("login_locked", username, f"{MAX_FAILS} failed logins in {LOCK_SECONDS}s: account locked")
            else:
                self.log.add("login_failed", username, "invalid username or password")
            raise AuthError("Invalid username or password")
        self._fails.pop(username, None)
        return user

    # ---- tokens ----
    def issue_token(self, user: User) -> tuple[str, int]:
        exp = int(time.time()) + self.s.token_ttl_s
        body = _b64(json.dumps({"sub": user.username, "role": user.role, "exp": exp}).encode())
        sig = _b64(hmac.new(self.s.secret_key.encode(), body.encode(), hashlib.sha256).digest())
        return f"{body}.{sig}", self.s.token_ttl_s

    def verify_token(self, token: str) -> User:
        try:
            body, sig = token.split(".")
            good = _b64(hmac.new(self.s.secret_key.encode(), body.encode(), hashlib.sha256).digest())
            if not hmac.compare_digest(sig, good):
                raise ValueError("signature")
            p = json.loads(_unb64(body))
            if p["exp"] < time.time():
                raise AuthError("Session expired, please log in again")
            if p["role"] == "hospital" and not self.accounts.is_active(p["sub"]):
                raise AuthError("This hospital account is not active", 403)
            return User(p["sub"], p["role"])
        except AuthError:
            raise
        except Exception:
            raise AuthError("Invalid or tampered token")
