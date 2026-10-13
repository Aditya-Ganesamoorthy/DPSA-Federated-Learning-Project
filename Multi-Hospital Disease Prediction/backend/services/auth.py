"""Stateless authentication (stdlib only): survives restarts on Render's ephemeral free disk.

* Admin: ADMIN_USERNAME / ADMIN_PASSWORD from the environment.
* Hospital credentials are DERIVED, not stored: password = base32(HMAC-SHA256(SECRET_KEY, "hospital:<id>"))[:20].
  The admin creates a hospital through the API (or `python -m backend.manage`), which shows the generated
  password. Only someone who holds it can log in; nothing secret sits in a database. Rotating SECRET_KEY revokes everyone;
  REVOKED_HOSPITALS revokes a single hospital.
* Access tokens: base64url(JSON payload) + "." + HMAC-SHA256 signature, with expiry.
* Failed logins are throttled per username (in memory).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
from dataclasses import dataclass

from backend.settings import Settings

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,38}[a-z0-9]$")
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


RESERVED = {"admin", "root", "server", "system"}


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:38]
    if not ID_RE.match(s):
        raise AuthError("Hospital name must contain at least 3 letters or digits", 422)
    if s in RESERVED:
        raise AuthError("That hospital name is reserved", 422)
    return s


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class Auth:
    def __init__(self, settings: Settings):
        self.s = settings
        self._fails: dict[str, list[float]] = {}

    # ---- hospital credentials ----
    def hospital_password(self, hospital_id: str) -> str:
        mac = hmac.new(self.s.secret_key.encode(), f"hospital:{hospital_id}".encode(), hashlib.sha256).digest()
        return base64.b32encode(mac).decode().rstrip("=")[:20].lower()

    # ---- login ----
    def _check_lock(self, username: str) -> None:
        now = time.time()
        recent = [t for t in self._fails.get(username, []) if now - t < LOCK_SECONDS]
        self._fails[username] = recent
        if len(recent) >= MAX_FAILS:
            raise AuthError(f"Too many failed attempts. Try again in {LOCK_SECONDS} seconds.", 429)

    def authenticate(self, username: str, password: str) -> User:
        username = (username or "").strip().lower()
        self._check_lock(username)
        user: User | None = None
        if self.s.admin_password and username == self.s.admin_username.lower():
            if hmac.compare_digest(password.encode(), self.s.admin_password.encode()):
                user = User(username, "admin")
        elif (ID_RE.match(username) and username != self.s.admin_username.lower()
              and username not in self.s.revoked_hospitals):
            if hmac.compare_digest(password.encode(), self.hospital_password(username).encode()):
                user = User(username, "hospital")
        if user is None:
            self._fails.setdefault(username, []).append(time.time())
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
            if p["role"] == "hospital" and p["sub"] in self.s.revoked_hospitals:
                raise AuthError("This hospital has been revoked")
            return User(p["sub"], p["role"])
        except AuthError:
            raise
        except Exception:
            raise AuthError("Invalid or tampered token")
