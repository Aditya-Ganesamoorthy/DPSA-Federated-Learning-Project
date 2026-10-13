"""Authentication, hospital registration and the live federated-learning endpoints."""
from __future__ import annotations

import base64
import binascii

import numpy as np
from fastapi import APIRouter, Depends, Header, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from backend.services.accounts import AccountError, AccountStore
from backend.services.auth import Auth, AuthError, User
from backend.services.federation import Federation, FederationError
from backend.services.security import SecurityLog
from ml import model_io

router = APIRouter(prefix="/api/v1")


def get_auth(request: Request) -> Auth:
    return request.app.state.auth


def get_fed(request: Request) -> Federation:
    return request.app.state.federation


def get_accounts(request: Request) -> AccountStore:
    return request.app.state.accounts


def get_log(request: Request) -> SecurityLog:
    return request.app.state.security


def current_user(request: Request, authorization: str | None = Header(default=None)) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthError("Authentication required: send 'Authorization: Bearer <token>'")
    return request.app.state.auth.verify_token(authorization[7:].strip())


def hospital_user(user: User = Depends(current_user)) -> User:
    if user.role != "hospital":
        raise AuthError("This action is for hospital accounts", 403)
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise AuthError("Administrator only", 403)
    return user


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hospital_name: str = Field(..., min_length=3, max_length=80)
    parent_network: str = Field("", max_length=80, description="optional parent hospital / network (sub-hospitals)")
    username: str = Field(..., min_length=4, max_length=32)
    password: str = Field(..., min_length=8, max_length=128)
    invite_code: str = Field("", max_length=64)


class PrivacyInfo(BaseModel):
    """Differential-privacy settings the hospital says it used (declared; the server recomputes epsilon from them)."""
    model_config = ConfigDict(extra="forbid")
    dp_enabled: bool = False
    sigma: float | None = Field(None, ge=0.3, le=100)
    sample_rate: float | None = Field(None, gt=0, le=1)
    steps: int | None = Field(None, ge=1, le=1_000_000)


class UpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_version: int = Field(..., ge=0)
    num_examples: int = Field(..., ge=1)
    npz_b64: str = Field(..., min_length=10, max_length=200_000, description="base64 of an .npz with arrays p0..p5")
    privacy: PrivacyInfo | None = None


class UpdateJsonRequest(BaseModel):
    """Browser-friendly update: six flat weight arrays in model order (see GET /features -> array_shapes)."""
    model_config = ConfigDict(extra="forbid")
    base_version: int = Field(..., ge=0)
    num_examples: int = Field(..., ge=1)
    weights: list[list[float]] = Field(..., min_length=6, max_length=6)
    privacy: PrivacyInfo | None = None


def _priv(p: PrivacyInfo | None) -> dict | None:
    return p.model_dump() if p else None


# ---- authentication & registration ----
@router.post("/auth/register", status_code=201)
def register(body: RegisterRequest, accounts: AccountStore = Depends(get_accounts), auth: Auth = Depends(get_auth),
             log: SecurityLog = Depends(get_log)):
    try:
        acc = accounts.register(body.hospital_name, body.parent_network, body.username, body.password, body.invite_code)
    except AccountError as e:
        if e.status in (403, 429):
            log.add("registration_blocked", body.username[:32], str(e))
        raise
    log.add("registration", acc["username"], f"hospital '{acc['name']}' joined the network")
    token, ttl = auth.issue_token(User(acc["username"], "hospital"))
    return {"account": acc, "access_token": token, "token_type": "bearer", "expires_in": ttl, "role": "hospital",
            "username": acc["username"]}


@router.post("/auth/login")
def login(body: LoginRequest, auth: Auth = Depends(get_auth)):
    user = auth.authenticate(body.username, body.password)
    token, ttl = auth.issue_token(user)
    return {"access_token": token, "token_type": "bearer", "expires_in": ttl, "role": user.role,
            "username": user.username}


@router.get("/auth/me")
def me(user: User = Depends(current_user)):
    return {"username": user.username, "role": user.role}


# ---- federated learning ----
@router.get("/fl/status")
def fl_status(user: User = Depends(current_user), fed: Federation = Depends(get_fed)):
    return fed.status()


@router.get("/fl/history")
def fl_history(user: User = Depends(current_user), fed: Federation = Depends(get_fed)):
    return {"history": fed.history}


@router.get("/fl/network")
def fl_network(user: User = Depends(current_user), fed: Federation = Depends(get_fed),
               accounts: AccountStore = Depends(get_accounts)):
    activity = fed.hospital_activity()
    rows = []
    for a in accounts.public_list():
        rows.append({**a, **activity.get(a["username"], {"accepted": 0, "rejected": 0, "last_version": None,
                                                       "dp_updates": 0, "epsilon": None, "pending": False})})
    return {"hospitals": rows, "you": user.username}


@router.get("/fl/model")
def fl_model(user: User = Depends(current_user), fed: Federation = Depends(get_fed)):
    data, version = fed.model_bytes()
    return Response(content=data, media_type="application/octet-stream",
                    headers={"X-Model-Version": str(version),
                             "Content-Disposition": f'attachment; filename="global_v{version}.npz"'})


@router.get("/fl/model-json")
def fl_model_json(user: User = Depends(current_user), fed: Federation = Depends(get_fed)):
    """The same model as flat arrays for the in-browser trainer (no npz parsing needed)."""
    arrays, version = fed.model_arrays()
    return {"version": version, "weights": [a.ravel().astype(float).tolist() for a in arrays]}


@router.post("/fl/update")
def fl_update(body: UpdateRequest, request: Request, user: User = Depends(hospital_user),
              fed: Federation = Depends(get_fed)):
    try:
        raw = base64.b64decode(body.npz_b64, validate=True)
        arrays, _ = model_io.from_bytes(raw)
    except (binascii.Error, ValueError, OSError, KeyError) as e:
        fed.reject(user.hospital_id, "invalid", f"unreadable payload: {e}")
        raise FederationError(f"Could not read the model update: {e}")
    return fed.submit(user.hospital_id, body.base_version, body.num_examples, arrays, _priv(body.privacy),
                      payload_bytes=int(request.headers.get("content-length") or 0))


@router.post("/fl/update-json")
def fl_update_json(body: UpdateJsonRequest, request: Request, user: User = Depends(hospital_user),
                   fed: Federation = Depends(get_fed)):
    shapes = model_io.expected_shapes()
    try:
        if any(len(w) != int(np.prod(s)) for w, s in zip(body.weights, shapes)):
            raise ValueError("array sizes do not match the model architecture")
        arrays = [np.asarray(w, dtype=np.float32).reshape(s) for w, s in zip(body.weights, shapes)]
    except ValueError as e:
        fed.reject(user.hospital_id, "invalid", f"bad weights: {e}")
        raise FederationError(f"Invalid model update: {e}")
    return fed.submit(user.hospital_id, body.base_version, body.num_examples, arrays, _priv(body.privacy),
                      payload_bytes=int(request.headers.get("content-length") or 0))


# ---- administration ----
@router.post("/admin/aggregate")
def force_aggregate(admin: User = Depends(admin_user), fed: Federation = Depends(get_fed)):
    return fed.force_aggregate()


@router.post("/admin/hospitals/{username}/suspend")
def suspend(username: str, admin: User = Depends(admin_user), accounts: AccountStore = Depends(get_accounts),
            log: SecurityLog = Depends(get_log)):
    acc = accounts.set_status(username.lower(), "suspended")
    log.add("hospital_suspended", username.lower(), "suspended by the administrator")
    return acc


@router.post("/admin/hospitals/{username}/restore")
def restore(username: str, admin: User = Depends(admin_user), accounts: AccountStore = Depends(get_accounts),
            log: SecurityLog = Depends(get_log)):
    acc = accounts.set_status(username.lower(), "active")
    log.add("hospital_restored", username.lower(), "restored by the administrator")
    return acc
