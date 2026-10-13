"""Authentication + live federated-learning endpoints."""
from __future__ import annotations

import base64
import binascii

from fastapi import APIRouter, Depends, Header, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from backend.services.auth import Auth, AuthError, User, slugify
from backend.services.federation import Federation, FederationError
from ml import model_io

router = APIRouter(prefix="/api/v1")


def get_auth(request: Request) -> Auth:
    return request.app.state.auth


def get_fed(request: Request) -> Federation:
    return request.app.state.federation


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


class UpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_version: int = Field(..., ge=0)
    num_examples: int = Field(..., ge=1)
    npz_b64: str = Field(..., min_length=10, max_length=200_000, description="base64 of an .npz with arrays p0..p5")


class NewHospital(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., min_length=3, max_length=80)


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


@router.get("/fl/model")
def fl_model(user: User = Depends(current_user), fed: Federation = Depends(get_fed)):
    data, version = fed.model_bytes()
    return Response(content=data, media_type="application/octet-stream",
                    headers={"X-Model-Version": str(version),
                             "Content-Disposition": f'attachment; filename="global_v{version}.npz"'})


@router.post("/fl/update")
def fl_update(body: UpdateRequest, user: User = Depends(hospital_user), fed: Federation = Depends(get_fed)):
    try:
        raw = base64.b64decode(body.npz_b64, validate=True)
        arrays, _ = model_io.from_bytes(raw)
    except (binascii.Error, ValueError, OSError, KeyError) as e:
        raise FederationError(f"Could not read the model update: {e}")
    return fed.submit(user.hospital_id, body.base_version, body.num_examples, arrays)


# ---- administration ----
@router.post("/admin/hospitals")
def create_hospital(body: NewHospital, admin: User = Depends(admin_user), auth: Auth = Depends(get_auth)):
    """Issue credentials for a hospital. The password is derived from SECRET_KEY, so nothing is stored."""
    hid = slugify(body.name)
    return {"hospital_id": hid, "username": hid, "password": auth.hospital_password(hid),
            "note": "Share these credentials with the hospital over a secure channel."}


@router.post("/admin/aggregate")
def force_aggregate(admin: User = Depends(admin_user), fed: Federation = Depends(get_fed)):
    return fed.force_aggregate()
