import shutil

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.settings import Settings
from ml import config

ADMIN_PW = "admin-password-for-tests"
PW = "correct-horse-battery"


def make_settings(tmp_path, **kw) -> Settings:
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    for f in config.MODELS_DIR.glob("*.npz"):
        shutil.copy(f, models / f.name)
    shutil.copy(config.MODELS_DIR / "model_config.json", models / "model_config.json")
    base = dict(state_dir=tmp_path / "state", models_dir=models, secret_key="unit-test-secret-key-0123456789",
                admin_password=ADMIN_PW, min_clients=3, hf_repo_id="", hf_token="", frontend_dist=None,
                pbkdf2_iterations=1000, invite_code="", registrations_per_hour=1000)
    base.update(kw)
    return Settings(**base)


@pytest.fixture
def app(tmp_path):
    return create_app(make_settings(tmp_path))


@pytest.fixture
def client(app):
    return TestClient(app)


def login(client, username, password):
    r = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def register(client, username, password=PW, name=None, parent="", invite=""):
    r = client.post("/api/v1/auth/register", json={
        "hospital_name": name or f"Hospital {username}", "parent_network": parent, "username": username,
        "password": password, "invite_code": invite})
    return r


def hospital_headers(client, app, hid):
    """Register the hospital if new, then return auth headers."""
    r = register(client, hid)
    if r.status_code == 201:
        return {"Authorization": f"Bearer {r.json()['access_token']}"}
    return login(client, hid, PW)


@pytest.fixture
def admin_headers(client):
    return login(client, "admin", ADMIN_PW)
