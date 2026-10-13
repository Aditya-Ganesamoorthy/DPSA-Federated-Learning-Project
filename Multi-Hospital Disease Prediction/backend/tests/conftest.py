import shutil

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.settings import Settings
from ml import config

ADMIN_PW = "admin-password-for-tests"


def make_settings(tmp_path, **kw) -> Settings:
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    for f in config.MODELS_DIR.glob("*.npz"):
        shutil.copy(f, models / f.name)
    shutil.copy(config.MODELS_DIR / "model_config.json", models / "model_config.json")
    base = dict(state_dir=tmp_path / "state", models_dir=models, secret_key="unit-test-secret-key-0123456789",
                admin_password=ADMIN_PW, min_clients=3, hf_repo_id="", hf_token="", frontend_dist=None)
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


@pytest.fixture
def admin_headers(client):
    return login(client, "admin", ADMIN_PW)


def hospital_headers(client, app, hid):
    return login(client, hid, app.state.auth.hospital_password(hid))
