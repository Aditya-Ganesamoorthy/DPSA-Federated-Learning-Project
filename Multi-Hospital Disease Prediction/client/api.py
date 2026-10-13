"""Thin HTTP client for the federation server (works against Render, a local server, or a Starlette TestClient)."""
from __future__ import annotations

import base64

import httpx
import numpy as np

from ml import model_io


class ApiError(Exception):
    def __init__(self, status: int, message: str, body: dict | None = None):
        super().__init__(f"[{status}] {message}")
        self.status, self.body = status, body or {}


class ServerAPI:
    def __init__(self, base_url: str, http: httpx.Client | None = None, timeout: float = 120.0):
        # long timeout: a free Render instance needs up to ~1 minute to wake up
        self.http = http or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self.token: str | None = None

    def _call(self, method: str, path: str, **kw):
        headers = kw.pop("headers", {})
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        r = self.http.request(method, f"/api/v1{path}", headers=headers, **kw)
        if r.status_code >= 400:
            try:
                body = r.json()
                msg = body.get("error", {}).get("message", r.text)
            except ValueError:
                body, msg = None, r.text[:200]
            raise ApiError(r.status_code, msg, body)
        return r

    def login(self, username: str, password: str) -> dict:
        data = self._call("POST", "/auth/login", json={"username": username, "password": password}).json()
        self.token = data["access_token"]
        return data

    def status(self) -> dict:
        return self._call("GET", "/fl/status").json()

    def download_model(self) -> tuple[list[np.ndarray], int]:
        r = self._call("GET", "/fl/model")
        arrays, version = model_io.from_bytes(r.content)
        return arrays, int(r.headers.get("X-Model-Version", version))

    def submit(self, base_version: int, num_examples: int, arrays: list[np.ndarray]) -> dict:
        b64 = base64.b64encode(model_io.to_bytes(arrays, base_version)).decode()
        return self._call("POST", "/fl/update", json={"base_version": base_version,
                                                      "num_examples": num_examples, "npz_b64": b64}).json()


def download_from_hf(repo_id: str) -> tuple[list[np.ndarray], int]:
    """Public download of the latest published global model straight from Hugging Face."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id, "global_model.npz", repo_type="model")
    return model_io.load(p)
