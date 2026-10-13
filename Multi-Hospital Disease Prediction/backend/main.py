"""FastAPI app. Run from the project root:  uvicorn backend.main:app --port 8000

Serves the JSON API under /api/v1 and, when frontend/dist exists, the built React dashboard at / (single
service, single origin: simplest for a free Render instance).
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.routers import api, fl
from backend.services.auth import Auth, AuthError
from backend.services.federation import Federation, FederationError
from backend.services.predictor import Predictor
from backend.services.store import ResultsMissing, Store
from backend.settings import Settings

log = logging.getLogger("app")


def _err(status: int, code: str, message: str, headers: dict | None = None, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, headers=headers,
                        content={"error": {"code": code, "message": message, **extra}})


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or Settings()
    app = FastAPI(title="Federated Lung Cancer Prediction", version="2.0.0", docs_url="/api/docs",
                  openapi_url="/api/openapi.json")
    app.add_middleware(CORSMiddleware, allow_origins=list(s.cors_origins), allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "Authorization"], expose_headers=["X-Model-Version"])
    app.state.settings = s
    app.state.store = Store(s.results_dir)
    app.state.auth = Auth(s)
    app.state.federation = Federation(s)
    app.state.predictor = Predictor(s.models_dir, app.state.federation)
    if s.ephemeral_secret:
        log.warning("SECRET_KEY is not set: using a random per-process key (logins reset on restart)")
    if not s.admin_password:
        log.warning("ADMIN_PASSWORD is not set: the admin account is disabled")

    @app.exception_handler(ResultsMissing)
    async def _missing(_: Request, exc: ResultsMissing):
        return _err(503, "results_missing", str(exc))

    @app.exception_handler(AuthError)
    async def _auth(_: Request, exc: AuthError):
        hdr = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else None
        return _err(exc.status, {401: "unauthorized", 403: "forbidden", 429: "too_many_attempts"}.get(exc.status, "auth_error"),
                    str(exc), headers=hdr)

    @app.exception_handler(FederationError)
    async def _fed(_: Request, exc: FederationError):
        return _err(exc.status, "stale_update" if exc.status == 409 else "invalid_update", str(exc), **exc.extra)

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException):
        code = {404: "not_found", 503: "unavailable"}.get(exc.status_code, "http_error")
        return _err(exc.status_code, code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        fields = [{"field": ".".join(str(p) for p in e["loc"][1:]) or str(e["loc"][0]), "message": e["msg"]}
                  for e in exc.errors()]
        return _err(422, "validation_error", "Invalid request", fields=fields)

    app.include_router(api.router)
    app.include_router(fl.router)

    dist = Path(s.frontend_dist) if s.frontend_dist else None
    if dist and (dist / "index.html").is_file():
        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            if path.startswith("api/"):
                return _err(404, "not_found", "Unknown API route")
            f = (dist / path).resolve()
            if path and f.is_file() and dist.resolve() in f.parents:
                return FileResponse(f)
            return FileResponse(dist / "index.html")
    return app


app = create_app()
