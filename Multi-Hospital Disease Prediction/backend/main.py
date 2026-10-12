"""FastAPI app. Run from the project root:  uvicorn backend.main:app --reload --port 8000"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routers import api
from backend.services.predictor import Predictor
from backend.services.store import ResultsMissing, Store
from ml import config


def _err(status: int, code: str, message: str, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, **extra}})


def create_app(results_dir: Path | None = None, models_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Federated Lung Cancer Prediction API", version="1.0.0")
    origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in origins], allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])
    app.state.store = Store(results_dir or config.RESULTS_DIR)
    app.state.predictor = Predictor(models_dir or config.MODELS_DIR)

    @app.exception_handler(ResultsMissing)
    async def _missing(_: Request, exc: ResultsMissing):
        return _err(503, "results_missing", str(exc))

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
    return app


app = create_app()
