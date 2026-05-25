"""
FastAPI application entry point.

Creates the app, configures middleware, loads the model and artifacts
at startup, and defines all routes.

Run locally with:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.model import get_readiness_status, load_runtime_state
from app.predict import run_prediction
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
)
from app.metrics import setup_metrics

settings = get_settings()


# ===============================
# App lifespan (startup / shutdown)
# ===============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and artifacts into memory at startup."""
    load_runtime_state()
    yield


# ===============================
# App instance
# ===============================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Used car price prediction API powered by CatBoost.",
    lifespan=lifespan,
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    openapi_url=f"{settings.api_prefix}/openapi.json",
    responses={
        500: {"model": ErrorResponse},
    },
)


# ===============================
# Middleware
# ===============================

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

setup_metrics(app)

# ===============================
# Prediction router
# ===============================

api_router = APIRouter(prefix=settings.api_prefix, tags=["prediction"])


@api_router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict used car price",
    description="Submit vehicle attributes and receive a predicted listing price in USD.",
    responses={
        200: {"description": "Prediction returned successfully"},
        422: {"description": "Validation error in request body"},
        500: {"model": ErrorResponse, "description": "Server or model error"},
    },
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    return run_prediction(request)


app.include_router(api_router)


# ===============================
# Operational endpoints
# ===============================

@app.get(
    "/",
    tags=["ops"],
    summary="API root",
)
async def root() -> dict[str, str]:
    """Convenience endpoint listing available URLs."""
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "docs_url": f"{settings.api_prefix}/docs",
        "health_url": "/health",
        "ready_url": "/ready",
        "predict_url": f"{settings.api_prefix}/predict",
        "metrics_url": "/metrics", 
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["ops"],
    summary="Liveness check",
    description="Returns ok if the process is running. Used by K8s liveness probe.",
)
async def health() -> HealthResponse:
    return HealthResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["ops"],
    summary="Readiness check",
    description="Returns ready if the model and artifacts are loaded. Used by K8s readiness probe.",
)
async def ready() -> ReadinessResponse:
    return ReadinessResponse(**get_readiness_status())