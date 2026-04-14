"""
Runtime model/artifact loading and access.

This module owns the in-memory runtime state for:
- the CatBoost regression model
- JSON artifacts loaded from models/api_artifacts/

Typical usage:
    from app.model import load_runtime_state, get_model, get_artifacts

    # at app startup
    load_runtime_state()

    # later inside request handlers / predict.py
    model = get_model()
    artifacts = get_artifacts()
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from catboost import CatBoostRegressor

from app.config import Settings, get_settings, load_artifacts


@dataclass(frozen=True, slots=True)
class RuntimeState:
    """In-memory objects shared across requests."""

    model: CatBoostRegressor
    artifacts: dict[str, dict[str, Any]]

    @property
    def model_metadata(self) -> dict[str, Any]:
        return self.artifacts["model_metadata"]

    @property
    def categorical_vocabs(self) -> dict[str, Any]:
        return self.artifacts["categorical_vocabs"]

    @property
    def models_by_manufacturer(self) -> dict[str, Any]:
        return self.artifacts["models_by_manufacturer"]

    @property
    def numeric_medians(self) -> dict[str, Any]:
        return self.artifacts["numeric_medians"]

    @property
    def feature_pipeline_version(self) -> str:
        return self.model_metadata["feature_pipeline_version"]


_runtime_state: RuntimeState | None = None
_runtime_lock = Lock()


def _build_runtime_state(settings: Settings) -> RuntimeState:
    """Create a fresh RuntimeState from disk."""
    settings.validate_required_files()

    model = CatBoostRegressor()
    model.load_model(str(settings.catboost_model_path))

    artifacts = load_artifacts(settings=settings)

    return RuntimeState(
        model=model,
        artifacts=artifacts,
    )


def load_runtime_state(
    settings: Settings | None = None,
    *,
    force_reload: bool = False,
) -> RuntimeState:
    """Load model + artifacts into memory once and return the cached state.

    Parameters
    ----------
    settings
        Optional Settings instance. Uses get_settings() if omitted.
    force_reload
        If True, rebuilds the runtime state even if already loaded.
    """
    global _runtime_state

    if settings is None:
        settings = get_settings()

    if _runtime_state is not None and not force_reload:
        return _runtime_state

    with _runtime_lock:
        # Double-check after acquiring lock 
        if _runtime_state is not None and not force_reload:
            return _runtime_state

        _runtime_state = _build_runtime_state(settings)
        return _runtime_state


def reset_runtime_state() -> None:
    """Clear cached runtime state. Mainly useful in tests."""
    global _runtime_state
    with _runtime_lock:
        _runtime_state = None


def is_runtime_ready() -> bool:
    """Return True when model + artifacts have already been loaded."""
    return _runtime_state is not None


def get_runtime_state() -> RuntimeState:
    """Return the already-loaded runtime state.

    Raises RuntimeError if startup loading has not happened yet.
    """
    if _runtime_state is None:
        raise RuntimeError(
            "Runtime state not loaded. "
            "Call load_runtime_state() during app startup before serving requests."
        )
    return _runtime_state


# ===============================
# Convenience accessors
# ===============================

def get_model() -> CatBoostRegressor:
    """Return the loaded CatBoost model."""
    return get_runtime_state().model


def get_artifacts() -> dict[str, dict[str, Any]]:
    """Return all loaded artifacts."""
    return get_runtime_state().artifacts


def get_model_metadata() -> dict[str, Any]:
    return get_runtime_state().model_metadata


def get_categorical_vocabs() -> dict[str, Any]:
    return get_runtime_state().categorical_vocabs


def get_models_by_manufacturer() -> dict[str, Any]:
    return get_runtime_state().models_by_manufacturer


def get_numeric_medians() -> dict[str, Any]:
    return get_runtime_state().numeric_medians


def get_feature_pipeline_version() -> str:
    return get_runtime_state().feature_pipeline_version


# ===============================
# Readiness probe helper
# ===============================

def get_readiness_status() -> dict[str, Any]:
    """Return a payload shaped for the /ready endpoint."""
    if _runtime_state is None:
        return {
            "status": "not_ready",
            "model_loaded": False,
            "artifacts_loaded": False,
            "feature_pipeline_version": "unknown",
        }

    return {
        "status": "ready",
        "model_loaded": True,
        "artifacts_loaded": True,
        "feature_pipeline_version": _runtime_state.feature_pipeline_version,
    }