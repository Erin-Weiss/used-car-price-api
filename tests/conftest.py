"""
Shared pytest fixtures for the test suite.

The client fixture monkeypatches load_runtime_state to prevent the
real lifespan from loading model/artifacts from disk. Mock runtime
state is installed separately so accessor functions work in tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
import app.predict as predict_module
from app import model as model_module
from app.model import RuntimeState


# ===============================
# Sample artifact data
# ===============================

SAMPLE_MODEL_METADATA = {
    "feature_pipeline_version": "v3",
    "reference_year": 2023,
    "feature_columns": [
        "manufacturer", "model", "mileage", "mpg_avg", "price_drop",
        "seller_rating", "driver_rating", "driver_reviews_num",
        "accidents_or_damage", "one_owner", "personal_use_only",
        "drivetrain", "fuel_type", "age", "mileage_per_year",
        "is_luxury_brand", "luxury_age_interaction",
        "engine_liters", "engine_cylinders", "engine_layout",
        "engine_turbo", "engine_hybrid",
        "transmission_clean", "transmission_gears", "transmission_gears_missing",
        "exterior_color_base", "interior_color_base",
    ],
    "numeric_features": [
        "mileage", "mpg_avg", "price_drop", "seller_rating", "driver_rating",
        "driver_reviews_num", "accidents_or_damage", "one_owner", "personal_use_only",
        "age", "mileage_per_year", "engine_liters", "engine_cylinders",
        "engine_turbo", "engine_hybrid", "transmission_gears", "transmission_gears_missing",
        "is_luxury_brand", "luxury_age_interaction",
    ],
    "categorical_features": [
        "manufacturer", "model", "drivetrain", "fuel_type", "engine_layout",
        "transmission_clean", "exterior_color_base", "interior_color_base",
    ],
    "target_transform": "log1p",
    "inverse_transform": "expm1",
}

SAMPLE_CATEGORICAL_VOCABS = {
    "manufacturer": ["bmw", "honda", "toyota"],
    "model": ["camry le", "civic", "x5 xdrive40i"],
    "drivetrain": ["4wd", "awd", "fwd", "rwd"],
    "fuel_type": ["diesel", "electric", "gasoline", "hybrid"],
    "engine_layout": ["i", "v", "unknown"],
    "transmission_clean": ["automatic", "cvt", "manual", "unknown"],
    "exterior_color_base": ["black", "blue", "gray", "green", "other", "red", "white"],
    "interior_color_base": ["beige", "black", "brown", "gray", "other", "red", "white"],
}

SAMPLE_MODELS_BY_MANUFACTURER = {
    "toyota": ["camry le", "corolla se", "rav4 xle"],
    "honda": ["civic", "accord", "cr v"],
    "bmw": ["x5 xdrive40i", "330i", "x3 sdrive30i"],
}

SAMPLE_NUMERIC_MEDIANS = {
    "mileage": 35000.0,
    "mpg_avg": 27.5,
    "price_drop": 0.0,
    "seller_rating": 4.5,
    "driver_rating": 4.2,
    "driver_reviews_num": 120.0,
    "accidents_or_damage": 0.0,
    "one_owner": 1.0,
    "personal_use_only": 1.0,
    "age": 3.0,
    "mileage_per_year": 11666.67,
    "engine_liters": 2.5,
    "engine_cylinders": 4.0,
    "engine_turbo": 0.0,
    "engine_hybrid": 0.0,
    "transmission_gears": 8.0,
    "transmission_gears_missing": 0.0,
    "is_luxury_brand": 0.0,
    "luxury_age_interaction": 0.0,
}


# ===============================
# Fixtures
# ===============================

@pytest.fixture
def sample_payload() -> dict:
    """A complete, valid prediction request payload."""
    return {
        "manufacturer": "toyota",
        "model": "camry le",
        "year": 2020,
        "mileage": 35000,
        "engine": "2.5l i4 dohc 16v",
        "transmission": "8 speed automatic",
        "drivetrain": "fwd",
        "fuel_type": "gasoline",
        "exterior_color": "silver metallic",
        "interior_color": "black leather",
        "accidents_or_damage": 0,
        "one_owner": 1,
        "personal_use_only": 1,
        "mpg": "28-32",
        "price_drop": 0.0,
        "seller_rating": 4.5,
        "driver_rating": 4.2,
        "driver_reviews_num": 120.0,
    }


@pytest.fixture
def minimal_payload() -> dict:
    """A valid request with only required fields (optional fields omitted)."""
    return {
        "manufacturer": "toyota",
        "model": "camry le",
        "year": 2020,
        "mileage": 35000,
        "engine": "2.5l i4 dohc 16v",
        "transmission": "8 speed automatic",
        "drivetrain": "fwd",
        "fuel_type": "gasoline",
        "exterior_color": "silver metallic",
        "interior_color": "black leather",
        "accidents_or_damage": 0,
        "one_owner": 1,
        "personal_use_only": 1,
    }


@pytest.fixture
def mock_catboost_model():
    """A mock CatBoost model that returns a fixed log1p prediction."""
    mock_model = MagicMock()
    # log1p(27474) ≈ 10.22
    mock_model.predict.return_value = np.array([10.22])
    return mock_model


@pytest.fixture
def sample_artifacts():
    """Sample artifacts dict matching the structure from load_artifacts()."""
    return {
        "model_metadata": SAMPLE_MODEL_METADATA,
        "categorical_vocabs": SAMPLE_CATEGORICAL_VOCABS,
        "models_by_manufacturer": SAMPLE_MODELS_BY_MANUFACTURER,
        "numeric_medians": SAMPLE_NUMERIC_MEDIANS,
    }


@pytest.fixture
def mock_runtime_state(mock_catboost_model, sample_artifacts):
    """A RuntimeState with a mocked model and sample artifacts."""
    return RuntimeState(
        model=mock_catboost_model,
        artifacts=sample_artifacts,
    )


@pytest.fixture(autouse=True)
def _install_mock_runtime(mock_runtime_state):
    """Install mocked runtime state before each test, reset after.

    autouse=True means every test gets this without explicitly
    requesting the fixture. This ensures accessor functions like
    get_model(), get_artifacts() etc. work in all tests.
    """
    model_module._runtime_state = mock_runtime_state
    yield
    model_module._runtime_state = None


@pytest.fixture(autouse=True)
def _reset_predict_cache():
    """Reset the normalized catalog cache before and after each test.

    The catalog is a module-level cache in predict.py. Without this,
    a test that builds the catalog could leak state into later tests,
    making results depend on execution order.
    """
    predict_module._normalized_catalog = None
    yield
    predict_module._normalized_catalog = None


@pytest.fixture
def client(monkeypatch):
    """FastAPI test client.

    Monkeypatches load_runtime_state so the lifespan doesn't try
    to load the real model/artifacts from disk. The mock runtime
    installed by _install_mock_runtime handles accessor calls.
    """
    monkeypatch.setattr(main_module, "load_runtime_state", lambda: None)
    with TestClient(main_module.app) as test_client:
        yield test_client
