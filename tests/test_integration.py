"""
Integration tests using the real CatBoost model.

These tests require the actual model file and artifacts on disk.
They verify that the full pipeline produces reasonable predictions.

Run separately from the main test suite:
    pytest tests/test_integration.py -v

Run only integration-marked tests:
    pytest -m integration -v

Skip during normal test runs:
    pytest tests/ --ignore=tests/test_integration.py -v
"""

from __future__ import annotations

import pytest

from app.config import get_settings

# Skip the entire module if required files are missing
settings = get_settings()
try:
    settings.validate_required_files()
    FILES_AVAILABLE = True
except FileNotFoundError:
    FILES_AVAILABLE = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not FILES_AVAILABLE,
        reason="Model file or artifacts not found. Skipping integration tests.",
    ),
]

@pytest.fixture(autouse=True)
def _install_mock_runtime():
    """Override conftest's autouse fixture — integration tests use the real model."""
    yield


@pytest.fixture(autouse=True)
def _reset_predict_cache():
    """Override conftest's autouse fixture — let the real catalog persist across tests."""
    yield


@pytest.fixture(scope="module")
def real_runtime():
    """Load the real model and artifacts once for all integration tests."""
    from app.model import load_runtime_state, reset_runtime_state

    state = load_runtime_state(force_reload=True)
    yield state
    reset_runtime_state()


@pytest.fixture(scope="module")
def real_client(real_runtime):
    """FastAPI test client with the real model loaded."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        yield client


TOYOTA_CAMRY = {
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

BMW_X5 = {
    "manufacturer": "bmw",
    "model": "x5 xdrive40i",
    "year": 2021,
    "mileage": 18000,
    "engine": "3.0l i6 turbo",
    "transmission": "8 speed automatic",
    "drivetrain": "awd",
    "fuel_type": "gasoline",
    "exterior_color": "black sapphire metallic",
    "interior_color": "cognac leather",
    "accidents_or_damage": 0,
    "one_owner": 1,
    "personal_use_only": 1,
    "mpg": "21-26",
    "price_drop": 0.0,
    "seller_rating": 4.7,
    "driver_rating": 4.5,
    "driver_reviews_num": 85.0,
}

OLD_HIGH_MILEAGE = {
    "manufacturer": "honda",
    "model": "civic",
    "year": 2005,
    "mileage": 180000,
    "engine": "1.7l i4",
    "transmission": "5 speed manual",
    "drivetrain": "fwd",
    "fuel_type": "gasoline",
    "exterior_color": "blue",
    "interior_color": "gray cloth",
    "accidents_or_damage": 1,
    "one_owner": 0,
    "personal_use_only": 1,
    "mpg": "30-38",
    "price_drop": 200.0,
    "seller_rating": 3.8,
    "driver_rating": 4.0,
    "driver_reviews_num": 45.0,
}


def _predict(client, payload: dict) -> dict:
    """Post a prediction request and assert the endpoint returned 200."""
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200, (
        f"Expected 200 but got {response.status_code}: {response.text}"
    )
    return response.json()


class TestIntegrationPredictions:
    """Verify that predictions are in reasonable price ranges."""

    def test_toyota_camry_price_range(self, real_client):
        data = _predict(real_client, TOYOTA_CAMRY)
        price = data["predicted_price"]
        assert 10_000 < price < 50_000, f"Toyota Camry price {price} outside expected range"

    def test_bmw_x5_more_expensive_than_camry(self, real_client):
        camry_price = _predict(real_client, TOYOTA_CAMRY)["predicted_price"]
        bmw_price = _predict(real_client, BMW_X5)["predicted_price"]
        assert bmw_price > camry_price, "BMW X5 should be more expensive than Toyota Camry"

    def test_old_high_mileage_cheapest(self, real_client):
        camry_price = _predict(real_client, TOYOTA_CAMRY)["predicted_price"]
        old_price = _predict(real_client, OLD_HIGH_MILEAGE)["predicted_price"]
        assert old_price < camry_price, "2005 Civic with 180k miles should be cheaper than 2020 Camry"

    def test_prediction_is_positive(self, real_client):
        for payload in [TOYOTA_CAMRY, BMW_X5, OLD_HIGH_MILEAGE]:
            data = _predict(real_client, payload)
            assert data["predicted_price"] > 0

    def test_no_warnings_for_known_vehicles(self, real_client):
        data = _predict(real_client, TOYOTA_CAMRY)
        assert data["warnings"] == []


class TestIntegrationFuzzyCorrection:
    """Verify fuzzy matching works with the real vocabulary."""

    def test_manufacturer_typo_corrected(self, real_client):
        payload = dict(TOYOTA_CAMRY)
        payload["manufacturer"] = "Toyata"
        data = _predict(real_client, payload)
        assert any("corrected" in w for w in data["warnings"])
        assert data["predicted_price"] > 0

    def test_corrected_price_identical_to_correct_price(self, real_client):
        correct_price = _predict(real_client, TOYOTA_CAMRY)["predicted_price"]

        payload = dict(TOYOTA_CAMRY)
        payload["manufacturer"] = "Toyata"
        corrected_price = _predict(real_client, payload)["predicted_price"]

        assert correct_price == corrected_price


class TestIntegrationOptionalFields:
    """Verify median imputation produces reasonable results."""

    def test_minimal_request_succeeds(self, real_client):
        minimal = {
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
        data = _predict(real_client, minimal)
        assert data["predicted_price"] > 0

    def test_minimal_and_full_prices_are_close(self, real_client):
        full_price = _predict(real_client, TOYOTA_CAMRY)["predicted_price"]

        minimal = {k: v for k, v in TOYOTA_CAMRY.items()
                   if k not in ("mpg", "price_drop", "seller_rating", "driver_rating", "driver_reviews_num")}
        minimal_price = _predict(real_client, minimal)["predicted_price"]

        ratio = minimal_price / full_price
        assert 0.7 < ratio < 1.3, f"Minimal/full price ratio {ratio} is too far off"