"""
Tests for the prediction endpoint and orchestration logic.

Covers: route-level tests, validation errors, fuzzy matching,
optional field imputation, unknown categories, error handling,
and an end-to-end mock test.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

import app.main as main_module
import app.predict as predict_module
from app.predict import _fuzzy_match, _validate_and_correct, _impute_optional_fields
from app.schemas import PredictionRequest, PredictionResponse


# ===============================
# Route-level tests
# ===============================

class TestPredictRoute:
    def test_success_with_monkeypatched_run_prediction(self, client, monkeypatch, sample_payload):
        """Unit test the route in isolation by replacing run_prediction."""
        def fake_run_prediction(request):
            assert request.manufacturer == "toyota"
            return PredictionResponse(
                predicted_price=27450.0,
                warnings=[],
                input_echo={
                    "manufacturer": "toyota",
                    "mpg_avg": 30.0,
                    "age": 3,
                },
            )

        monkeypatch.setattr(main_module, "run_prediction", fake_run_prediction)
        response = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_price"] == 27450.0
        assert body["currency"] == "USD"
        assert body["model_used"] == "CatBoost"
        assert body["warnings"] == []
        assert body["input_echo"]["manufacturer"] == "toyota"

    def test_returns_500_when_prediction_fails(self, client, monkeypatch, sample_payload):
        def fake_run_prediction(_request):
            raise HTTPException(status_code=500, detail="Model prediction failed: boom")

        monkeypatch.setattr(main_module, "run_prediction", fake_run_prediction)
        response = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        )
        assert response.status_code == 500
        assert response.json() == {"detail": "Model prediction failed: boom"}

    def test_predict_returns_positive_price(self, client, sample_payload):
        response = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        )
        assert response.status_code == 200
        assert response.json()["predicted_price"] > 0

    def test_minimal_request_returns_200(self, client, minimal_payload):
        response = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=minimal_payload,
        )
        assert response.status_code == 200
        assert response.json()["predicted_price"] > 0

    def test_numeric_mpg_is_accepted(self, client, sample_payload):
        sample_payload["mpg"] = 30

        response = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        )

        assert response.status_code == 200
        body = response.json()
        assert body["predicted_price"] > 0
        assert body["input_echo"]["mpg_avg"] == 30.0


# ===============================
# Validation errors (422)
# ===============================

class TestPredictValidation:
    def test_missing_required_field(self, client, sample_payload):
        sample_payload.pop("manufacturer")
        response = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        )
        assert response.status_code == 422

    def test_empty_string_manufacturer(self, client, sample_payload):
        sample_payload["manufacturer"] = ""
        response = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        )
        assert response.status_code == 422

    def test_whitespace_only_manufacturer(self, client, sample_payload):
        sample_payload["manufacturer"] = "   "
        response = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        )
        assert response.status_code == 422

    def test_year_too_low(self, client, sample_payload):
        sample_payload["year"] = 1950
        response = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        )
        assert response.status_code == 422

    def test_year_too_high(self, client, sample_payload):
        sample_payload["year"] = 2100
        response = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        )
        assert response.status_code == 422

    def test_negative_mileage(self, client, sample_payload):
        sample_payload["mileage"] = -1000
        response = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        )
        assert response.status_code == 422

    def test_accidents_or_damage_out_of_range(self, client, sample_payload):
        sample_payload["accidents_or_damage"] = 2
        response = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        )
        assert response.status_code == 422

    def test_one_owner_out_of_range(self, client, sample_payload):
        sample_payload["one_owner"] = -1
        response = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        )
        assert response.status_code == 422

    def test_negative_price_drop(self, client, sample_payload):
        sample_payload["price_drop"] = -100
        response = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        )
        assert response.status_code == 422

    def test_extra_field_rejected(self, client, sample_payload):
        sample_payload["unknown_field"] = "hello"
        response = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        )
        assert response.status_code == 422

    def test_typo_field_name_rejected(self, client, sample_payload):
        sample_payload["manufacurer"] = "toyota"
        response = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        )
        assert response.status_code == 422

    def test_empty_body(self, client):
        response = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json={},
        )
        assert response.status_code == 422


# ===============================
# Fuzzy matching — unit tests
# ===============================

class TestFuzzyMatch:
    def test_finds_close_match(self):
        assert _fuzzy_match("toyata", ["toyota", "tesla", "honda"]) == "toyota"

    def test_returns_none_for_no_match(self):
        assert _fuzzy_match("xyzabc", ["toyota", "tesla", "honda"]) is None

    def test_returns_none_for_empty_value(self):
        assert _fuzzy_match("", ["toyota", "tesla", "honda"]) is None

    def test_returns_none_for_empty_vocabulary(self):
        assert _fuzzy_match("toyota", []) is None

    def test_works_with_set(self):
        assert _fuzzy_match("toyata", {"toyota", "tesla", "honda"}) == "toyota"


# ===============================
# Fuzzy matching — via endpoint
# ===============================

class TestFuzzyMatchEndpoint:
    def test_manufacturer_typo_corrected(self, client, sample_payload):
        sample_payload["manufacturer"] = "Toyata"
        body = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        ).json()
        assert any("corrected to" in w and "toyota" in w for w in body["warnings"])

    def test_model_typo_corrected(self, client, sample_payload):
        sample_payload["model"] = "camri le"
        body = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        ).json()
        assert any("corrected to" in w and "camry le" in w for w in body["warnings"])

    def test_correction_still_returns_prediction(self, client, sample_payload):
        sample_payload["manufacturer"] = "Toyata"
        body = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        ).json()
        assert body["predicted_price"] > 0


# ===============================
# Unknown categories (no fuzzy match)
# ===============================

class TestUnknownCategories:
    def test_completely_unknown_manufacturer(self, client, sample_payload):
        sample_payload["manufacturer"] = "xyzmotor"
        body = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        ).json()
        assert any("No close match found" in w for w in body["warnings"])
        assert body["predicted_price"] > 0

    def test_unknown_model_for_known_manufacturer(self, client, sample_payload):
        sample_payload["model"] = "xyzmodel"
        body = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        ).json()
        assert any("No close match found" in w for w in body["warnings"])
        assert body["predicted_price"] > 0


# ===============================
# Validate and correct — unit tests
# ===============================

class TestValidateAndCorrect:
    def test_valid_input_no_warnings(self, monkeypatch):
        monkeypatch.setattr(
            predict_module, "get_models_by_manufacturer",
            lambda: {"toyota": ["camry le", "corolla le"]},
        )
        monkeypatch.setattr(
            predict_module, "get_categorical_vocabs",
            lambda: {"drivetrain": ["fwd", "awd"], "fuel_type": ["gasoline", "hybrid"]},
        )

        raw = {"manufacturer": "toyota", "model": "camry le", "drivetrain": "fwd", "fuel_type": "gasoline"}
        warnings = _validate_and_correct(raw)
        assert warnings == []

    def test_corrects_manufacturer_in_place(self, monkeypatch):
        monkeypatch.setattr(
            predict_module, "get_models_by_manufacturer",
            lambda: {"toyota": ["camry le", "corolla le"]},
        )
        monkeypatch.setattr(
            predict_module, "get_categorical_vocabs",
            lambda: {"drivetrain": ["fwd", "awd"], "fuel_type": ["gasoline", "hybrid"]},
        )

        raw = {"manufacturer": "Toyta", "model": "camry le", "drivetrain": "fwd", "fuel_type": "gasoline"}
        warnings = _validate_and_correct(raw)
        assert raw["manufacturer"] == "toyota"
        assert any("corrected to 'toyota'" in w for w in warnings)

    def test_corrects_model_in_place(self, monkeypatch):
        monkeypatch.setattr(
            predict_module, "get_models_by_manufacturer",
            lambda: {"toyota": ["camry le", "corolla le"]},
        )
        monkeypatch.setattr(
            predict_module, "get_categorical_vocabs",
            lambda: {"drivetrain": ["fwd", "awd"], "fuel_type": ["gasoline", "hybrid"]},
        )

        raw = {"manufacturer": "toyota", "model": "Camri LE", "drivetrain": "fwd", "fuel_type": "gasoline"}
        warnings = _validate_and_correct(raw)
        assert raw["model"] == "camry le"
        assert any("corrected to 'camry le'" in w for w in warnings)


# ===============================
# Optional field imputation — unit tests
# ===============================

class TestImputeOptionalFields:
    def test_fills_all_none_values(self, monkeypatch):
        monkeypatch.setattr(
            predict_module, "get_numeric_medians",
            lambda: {
                "mpg_avg": 31.5,
                "price_drop": 0.0,
                "seller_rating": 4.7,
                "driver_rating": 4.4,
                "driver_reviews_num": 92.0,
            },
        )

        raw = {
            "mpg": None,
            "price_drop": None,
            "seller_rating": None,
            "driver_rating": None,
            "driver_reviews_num": None,
        }
        result = _impute_optional_fields(raw)
        assert "mpg" not in result
        assert result["mpg_avg"] == 31.5
        assert result["price_drop"] == 0.0
        assert result["seller_rating"] == 4.7
        assert result["driver_rating"] == 4.4
        assert result["driver_reviews_num"] == 92.0

    def test_preserves_provided_values(self, monkeypatch):
        monkeypatch.setattr(
            predict_module, "get_numeric_medians",
            lambda: {
                "mpg_avg": 31.5,
                "price_drop": 0.0,
                "seller_rating": 4.7,
                "driver_rating": 4.4,
                "driver_reviews_num": 92.0,
            },
        )

        raw = {
            "mpg": "30-35",
            "price_drop": 500.0,
            "seller_rating": 3.0,
            "driver_rating": 3.5,
            "driver_reviews_num": 50.0,
        }
        result = _impute_optional_fields(raw)
        assert result["mpg"] == "30-35"
        assert result["price_drop"] == 500.0
        assert result["seller_rating"] == 3.0

    def test_imputation_via_endpoint(self, client, minimal_payload):
        body = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=minimal_payload,
        ).json()
        assert body["input_echo"]["mpg_avg"] is not None
        assert body["input_echo"]["price_drop"] is not None
        assert body["input_echo"]["seller_rating"] is not None


# ===============================
# Error handling
# ===============================

class TestErrorHandling:
    def test_model_predict_failure_returns_500(self, client, sample_payload, mock_catboost_model):
        mock_catboost_model.predict.side_effect = RuntimeError("Model crashed")
        response = client.post(
            f"{main_module.settings.api_prefix}/predict",
            json=sample_payload,
        )
        assert response.status_code == 500
        assert "Model prediction failed" in response.json()["detail"]

    def test_negative_prediction_clamped_to_zero(self, monkeypatch):
        """If expm1 produces a negative, price should be clamped to 0."""
        monkeypatch.setattr(
            predict_module, "get_models_by_manufacturer",
            lambda: {"toyota": ["camry le"]},
        )
        monkeypatch.setattr(
            predict_module, "get_categorical_vocabs",
            lambda: {"drivetrain": ["fwd"], "fuel_type": ["gasoline"]},
        )
        monkeypatch.setattr(
            predict_module, "get_numeric_medians",
            lambda: {
                "mpg_avg": 30.0, "price_drop": 0.0,
                "seller_rating": 4.5, "driver_rating": 4.2,
                "driver_reviews_num": 120.0,
            },
        )

        class NegativeModel:
            def predict(self, df):
                return np.array([-100.0])

        monkeypatch.setattr(predict_module, "get_model", lambda: NegativeModel())

        request = PredictionRequest(
            manufacturer="toyota", model="camry le", year=2020, mileage=35000,
            engine="2.5l i4", transmission="8 speed automatic",
            drivetrain="fwd", fuel_type="gasoline",
            exterior_color="silver", interior_color="black",
            accidents_or_damage=0, one_owner=1, personal_use_only=1,
            mpg="30", price_drop=0.0, seller_rating=4.5,
            driver_rating=4.2, driver_reviews_num=120.0,
        )

        response = predict_module.run_prediction(request)
        assert response.predicted_price >= 0


# ===============================
# End-to-end mock test
# ===============================

class TestEndToEndMock:
    def test_full_flow_with_fuzzy_correction_and_imputation(self, monkeypatch):
        """Tests the complete run_prediction flow with controlled mocks.

        Verifies that fuzzy correction, imputation, feature engineering,
        and model inference all chain together correctly.
        """
        monkeypatch.setattr(
            predict_module, "get_models_by_manufacturer",
            lambda: {"toyota": ["camry le", "corolla le"]},
        )
        monkeypatch.setattr(
            predict_module, "get_categorical_vocabs",
            lambda: {"drivetrain": ["fwd", "awd"], "fuel_type": ["gasoline", "hybrid"]},
        )
        monkeypatch.setattr(
            predict_module, "get_numeric_medians",
            lambda: {
                "mpg_avg": 30.0, "price_drop": 0.0,
                "seller_rating": 4.6, "driver_rating": 4.3,
                "driver_reviews_num": 100.0,
            },
        )
        monkeypatch.setattr(
            predict_module, "get_model_metadata",
            lambda: {"feature_columns": ["manufacturer", "mpg_avg", "age"]},
        )

        def fake_prepare_for_prediction(raw_input, feature_columns):
            # Assert fuzzy correction and imputation happened
            assert raw_input["manufacturer"] == "toyota"
            assert raw_input["model"] == "camry le"
            assert raw_input["mpg_avg"] == 30.0
            assert raw_input["price_drop"] == 0.0

            df = pd.DataFrame([{
                "manufacturer": raw_input["manufacturer"],
                "mpg_avg": raw_input["mpg_avg"],
                "age": 3,
            }])
            return df[feature_columns]

        class DummyModel:
            def predict(self, features_df):
                assert list(features_df.columns) == ["manufacturer", "mpg_avg", "age"]
                return [math.log1p(25000.0)]

        monkeypatch.setattr(predict_module, "prepare_for_prediction", fake_prepare_for_prediction)
        monkeypatch.setattr(predict_module, "get_model", lambda: DummyModel())

        request = PredictionRequest(
            manufacturer="Toyta", model="Camri LE", year=2020, mileage=35000,
            engine="2.5l i4 dohc 16v", transmission="8 speed automatic",
            drivetrain="awd", fuel_type="gasoline",
            exterior_color="silver metallic", interior_color="black leather",
            accidents_or_damage=0, one_owner=1, personal_use_only=1,
            mpg=None, price_drop=None, seller_rating=None,
            driver_rating=None, driver_reviews_num=None,
        )

        response = predict_module.run_prediction(request)

        assert response.predicted_price == 25000.0
        assert response.currency == "USD"
        assert response.model_used == "CatBoost"
        assert response.input_echo == {
            "manufacturer": "toyota",
            "mpg_avg": 30.0,
            "age": 3,
        }
        assert any("corrected to 'toyota'" in w for w in response.warnings)
        assert any("corrected to 'camry le'" in w for w in response.warnings)
