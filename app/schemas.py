"""
Pydantic request/response schemas for the prediction API.

Design choice: The request model reflects raw user input, not engineered
features. Fields like age, mileage_per_year, and engine_liters are
created later inside pipeline.py / predict.py.

Vehicle identity and history fields are required. Listing metadata
fields (mpg, price_drop, seller_rating, driver_rating, driver_reviews_num)
are optional — when omitted, predict.py fills them with training-set
medians from numeric_medians.json.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ===============================
# Shared base model
# ===============================

class APIModel(BaseModel):
    """Common model config for all API schemas.

    extra='forbid' rejects unknown fields (catches typos like 'manufacurer').
    str_strip_whitespace strips leading/trailing whitespace on all strings.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


# ===============================
# Prediction request
# ===============================

class PredictionRequest(APIModel):
    """Raw vehicle attributes submitted by the API caller.

    These correspond to the original dataset columns before any
    feature engineering. The pipeline handles normalization,
    parsing, and derived feature creation internally.
    """

    # --- Vehicle identity (required) ---
    manufacturer: str = Field(
        ...,
        min_length=1,
        description="Vehicle manufacturer, e.g. 'toyota', 'bmw'",
        examples=["toyota"],
    )
    model: str = Field(
        ...,
        min_length=1,
        description="Vehicle model name, e.g. 'camry le', 'x5 xdrive40i'",
        examples=["camry le"],
    )
    year: int = Field(
        ...,
        description="Model year",
        ge=1983,
        le=2025,
        examples=[2020],
    )
    mileage: float = Field(
        ...,
        description="Odometer reading in miles",
        ge=0,
        examples=[35000.0],
    )

    # --- Vehicle specs (required) ---
    engine: str = Field(
        ...,
        min_length=1,
        description="Engine description, e.g. '2.5L I4 DOHC 16V'",
        examples=["2.5l i4 dohc 16v"],
    )
    transmission: str = Field(
        ...,
        min_length=1,
        description="Transmission description, e.g. '8-speed automatic'",
        examples=["8 speed automatic"],
    )
    drivetrain: str = Field(
        ...,
        min_length=1,
        description="Drivetrain type: fwd, rwd, awd, 4wd, or full name",
        examples=["fwd"],
    )
    fuel_type: str = Field(
        ...,
        min_length=1,
        description="Fuel type, e.g. 'gasoline', 'hybrid', 'diesel'",
        examples=["gasoline"],
    )

    # --- Appearance (required) ---
    exterior_color: str = Field(
        ...,
        min_length=1,
        description="Exterior color as listed, e.g. 'silver metallic'",
        examples=["silver metallic"],
    )
    interior_color: str = Field(
        ...,
        min_length=1,
        description="Interior color as listed, e.g. 'black leather'",
        examples=["black leather"],
    )

    # --- Vehicle history flags (required, 0 or 1) ---
    accidents_or_damage: int = Field(
        ...,
        description="1 if accidents or damage reported, 0 otherwise",
        ge=0,
        le=1,
        examples=[0],
    )
    one_owner: int = Field(
        ...,
        description="1 if single-owner vehicle, 0 otherwise",
        ge=0,
        le=1,
        examples=[1],
    )
    personal_use_only: int = Field(
        ...,
        description="1 if personal use only, 0 otherwise",
        ge=0,
        le=1,
        examples=[1],
    )

    # --- Listing metadata (optional — median-imputed if omitted) ---
    mpg: str | float | int | None = Field(
        default=None,
        description=(
            "MPG as a range string ('28-32'), single value string ('30'), "
            "or numeric value (30). If omitted, the training-set median mpg_avg is used."
        ),
        examples=["28-32", "30", 30],
    )
    price_drop: float | None = Field(
        default=None,
        description="Price drop amount in dollars. If omitted, training-set median is used.",
        ge=0,
        examples=[0.0],
    )
    seller_rating: float | None = Field(
        default=None,
        description="Seller rating (typically 1.0-5.0). If omitted, training-set median is used.",
        examples=[4.5],
    )
    driver_rating: float | None = Field(
        default=None,
        description="Driver/consumer rating (typically 1.0-5.0). If omitted, training-set median is used.",
        examples=[4.2],
    )
    driver_reviews_num: float | None = Field(
        default=None,
        description="Number of driver/consumer reviews. If omitted, training-set median is used.",
        ge=0,
        examples=[120.0],
    )

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
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
            ]
        },
    )


# ===============================
# Prediction response
# ===============================

class PredictionResponse(APIModel):
    """Prediction result returned to the API caller."""

    predicted_price: float = Field(
        ...,
        ge=0,
        description="Predicted listing price in USD",
        examples=[27450.00],
    )
    currency: Literal["USD"] = "USD"
    model_used: str = Field(
        default="CatBoost",
        description="Name of the model that produced the prediction",
        examples=["CatBoost"],
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Validation warnings (e.g. unknown manufacturer/model). "
        "The prediction is still returned but may be less reliable.",
    )
    input_echo: dict = Field(
        ...,
        description="Cleaned and normalized version of the submitted inputs, "
        "showing exactly what the model received after preprocessing.",
    )


# ===============================
# Health check response
# ===============================

class HealthResponse(APIModel):
    """Response for the /health liveness probe."""

    status: Literal["ok"] = "ok"
    app_name: str = Field(
        ...,
        description="Application name",
        examples=["Used Car Price API"],
    )
    version: str = Field(
        ...,
        description="Application version",
        examples=["0.1.0"],
    )
    environment: Literal["dev", "test", "prod"] = Field(
        ...,
        description="Current deployment environment",
        examples=["dev"],
    )


# ===============================
# Readiness check response
# ===============================

class ReadinessResponse(APIModel):
    """Response for the /ready readiness probe."""

    status: Literal["ready", "not_ready"]
    model_loaded: bool = Field(
        ...,
        description="Whether the CatBoost model is loaded and ready",
    )
    artifacts_loaded: bool = Field(
        ...,
        description="Whether all JSON artifacts are loaded",
    )
    feature_pipeline_version: str = Field(
        ...,
        description="Version of the feature engineering pipeline",
        examples=["v3"],
    )


# ===============================
# Error response
# ===============================

class ErrorResponse(APIModel):
    """Standard error response body."""

    detail: str = Field(
        ...,
        description="Human-readable error message",
        examples=["Missing required field: manufacturer"],
    )