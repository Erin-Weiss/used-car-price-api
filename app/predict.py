"""
Prediction orchestration for the Used Car Price API.

Wires together:
- fuzzy matching and validation against training vocabularies
- median imputation for optional fields
- feature engineering via pipeline.prepare_for_prediction()
- CatBoost model inference
- log1p → dollar conversion

Typical usage:
    from app.predict import run_prediction
    response = run_prediction(request)
"""

from __future__ import annotations

import math
from difflib import get_close_matches
from typing import Any

import pandas as pd
from fastapi import HTTPException

from app.model import (
    get_model,
    get_model_metadata,
    get_categorical_vocabs,
    get_models_by_manufacturer,
    get_numeric_medians,
)
from app.pipeline import normalize_text_single, prepare_for_prediction
from app.schemas import PredictionRequest, PredictionResponse


# ===============================
# Fuzzy matching
# ===============================

# Minimum similarity score (0-1) for a fuzzy match to be accepted.
# 0.6 is the difflib default. Lower catches more typos but risks
# false corrections.
FUZZY_CUTOFF = 0.6

# Maximum number of fuzzy suggestions to consider. We always pick
# the top match if it meets the cutoff.
FUZZY_MAX_MATCHES = 1


def _fuzzy_match(value: str, vocabulary: list | set, cutoff: float = FUZZY_CUTOFF) -> str | None:
    """Find the closest match for a value in a vocabulary.

    Returns the best match if one is found above the cutoff, otherwise None.
    Uses difflib.get_close_matches which compares using SequenceMatcher
    (ratio of matching characters).
    """
    if not value or not vocabulary:
        return None

    # get_close_matches expects a list
    vocab_list = list(vocabulary) if isinstance(vocabulary, set) else vocabulary

    matches = get_close_matches(
        value,
        vocab_list,
        n=FUZZY_MAX_MATCHES,
        cutoff=cutoff,
    )

    return matches[0] if matches else None


# ===============================
# Cached normalized catalog
# ===============================

_normalized_catalog: dict[str, set[str]] | None = None


def _get_normalized_catalog() -> dict[str, set[str]]:
    """Build and cache the normalized models-by-manufacturer catalog.

    Converts the raw JSON artifact into:
        {normalized_manufacturer: {normalized_model_1, normalized_model_2, ...}}

    Built once on first call, then reused for all subsequent requests.
    Set lookup is O(1) vs list lookup is O(n) — matters with 5,680 models.
    """
    global _normalized_catalog

    if _normalized_catalog is not None:
        return _normalized_catalog

    raw_catalog = get_models_by_manufacturer()
    normalized: dict[str, set[str]] = {}

    for manufacturer, models in raw_catalog.items():
        mfr_key = normalize_text_single(manufacturer)
        if not mfr_key:
            continue

        normalized[mfr_key] = {
            normalize_text_single(m)
            for m in models
            if normalize_text_single(m)
        }

    _normalized_catalog = normalized
    return _normalized_catalog


# ===============================
# Validation and fuzzy correction
# ===============================

def _validate_and_correct(raw: dict) -> list[str]:
    """Validate and fuzzy-correct categorical fields against training vocabularies.

    Modifies raw dict in place when a fuzzy match is found.
    Returns a list of warning strings.

    Checks (in order):
    1. manufacturer — fuzzy matched against all known manufacturers
    2. model — fuzzy matched against known models for the (corrected) manufacturer
    3. drivetrain — fuzzy matched against known drivetrain values
    4. fuel_type — fuzzy matched against known fuel type values
    """
    warnings: list[str] = []
    catalog = _get_normalized_catalog()
    vocabs = get_categorical_vocabs()

    # --- Manufacturer ---
    manufacturer = normalize_text_single(raw.get("manufacturer", ""))
    known_manufacturers = set(catalog.keys())

    if manufacturer not in known_manufacturers:
        match = _fuzzy_match(manufacturer, known_manufacturers)
        if match:
            warnings.append(
                f"Unknown manufacturer '{manufacturer}', "
                f"corrected to '{match}'."
            )
            raw["manufacturer"] = match
            manufacturer = match
        else:
            warnings.append(
                f"Unknown manufacturer '{manufacturer}'. "
                f"No close match found. Prediction may be less reliable."
            )

    # --- Model (conditional on manufacturer) ---
    model_name = normalize_text_single(raw.get("model", ""))
    known_models = catalog.get(manufacturer)

    if known_models is not None and model_name not in known_models:
        match = _fuzzy_match(model_name, known_models)
        if match:
            warnings.append(
                f"Unknown model '{model_name}' for manufacturer '{manufacturer}', "
                f"corrected to '{match}'."
            )
            raw["model"] = match
        else:
            warnings.append(
                f"Unknown model '{model_name}' for manufacturer '{manufacturer}'. "
                f"No close match found. Prediction may be less reliable."
            )

    # --- Drivetrain ---
    drivetrain = normalize_text_single(raw.get("drivetrain", ""))
    known_drivetrains = set(vocabs.get("drivetrain", []))

    if drivetrain and known_drivetrains and drivetrain not in known_drivetrains:
        match = _fuzzy_match(drivetrain, known_drivetrains)
        if match:
            warnings.append(
                f"Unknown drivetrain '{drivetrain}', "
                f"corrected to '{match}'."
            )
            raw["drivetrain"] = match
        else:
            warnings.append(
                f"Unknown drivetrain '{drivetrain}'. "
                f"No close match found. Prediction may be less reliable."
            )

    # --- Fuel type ---
    fuel_type = normalize_text_single(raw.get("fuel_type", ""))
    known_fuel_types = set(vocabs.get("fuel_type", []))

    if fuel_type and known_fuel_types and fuel_type not in known_fuel_types:
        match = _fuzzy_match(fuel_type, known_fuel_types)
        if match:
            warnings.append(
                f"Unknown fuel type '{fuel_type}', "
                f"corrected to '{match}'."
            )
            raw["fuel_type"] = match
        else:
            warnings.append(
                f"Unknown fuel type '{fuel_type}'. "
                f"No close match found. Prediction may be less reliable."
            )

    return warnings


# ===============================
# Imputation
# ===============================

def _impute_optional_fields(raw: dict) -> dict:
    """Fill None values for optional fields with training-set medians.

    Handles mpg specially: if omitted, removes the 'mpg' key and
    inserts 'mpg_avg' directly so the pipeline skips string parsing.
    """
    medians = get_numeric_medians()

    # --- MPG: special handling ---
    if raw.get("mpg") is None:
        raw.pop("mpg", None)
        raw["mpg_avg"] = medians["mpg_avg"]

    # --- Other optional numeric fields ---
    optional_fields = ["price_drop", "seller_rating", "driver_rating", "driver_reviews_num"]
    for field in optional_fields:
        if raw.get(field) is None:
            raw[field] = medians[field]

    return raw


# ===============================
# Output helpers
# ===============================

def _to_builtin_dict(record: dict) -> dict[str, Any]:
    """Convert pandas/numpy scalar values into plain Python types
    so the response is JSON-serializable.

    Handles np.int64, np.float64, np.nan, etc.
    """
    cleaned: dict[str, Any] = {}

    for key, value in record.items():
        if pd.isna(value):
            cleaned[key] = None
        elif hasattr(value, "item"):
            # numpy scalar → Python scalar (e.g. np.int64 → int)
            cleaned[key] = value.item()
        else:
            cleaned[key] = value

    return cleaned


# ===============================
# Main entry point
# ===============================

def run_prediction(request: PredictionRequest) -> PredictionResponse:
    """Execute a full prediction from raw request to response.

    Steps:
    1. Convert request to dict
    2. Validate and fuzzy-correct against training vocabularies
    3. Impute optional fields with training-set medians
    4. Run feature engineering via prepare_for_prediction()
    5. Predict with CatBoost (returns log1p scale)
    6. Convert to dollars via expm1
    7. Build and return response with input echo
    """
    # 1. Convert to dict
    raw = request.model_dump()

    # 2. Validate and fuzzy-correct
    warnings = _validate_and_correct(raw)

    # 3. Impute
    raw = _impute_optional_fields(raw)

    # 4. Feature engineering
    feature_columns = get_model_metadata()["feature_columns"]

    try:
        features_df = prepare_for_prediction(raw, feature_columns)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Feature preparation failed: {exc}",
        ) from exc

    # 5. Predict (log1p scale)
    try:
        log_prediction = float(get_model().predict(features_df)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {exc}",
        ) from exc

    # 6. Convert to dollars
    predicted_price = round(max(math.expm1(log_prediction), 0.0), 2)

    # 7. Build response
    input_echo = _to_builtin_dict(features_df.iloc[0].to_dict())

    return PredictionResponse(
        predicted_price=predicted_price,
        warnings=warnings,
        input_echo=input_echo,
    )