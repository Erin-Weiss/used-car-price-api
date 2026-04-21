"""
Unit tests for pipeline.py functions.

Tests normalization, engine parsing, transmission parsing,
color collapsing, drivetrain mapping, and prepare_for_prediction.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.pipeline import (
    DRIVETRAIN_MAP,
    LUXURY_BRANDS,
    base_color,
    interior_color_base,
    normalize_text_basic,
    normalize_text_single,
    parse_engine,
    prepare_for_prediction,
    transmission_gears,
    transmission_type,
)

from tests.conftest import SAMPLE_MODEL_METADATA


# ===============================
# Text normalization
# ===============================

class TestNormalizeTextBasic:
    def test_lowercases(self):
        result = normalize_text_basic(pd.Series(["TOYOTA"]))
        assert result.iloc[0] == "toyota"

    def test_strips_whitespace(self):
        result = normalize_text_basic(pd.Series(["  toyota  "]))
        assert result.iloc[0] == "toyota"

    def test_replaces_hyphens_with_space(self):
        result = normalize_text_basic(pd.Series(["A-SPEC"]))
        assert result.iloc[0] == "a spec"

    def test_replaces_slashes_with_space(self):
        result = normalize_text_basic(pd.Series(["Gas/Electric"]))
        assert result.iloc[0] == "gas electric"

    def test_pads_ampersand(self):
        result = normalize_text_basic(pd.Series(["TECHNOLOGY&A-SPEC"]))
        assert " & " in result.iloc[0]

    def test_collapses_multiple_spaces(self):
        result = normalize_text_basic(pd.Series(["hello    world"]))
        assert result.iloc[0] == "hello world"

    def test_combined_normalization(self):
        result = normalize_text_basic(pd.Series(["  ILX TECHNOLOGY&A-SPEC PACKAGES  "]))
        assert result.iloc[0] == "ilx technology & a spec packages"


class TestNormalizeTextSingle:
    def test_lowercases(self):
        assert normalize_text_single("TOYOTA") == "toyota"

    def test_strips_whitespace(self):
        assert normalize_text_single("  toyota  ") == "toyota"

    def test_replaces_hyphens(self):
        assert normalize_text_single("A-SPEC") == "a spec"

    def test_replaces_slashes(self):
        assert normalize_text_single("Gas/Electric") == "gas electric"

    def test_pads_ampersand(self):
        result = normalize_text_single("TECHNOLOGY&A-SPEC")
        assert " & " in result

    def test_matches_basic_output(self):
        """Single and Series normalizers should produce identical results."""
        test_values = [
            "Mercedes-Benz",
            "ILX TECHNOLOGY&A-SPEC PACKAGES",
            "3.5L V6 24V PDI DOHC Twin Turbo Hybrid",
            "9-Speed Automatic with Auto-Shift",
        ]
        for val in test_values:
            single = normalize_text_single(val)
            series = normalize_text_basic(pd.Series([val])).iloc[0]
            assert single == series, f"Mismatch for '{val}': single='{single}', series='{series}'"


# ===============================
# Engine parsing
# ===============================

class TestParseEngine:
    def test_parses_liters(self):
        result = parse_engine("2.5l i4 dohc 16v")
        assert result["engine_liters"] == 2.5

    def test_parses_cylinders(self):
        result = parse_engine("2.5l i4 dohc 16v")
        assert result["engine_cylinders"] == 4

    def test_parses_layout_inline(self):
        result = parse_engine("2.5l i4 dohc 16v")
        assert result["engine_layout"] == "i"

    def test_parses_layout_v(self):
        result = parse_engine("3.5l v6 24v")
        assert result["engine_layout"] == "v"

    def test_detects_turbo(self):
        result = parse_engine("2.0l i4 turbo")
        assert result["engine_turbo"] == 1

    def test_detects_twin_turbo(self):
        result = parse_engine("3.5l v6 twin turbo")
        assert result["engine_turbo"] == 1

    def test_detects_supercharger(self):
        result = parse_engine("5.0l v8 supercharged")
        assert result["engine_turbo"] == 1

    def test_no_turbo(self):
        result = parse_engine("2.5l i4 dohc")
        assert result["engine_turbo"] == 0

    def test_detects_hybrid(self):
        result = parse_engine("2.5l i4 hybrid")
        assert result["engine_hybrid"] == 1

    def test_detects_gas_electric(self):
        result = parse_engine("twin turbo gas electric v 6 3.5 l 213")
        assert result["engine_hybrid"] == 1

    def test_detects_phev(self):
        result = parse_engine("2.0l i4 phev")
        assert result["engine_hybrid"] == 1

    def test_detects_plug_in(self):
        result = parse_engine("1.5l i4 plug in hybrid")
        assert result["engine_hybrid"] == 1

    def test_no_hybrid(self):
        result = parse_engine("2.5l i4 dohc")
        assert result["engine_hybrid"] == 0

    def test_missing_liters_returns_nan(self):
        result = parse_engine("v8 flex fuel")
        assert pd.isna(result["engine_liters"])

    def test_missing_layout_returns_nan(self):
        result = parse_engine("2.5l dohc")
        assert pd.isna(result["engine_layout"])


# ===============================
# Transmission parsing
# ===============================

class TestTransmissionType:
    def test_automatic(self):
        assert transmission_type("8 speed automatic") == "automatic"

    def test_manual(self):
        assert transmission_type("6 speed manual") == "manual"

    def test_cvt(self):
        assert transmission_type("cvt") == "cvt"

    def test_variable(self):
        assert transmission_type("continuously variable") == "cvt"

    def test_a_t_maps_to_automatic(self):
        assert transmission_type("a t") == "automatic"

    def test_unknown_for_none(self):
        assert transmission_type(None) == "unknown"

    def test_unknown_for_not_specified(self):
        assert transmission_type("not specified") == "unknown"


class TestTransmissionGears:
    def test_parses_gear_count(self):
        assert transmission_gears("8 speed automatic") == 8

    def test_parses_six_speed(self):
        assert transmission_gears("6 speed manual") == 6

    def test_returns_nan_when_missing(self):
        result = transmission_gears("automatic")
        assert pd.isna(result)

    def test_returns_nan_for_none(self):
        result = transmission_gears(None)
        assert pd.isna(result)


# ===============================
# Color collapsing
# ===============================

class TestBaseColor:
    def test_black(self):
        assert base_color("jet black") == "black"

    def test_ebony(self):
        assert base_color("ebony pearl") == "black"

    def test_blue(self):
        assert base_color("deep blue metallic") == "blue"

    def test_red(self):
        assert base_color("scarlet ember") == "red"

    def test_green(self):
        assert base_color("army green") == "green"

    def test_white(self):
        assert base_color("pearl white") == "white"

    def test_gray_from_silver(self):
        assert base_color("silver metallic") == "gray"

    def test_gray_from_grey(self):
        assert base_color("dark grey") == "gray"

    def test_other(self):
        assert base_color("orange sunset") == "other"


class TestInteriorColorBase:
    def test_black(self):
        assert interior_color_base("black leather") == "black"

    def test_gray(self):
        assert interior_color_base("charcoal cloth") == "gray"

    def test_brown(self):
        assert interior_color_base("espresso leather") == "brown"

    def test_beige(self):
        assert interior_color_base("tan leather") == "beige"

    def test_white(self):
        assert interior_color_base("ivory") == "white"

    def test_red(self):
        assert interior_color_base("red leather") == "red"

    def test_other(self):
        assert interior_color_base("orange suede") == "other"


# ===============================
# Drivetrain mapping
# ===============================

class TestDrivetrainMap:
    def test_fwd(self):
        assert DRIVETRAIN_MAP["front wheel drive"] == "fwd"
        assert DRIVETRAIN_MAP["fwd"] == "fwd"

    def test_rwd(self):
        assert DRIVETRAIN_MAP["rear wheel drive"] == "rwd"
        assert DRIVETRAIN_MAP["rwd"] == "rwd"

    def test_awd(self):
        assert DRIVETRAIN_MAP["all wheel drive"] == "awd"
        assert DRIVETRAIN_MAP["awd"] == "awd"

    def test_4wd(self):
        assert DRIVETRAIN_MAP["four wheel drive"] == "4wd"
        assert DRIVETRAIN_MAP["4wd"] == "4wd"


# ===============================
# Luxury brands
# ===============================

class TestLuxuryBrands:
    def test_bmw_is_luxury(self):
        assert "bmw" in LUXURY_BRANDS

    def test_toyota_is_not_luxury(self):
        assert "toyota" not in LUXURY_BRANDS

    def test_rolls_royce_no_hyphen(self):
        assert "rolls royce" in LUXURY_BRANDS
        assert "rolls-royce" not in LUXURY_BRANDS

    def test_mercedes_no_hyphen(self):
        assert "mercedes benz" in LUXURY_BRANDS


# ===============================
# prepare_for_prediction
# ===============================

class TestPrepareForPrediction:
    """Direct tests for the API inference path."""

    @pytest.fixture
    def feature_columns(self):
        return SAMPLE_MODEL_METADATA["feature_columns"]

    @pytest.fixture
    def base_input(self):
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

    def test_mpg_range_parsed_to_average(self, base_input, feature_columns):
        df = prepare_for_prediction(base_input, feature_columns)
        assert df["mpg_avg"].iloc[0] == 30.0

    def test_output_columns_match_feature_columns(self, base_input, feature_columns):
        df = prepare_for_prediction(base_input, feature_columns)
        assert list(df.columns) == feature_columns

    def test_age_calculated_correctly(self, base_input, feature_columns):
        df = prepare_for_prediction(base_input, feature_columns)
        # REFERENCE_YEAR (2023) - 2020 = 3
        assert df["age"].iloc[0] == 3

    def test_mileage_per_year_calculated(self, base_input, feature_columns):
        df = prepare_for_prediction(base_input, feature_columns)
        # 35000 / 3 ≈ 11666.67
        assert abs(df["mileage_per_year"].iloc[0] - 11666.67) < 1

    def test_luxury_brand_flag_for_toyota(self, base_input, feature_columns):
        df = prepare_for_prediction(base_input, feature_columns)
        assert df["is_luxury_brand"].iloc[0] == 0

    def test_luxury_brand_flag_for_bmw(self, base_input, feature_columns):
        base_input["manufacturer"] = "bmw"
        base_input["model"] = "x5"
        df = prepare_for_prediction(base_input, feature_columns)
        assert df["is_luxury_brand"].iloc[0] == 1

    def test_mercedes_benz_is_luxury_after_normalization(self, base_input, feature_columns):
        base_input["manufacturer"] = "Mercedes-Benz"
        base_input["model"] = "c300"
        df = prepare_for_prediction(base_input, feature_columns)
        assert df["is_luxury_brand"].iloc[0] == 1

    def test_exterior_color_collapsed(self, base_input, feature_columns):
        df = prepare_for_prediction(base_input, feature_columns)
        assert df["exterior_color_base"].iloc[0] == "gray"

    def test_interior_color_collapsed(self, base_input, feature_columns):
        df = prepare_for_prediction(base_input, feature_columns)
        assert df["interior_color_base"].iloc[0] == "black"

    def test_unknown_transmission_type(self, base_input, feature_columns):
        base_input["transmission"] = "not specified"
        df = prepare_for_prediction(base_input, feature_columns)
        assert df["transmission_clean"].iloc[0] == "unknown"

    def test_missing_gear_count_sets_flag(self, base_input, feature_columns):
        base_input["transmission"] = "automatic"
        df = prepare_for_prediction(base_input, feature_columns)
        assert df["transmission_gears_missing"].iloc[0] == 1

    def test_seller_name_does_not_break(self, base_input, feature_columns):
        base_input["seller_name"] = "Bob's Auto Shop"
        df = prepare_for_prediction(base_input, feature_columns)
        assert "seller_name" not in df.columns

    def test_drivetrain_mapped_correctly(self, base_input, feature_columns):
        base_input["drivetrain"] = "Front-Wheel Drive"
        df = prepare_for_prediction(base_input, feature_columns)
        assert df["drivetrain"].iloc[0] == "fwd"

    def test_output_is_single_row(self, base_input, feature_columns):
        df = prepare_for_prediction(base_input, feature_columns)
        assert len(df) == 1
