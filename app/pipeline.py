
from __future__ import annotations

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path


# ===============================
# Constants
# ===============================

REFERENCE_YEAR = 2023  # The dataset ends at the beginning of 2023

TEXT_COLUMNS = [
    "manufacturer", "model", "drivetrain", "fuel_type",
    "engine", "transmission", "exterior_color", "interior_color",
]

DRIVETRAIN_MAP = {
    "all wheel drive": "awd",
    "awd":             "awd",
    "four wheel drive": "4wd",
    "4wd":              "4wd",
    "front wheel drive": "fwd",
    "fwd":               "fwd",
    "rear wheel drive":  "rwd",
    "rwd":               "rwd",
}

LUXURY_BRANDS = {
    "acura", "alfa romeo", "aston martin", "audi", "bentley",
    "bmw", "bugatti", "cadillac", "ferrari", "genesis",
    "infiniti", "jaguar", "lamborghini", "land rover", "lexus",
    "lincoln", "lotus", "maserati", "mclaren", "mercedes benz",
    "polestar", "porsche", "rolls royce", "tesla", "volvo",
}


# ===============================
# Text normalization
# ===============================

def normalize_text_basic(s: pd.Series) -> pd.Series:
    """Normalize a pandas Series of strings: lowercase, strip,
    replace hyphens/slashes with spaces, pad &, collapse whitespace."""
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .str.replace(r'[-/]', ' ', regex=True)
         .str.replace(r'&', ' & ', regex=False)
         .str.replace(r'\s+', ' ', regex=True)
         .str.strip()
    )


def normalize_text_single(s: str) -> str:
    """Normalize a single string. Same logic as normalize_text_basic
    but for scalar values (used by the API inference path)."""
    s = str(s).strip().lower()
    s = re.sub(r'[-/]', ' ', s)
    s = s.replace('&', ' & ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


# ===============================
# Engine parsing
# ===============================

def parse_engine(engine: str) -> pd.Series:
    """Extract structured features from a pre-normalized engine string."""
    s = str(engine)

    m_l = re.search(r"(\d(?:\.\d)?)\s*l\b", s)
    liters = float(m_l.group(1)) if m_l else np.nan

    m_c = re.search(r"\b([ivh])\s*(\d+)\b", s)
    if m_c:
        layout = m_c.group(1)
        cylinders = int(m_c.group(2))
    else:
        layout = np.nan
        cylinders = np.nan

    turbo = int(("turbo" in s) or ("twin turbo" in s) or ("supercharg" in s))

    hybrid = int(
        ("hybrid" in s) or
        ("gas electric" in s) or
        ("phev" in s) or
        ("plug in" in s) or
        ("electric" in s and "gas" in s)
    )

    return pd.Series({
        "engine_liters": liters,
        "engine_cylinders": cylinders,
        "engine_layout": layout,
        "engine_turbo": turbo,
        "engine_hybrid": hybrid,
    })


# ===============================
# Transmission parsing
# ===============================

def normalize_transmission(t: str) -> str:
    if pd.isna(t):
        return "unknown"
    t = str(t)
    t = t.replace("a t", "automatic")
    t = t.replace("auto", "automatic")
    return t


def transmission_type(t: str) -> str:
    t = normalize_transmission(t)
    if t == "unknown" or "not specified" in t:
        return "unknown"
    if "manual" in t:
        return "manual"
    if ("cvt" in t) or ("variable" in t) or ("ivt" in t) or ("ecvt" in t):
        return "cvt"
    return "automatic"


def transmission_gears(t: str) -> float:
    t = normalize_transmission(t)
    match = re.search(r"(\d+)\s?speed", t)
    return int(match.group(1)) if match else np.nan


# ===============================
# Color collapsing
# ===============================

def base_color(c: str) -> str:
    c = str(c)
    if "black" in c or "ebony" in c:
        return "black"
    if "blue" in c or "deep cerulean" in c:
        return "blue"
    if "red" in c or "scarlet ember" in c or "maroon" in c or "dark cherry" in c:
        return "red"
    if "green" in c or "army green" in c or "f8 green" in c or "dark moss" in c or "sarge green clearcoat" in c:
        return "green"
    if "white" in c or "pearl" in c or "whiite" in c:
        return "white"
    if "gray" in c or "silver" in c or "grey" in c or "gun" in c or "steel" in c or "magnetic" in c or " metal " in c or "carbon" in c or "granite" in c or "graphite" in c:
        return "gray"
    return "other"


def interior_color_base(c: str) -> str:
    c = str(c)
    if "black" in c or "ebony" in c or "jet" in c:
        return "black"
    if "red" in c:
        return "red"
    if "gray" in c or "grey" in c or "graphite" in c or "charcoal" in c or "shale" in c or "steel" in c or "pewter" in c or "slate" in c:
        return "gray"
    if "brown" in c or "cappuccino" in c or "mocha" in c or "espresso" in c or "cocoa" in c or "coffee" in c or "nutmeg" in c or "walnut" in c or "chestnut" in c or "hazelnut" in c or "roast" in c:
        return "brown"
    if "beige" in c or "tan" in c or "taupe" in c or "sand" in c or "ash" in c or "camel" in c or "cognac" in c or "parchment" in c or "stone" in c or "wheat" in c or "sandstone" in c or "cement" in c or "almond" in c or "blond" in c or "neutral" in c:
        return "beige"
    if "white" in c or "ivory" in c or "cream" in c:
        return "white"
    return "other"


# ===============================
# Batch pipeline: build_cleaned_df
# ===============================

def build_cleaned_df(raw_csv_path) -> pd.DataFrame:
    """Load raw CSV, compute mpg_avg, drop nulls, cast binary flags,
    normalize text columns, map drivetrain."""

    df_local = pd.read_csv(raw_csv_path)
    df_cleaned = df_local.copy()

    mpg_split = df_cleaned["mpg"].astype(str).str.split("-", expand=True)
    df_cleaned["mpg_avg"] = (
        mpg_split
        .apply(pd.to_numeric, errors="coerce")
        .mean(axis=1)
    )
    df_cleaned = df_cleaned.drop(columns="mpg")

    df_cleaned = df_cleaned.dropna().reset_index(drop=True)

    bin_cols = ["accidents_or_damage", "one_owner", "personal_use_only"]
    for c in bin_cols:
        if c in df_cleaned.columns:
            df_cleaned[c] = df_cleaned[c].round().astype("Int64")

    for col in TEXT_COLUMNS:
        df_cleaned[col] = normalize_text_basic(df_cleaned[col])

    df_cleaned["drivetrain"] = df_cleaned["drivetrain"].replace(DRIVETRAIN_MAP)

    return df_cleaned


# ===============================
# Batch pipeline: build_features_df
# ===============================

def build_features_df(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    """Transform cleaned dataframe into model-ready features + target."""

    y = np.log1p(df_cleaned["price"])
    X = df_cleaned.drop(columns=["price", "seller_name"], errors="ignore").copy()

    # Age and mileage per year
    X["age"] = REFERENCE_YEAR - X["year"]
    X["age_for_mpy"] = X["age"].clip(lower=1)
    X["mileage_per_year"] = X["mileage"] / X["age_for_mpy"]
    X = X.drop(columns=["age_for_mpy", "year"])

    # Luxury brand flag + interaction
    X["is_luxury_brand"] = X["manufacturer"].isin(LUXURY_BRANDS).astype("int64")
    X["luxury_age_interaction"] = X["is_luxury_brand"] * X["age"]

    # Parse engine
    engine_features = X["engine"].apply(parse_engine)
    X = pd.concat([X.drop(columns=["engine"]), engine_features], axis=1)
    X["engine_layout"] = X["engine_layout"].fillna("unknown")

    # Parse transmission
    X["transmission_clean"] = X["transmission"].apply(transmission_type)
    X["transmission_gears"] = X["transmission"].apply(transmission_gears)
    X = X.drop(columns=["transmission"])
    X["transmission_gears_missing"] = X["transmission_gears"].isna().astype(int)

    # Collapse colors
    X["exterior_color_base"] = X["exterior_color"].apply(base_color)
    X = X.drop(columns=["exterior_color"])

    X["interior_color_base"] = X["interior_color"].apply(interior_color_base)
    X = X.drop(columns=["interior_color"])

    # Type casts
    numeric_casts = {
        "age": "int64",
        "mileage_per_year": "float64",
        "engine_liters": "float64",
        "engine_cylinders": "float64",
        "transmission_gears": "float64",
        "engine_turbo": "int64",
        "engine_hybrid": "int64",
        "transmission_gears_missing": "int64",
    }
    for col, dtype in numeric_casts.items():
        if col in X.columns:
            X[col] = X[col].astype(dtype)

    for col in ["engine_layout", "transmission_clean", "exterior_color_base", "interior_color_base"]:
        if col in X.columns:
            X[col] = X[col].astype("object")

    features_df = X.copy()
    features_df["target_log1p_price"] = y.values
    return features_df


# ===============================
# API inference: prepare_for_prediction
# ===============================

def prepare_for_prediction(
    raw_input: dict,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Transform a single raw API input dict into a feature DataFrame
    matching the training schema.

    Parameters
    ----------
    raw_input : dict
        Raw vehicle attributes from the API request. Expected keys:
        manufacturer, model, year, mileage, engine, transmission,
        drivetrain, fuel_type, exterior_color, interior_color,
        mpg (string like "30-32" or numeric), price_drop,
        seller_rating, driver_rating, driver_reviews_num,
        accidents_or_damage, one_owner, personal_use_only
    feature_columns : list[str]
        Ordered list of feature column names from training.
        Used to ensure the output matches the model's expected schema.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns in feature_columns order,
        ready for model.predict().
    """
    df = pd.DataFrame([raw_input])

    # --- MPG: handle "30-32" string or plain numeric ---
    if "mpg" in df.columns:
        mpg_split = df["mpg"].astype(str).str.split("-", expand=True)
        df["mpg_avg"] = (
            mpg_split
            .apply(pd.to_numeric, errors="coerce")
            .mean(axis=1)
        )
        df = df.drop(columns=["mpg"])

    # --- Binary flags ---
    bin_cols = ["accidents_or_damage", "one_owner", "personal_use_only"]
    for c in bin_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")

    # --- Normalize text ---
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(normalize_text_single)

    # --- Drivetrain mapping ---
    if "drivetrain" in df.columns:
        df["drivetrain"] = df["drivetrain"].replace(DRIVETRAIN_MAP)

    # --- Drop seller_name if present ---
    df = df.drop(columns=["seller_name"], errors="ignore")

    # --- Age and mileage per year ---
    df["age"] = REFERENCE_YEAR - df["year"].astype(int)
    age_for_mpy = df["age"].clip(lower=1)
    df["mileage_per_year"] = df["mileage"].astype(float) / age_for_mpy
    df = df.drop(columns=["year"], errors="ignore")

    # --- Luxury brand flag + interaction ---
    df["is_luxury_brand"] = df["manufacturer"].isin(LUXURY_BRANDS).astype("int64")
    df["luxury_age_interaction"] = df["is_luxury_brand"] * df["age"]

    # --- Parse engine ---
    engine_parsed = parse_engine(df["engine"].iloc[0])
    df[engine_parsed.index] = engine_parsed.values

    df = df.drop(columns=["engine"])
    df["engine_layout"] = df["engine_layout"].fillna("unknown")

    # --- Parse transmission ---
    trans_val = df["transmission"].iloc[0]
    df["transmission_clean"] = transmission_type(trans_val)
    df["transmission_gears"] = transmission_gears(trans_val)
    df = df.drop(columns=["transmission"])
    df["transmission_gears_missing"] = int(pd.isna(df["transmission_gears"].iloc[0]))

    # --- Collapse colors ---
    df["exterior_color_base"] = base_color(df["exterior_color"].iloc[0])
    df = df.drop(columns=["exterior_color"])

    df["interior_color_base"] = interior_color_base(df["interior_color"].iloc[0])
    df = df.drop(columns=["interior_color"])

    # --- Type casts ---
    numeric_casts = {
        "age": "int64",
        "mileage_per_year": "float64",
        "engine_liters": "float64",
        "engine_cylinders": "float64",
        "transmission_gears": "float64",
        "engine_turbo": "int64",
        "engine_hybrid": "int64",
        "transmission_gears_missing": "int64",
    }
    for col, dtype in numeric_casts.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    for col in ["engine_layout", "transmission_clean", "exterior_color_base", "interior_color_base"]:
        if col in df.columns:
            df[col] = df[col].astype("object")

    # --- Reorder columns to match training schema ---
    df = df[feature_columns]

    return df
