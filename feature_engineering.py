"""
feature_engineering.py
----------------------
SINGLE SOURCE OF TRUTH for feature transforms.

Both `train_model.py` and `app.py` import from here, so the exact same code
path runs at training time and at inference time. This eliminates
train/serve skew - one of the most common (and silent) bugs in deployed ML.

Two design decisions worth pointing out to a reviewer:

1. Categorical columns are given a FIXED level ordering via
   `pandas.CategoricalDtype`. XGBoost's native categorical support keys off the
   integer *codes* behind a category column. If a single-row inference frame
   only contains one brand, pandas would assign it code 0 - a different code
   than it had in training - and the model would read the wrong feature. Pinning
   the category levels here guarantees identical codes everywhere.

2. Engineered features encode domain knowledge (depreciation age, usage
   intensity, mileage anomaly vs. expected) rather than leaving the model to
   rediscover them from raw columns. This is what makes the model behave like a
   used-car valuer instead of a generic regressor.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

CURRENT_YEAR = 2025
EXPECTED_KM_PER_YEAR = 12_000  # typical Indian usage; drives the anomaly feature

# --- Brand knowledge -------------------------------------------------------
BRAND_TIER = {
    "Maruti Suzuki": "budget",
    "Hyundai": "mass",
    "Tata": "mass",
    "Honda": "mass",
    "Mahindra": "mass",
    "Kia": "mass",
    "Toyota": "premium",
    "Volkswagen": "premium",
    "Skoda": "premium",
    "BMW": "luxury",
    "Mercedes-Benz": "luxury",
    "Audi": "luxury",
}
TIER_RANK = {"budget": 0, "mass": 1, "premium": 2, "luxury": 3}

# --- Fixed categorical levels (pin codes; see module docstring) ------------
CATEGORY_LEVELS = {
    "brand": list(BRAND_TIER.keys()),
    "fuel_type": ["Petrol", "Diesel", "CNG", "Hybrid", "Electric"],
    "transmission": ["Manual", "Automatic"],
    "city_tier": ["Metro", "Tier-2", "Tier-3"],
    "service_history": ["Full", "Partial", "No Record"],
    "accident_history": ["No", "Yes"],
}

# Columns the model actually consumes
NUMERIC_FEATURES = [
    "age",
    "mileage_km",
    "engine_cc",
    "owner_count",
    "mileage_per_year",
    "mileage_ratio",
    "brand_tier_rank",
    "power_to_age",
]
CATEGORICAL_FEATURES = list(CATEGORY_LEVELS.keys())
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "price_lakh"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Pure function - safe to call on 1 row or 1M rows."""
    df = df.copy()

    df["age"] = (CURRENT_YEAR - df["year"]).clip(lower=0)

    # Usage intensity: km driven per year of life.
    df["mileage_per_year"] = df["mileage_km"] / df["age"].clip(lower=1)

    # Mileage anomaly: actual km vs. what a car of this age "should" have.
    # >1 means driven harder than average -> extra depreciation signal.
    expected_total = df["age"].clip(lower=0.5) * EXPECTED_KM_PER_YEAR
    df["mileage_ratio"] = df["mileage_km"] / expected_total.clip(lower=1)

    # Brand tier as an ordinal signal (budget < mass < premium < luxury).
    tier = df["brand"].map(BRAND_TIER).fillna("mass")
    df["brand_tier_rank"] = tier.map(TIER_RANK).astype(int)

    # Engine vitality relative to age (rough proxy for "is this still a
    # desirable powertrain or an old thirsty one").
    df["power_to_age"] = df["engine_cc"] / (df["age"] + 1)

    return df


def apply_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns to fixed-level category dtype (stable codes)."""
    df = df.copy()
    for col, levels in CATEGORY_LEVELS.items():
        df[col] = pd.Categorical(df[col], categories=levels)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: engineer -> select -> pin categoricals. Returns model X."""
    df = engineer(df)
    df = apply_categorical(df)
    return df[FEATURE_COLUMNS]
