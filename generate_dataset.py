"""
generate_dataset.py
-------------------
Builds a realistic used-car dataset for the Indian market (prices in lakh INR).

Why synthetic instead of a Kaggle CSV? Two reasons that actually help the
project:

  * The data-generating process is KNOWN, so we can check whether the model
    recovered the true economic relationships (depreciation curves, brand
    premiums, mileage penalties). That turns "the model gets low error" into
    "the model learned the right physics".
  * Features are generated with realistic *joint* structure - older cars have
    more km and more owners, luxury cars are bigger and more often automatic -
    so the model faces the same correlations it would in the wild.

The pricing function below is the ground truth. Training never sees it; it only
sees the noisy `price_lakh` it produces.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
CURRENT_YEAR = 2025

# Average ex-showroom anchor price (lakh INR) for a representative new model.
BRAND_BASE_PRICE = {
    "Maruti Suzuki": 7.0, "Hyundai": 9.0, "Tata": 10.0, "Honda": 11.0,
    "Mahindra": 13.0, "Kia": 12.0, "Toyota": 16.0, "Volkswagen": 15.0,
    "Skoda": 18.0, "BMW": 45.0, "Mercedes-Benz": 55.0, "Audi": 50.0,
}
BRAND_WEIGHTS = {  # market share -> sampling frequency
    "Maruti Suzuki": 0.26, "Hyundai": 0.17, "Tata": 0.12, "Honda": 0.09,
    "Mahindra": 0.08, "Kia": 0.07, "Toyota": 0.06, "Volkswagen": 0.04,
    "Skoda": 0.03, "BMW": 0.030, "Mercedes-Benz": 0.025, "Audi": 0.025,
}
TIER = {
    "Maruti Suzuki": "budget", "Hyundai": "mass", "Tata": "mass",
    "Honda": "mass", "Mahindra": "mass", "Kia": "mass", "Toyota": "premium",
    "Volkswagen": "premium", "Skoda": "premium", "BMW": "luxury",
    "Mercedes-Benz": "luxury", "Audi": "luxury",
}
# Luxury cars shed value faster than budget cars (well-known resale fact).
TIER_DECAY = {"budget": 0.11, "mass": 0.12, "premium": 0.14, "luxury": 0.17}

FUEL_MULT = {"Petrol": 1.00, "Diesel": 1.04, "CNG": 0.95, "Hybrid": 1.08, "Electric": 1.02}
SERVICE_MULT = {"Full": 1.05, "Partial": 1.00, "No Record": 0.93}
CITY_MULT = {"Metro": 1.03, "Tier-2": 1.00, "Tier-3": 0.97}


def _sample_brands(n):
    brands = list(BRAND_WEIGHTS)
    p = np.array([BRAND_WEIGHTS[b] for b in brands])
    p = p / p.sum()
    return RNG.choice(brands, size=n, p=p)


def _sample_fuel(tier, n):
    # Budget cars skew petrol/CNG; luxury skews petrol/diesel; some EVs recently.
    table = {
        "budget":  {"Petrol": .60, "Diesel": .08, "CNG": .28, "Hybrid": .02, "Electric": .02},
        "mass":    {"Petrol": .52, "Diesel": .30, "CNG": .10, "Hybrid": .05, "Electric": .03},
        "premium": {"Petrol": .45, "Diesel": .40, "CNG": .02, "Hybrid": .09, "Electric": .04},
        "luxury":  {"Petrol": .48, "Diesel": .42, "CNG": .00, "Hybrid": .06, "Electric": .04},
    }[tier]
    keys = list(table)
    return RNG.choice(keys, size=n, p=np.array(list(table.values())))


def generate(n: int = 12_000) -> pd.DataFrame:
    brand = _sample_brands(n)
    tier = np.array([TIER[b] for b in brand])

    # Age: skewed toward newer cars (more recent listings dominate the market).
    age = np.clip(RNG.gamma(shape=2.2, scale=2.4, size=n).round().astype(int), 0, 17)
    year = CURRENT_YEAR - age

    # Mileage grows with age but with heavy individual variation.
    base_km_per_year = RNG.normal(11_000, 3_500, size=n).clip(3_000, 35_000)
    mileage = (base_km_per_year * np.maximum(age, 0.4)).round().astype(int)
    mileage = np.clip(mileage, 500, 300_000)

    # Engine size depends on tier (luxury bigger).
    engine_mean = {"budget": 1100, "mass": 1300, "premium": 1600, "luxury": 2200}
    engine_cc = np.array([
        int(np.clip(RNG.normal(engine_mean[t], 250), 800, 4000)) for t in tier
    ])

    fuel = np.empty(n, dtype=object)
    for t in np.unique(tier):
        mask = tier == t
        fuel[mask] = _sample_fuel(t, mask.sum())
    # Electric cars don't have a normal engine_cc -> set small token value.
    engine_cc = np.where(fuel == "Electric", np.minimum(engine_cc, 1000), engine_cc)

    # Automatic is far more common in luxury.
    p_auto = np.where(tier == "luxury", 0.85,
              np.where(tier == "premium", 0.55,
              np.where(tier == "mass", 0.30, 0.18)))
    transmission = np.where(RNG.random(n) < p_auto, "Automatic", "Manual")

    # Owner count rises with age.
    owner_lambda = 1.0 + age * 0.12
    owner_count = np.clip(RNG.poisson(owner_lambda) + 1, 1, 5)

    accident = np.where(RNG.random(n) < 0.12, "Yes", "No")
    service = RNG.choice(["Full", "Partial", "No Record"], size=n, p=[0.45, 0.35, 0.20])
    city = RNG.choice(["Metro", "Tier-2", "Tier-3"], size=n, p=[0.45, 0.35, 0.20])

    # ---------------- GROUND-TRUTH PRICE -----------------------------------
    base = np.array([BRAND_BASE_PRICE[b] for b in brand])
    decay = np.array([TIER_DECAY[t] for t in tier])
    price = base * np.exp(-decay * age)                       # depreciation curve
    price *= np.exp(-0.15 * (mileage / 100_000))              # mileage wear
    price *= np.array([FUEL_MULT[f] for f in fuel])
    price *= np.where(transmission == "Automatic", 1.06, 1.0)
    price *= 0.93 ** (owner_count - 1)                        # each extra owner
    price *= np.where(accident == "Yes", 0.82, 1.0)
    price *= np.array([SERVICE_MULT[s] for s in service])
    price *= np.array([CITY_MULT[c] for c in city])
    price *= 1 + 0.02 * (engine_cc - 1300) / 500              # mild engine premium
    price *= RNG.lognormal(mean=0.0, sigma=0.10, size=n)      # real market noise
    price = np.clip(price, 0.4, None).round(2)

    return pd.DataFrame({
        "brand": brand,
        "year": year,
        "mileage_km": mileage,
        "fuel_type": fuel,
        "transmission": transmission,
        "engine_cc": engine_cc,
        "owner_count": owner_count,
        "accident_history": accident,
        "service_history": service,
        "city_tier": city,
        "price_lakh": price,
    })


if __name__ == "__main__":
    df = generate()
    df.to_csv("car_data.csv", index=False)
    print(f"Wrote car_data.csv with {len(df):,} rows")
    print(df.head())
    print("\nPrice (lakh) summary:")
    print(df["price_lakh"].describe().round(2))
