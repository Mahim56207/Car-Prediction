"""
model_service.py
----------------
Inference layer. Kept separate from the Streamlit UI so the valuation logic can
be unit-tested and reused (an API, a batch job) without importing streamlit.

Highlights:
  * predict_interval - returns the (q10, q50, q90) price band.
  * explain - converts XGBoost's native TreeSHAP contributions (computed in
    log-price space) into intuitive *multiplicative* price effects. Because we
    model log(price), a SHAP contribution c becomes a price multiplier exp(c):
    "Automatic transmission -> x1.07 (+7%)". This is the mathematically correct
    way to read a log-target model, and it needs no external `shap` package -
    XGBoost emits exact TreeSHAP values via pred_contribs.
  * comparables - pulls similar real listings from the reference sample so the
    estimate is anchored to an actual local market, not just the model.
  * sensitivity_curve - sweeps one feature (e.g. mileage) to expose the model's
    learned depreciation behaviour for THIS specific car.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

import feature_engineering as fe

# Human-readable labels for the explanation chart.
PRETTY = {
    "age": "Age", "mileage_km": "Mileage", "engine_cc": "Engine size",
    "owner_count": "Owners", "mileage_per_year": "Yearly usage",
    "mileage_ratio": "Usage vs. expected", "brand_tier_rank": "Brand tier",
    "power_to_age": "Engine/age", "brand": "Brand", "fuel_type": "Fuel",
    "transmission": "Transmission", "city_tier": "City", "service_history": "Service record",
    "accident_history": "Accident history",
}


def load_artifact(path: str = "model.joblib"):
    bundle = joblib.load(path)
    return bundle["model"], bundle["meta"]


def _row_frame(raw: dict) -> pd.DataFrame:
    return pd.DataFrame([raw])


def predict_interval(model, meta, raw: dict) -> dict:
    """Return q10/q50/q90 price (lakh) for one car."""
    X = fe.build_features(_row_frame(raw))
    pred_log = np.asarray(model.predict(X))
    if pred_log.ndim == 1:
        pred_log = pred_log.reshape(1, -1)
    price = np.exp(pred_log)[0]
    price = np.sort(price)  # guard against quantile crossing
    return {"low": float(price[0]), "mid": float(price[meta["median_idx"]]),
            "high": float(price[-1])}


def explain(model, meta, raw: dict, top_k: int = 7) -> list[dict]:
    """TreeSHAP contributions -> multiplicative price effects, biggest first."""
    X = fe.build_features(_row_frame(raw))
    eng = fe.engineer(_row_frame(raw)).iloc[0]   # for display values
    booster = model.get_booster()
    dm = xgb.DMatrix(X, enable_categorical=True)
    contribs = np.asarray(booster.predict(dm, pred_contribs=True))
    if contribs.ndim == 3:                       # (n, n_quantiles, f+1)
        contribs = contribs[:, meta["median_idx"], :]
    contribs = contribs[0]                        # (f+1,) -> last entry is base
    feat_logs = contribs[:-1]

    def _disp(col):
        v = eng[col] if col in eng.index else raw.get(col, "-")
        return round(float(v), 2) if isinstance(v, (int, float, np.floating)) else v

    out = []
    for col, c in zip(fe.FEATURE_COLUMNS, feat_logs):
        out.append({
            "feature": PRETTY.get(col, col),
            "raw_value": _disp(col),
            "multiplier": float(np.exp(c)),       # x1.07 = +7% on price
            "pct": float((np.exp(c) - 1) * 100),
            "abs": abs(float(c)),
        })
    out.sort(key=lambda d: d["abs"], reverse=True)
    return out[:top_k]


def comparables(meta, raw: dict, age_window: int = 2) -> dict | None:
    """Stats for similar real listings from the reference sample."""
    ref = meta["reference"]
    age = fe.CURRENT_YEAR - raw["year"]
    ref_age = fe.CURRENT_YEAR - ref["year"]
    mask = (ref["brand"] == raw["brand"]) & (ref_age.sub(age).abs() <= age_window)
    sub = ref[mask]
    if len(sub) < 8:  # relax to brand tier if the exact slice is thin
        tier = fe.BRAND_TIER.get(raw["brand"], "mass")
        tier_brands = [b for b, t in fe.BRAND_TIER.items() if t == tier]
        mask = ref["brand"].isin(tier_brands) & (ref_age.sub(age).abs() <= age_window)
        sub = ref[mask]
    if len(sub) < 5:
        return None
    p = sub["price_lakh"]
    return {"n": int(len(sub)), "q25": float(p.quantile(.25)),
            "median": float(p.median()), "q75": float(p.quantile(.75))}


def sensitivity_curve(model, meta, raw: dict, feature: str, grid: np.ndarray) -> pd.DataFrame:
    """Predict the band across a sweep of one feature, holding others fixed."""
    rows = []
    for v in grid:
        r = dict(raw); r[feature] = v
        band = predict_interval(model, meta, r)
        rows.append({feature: v, **band})
    return pd.DataFrame(rows)
