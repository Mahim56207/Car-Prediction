"""
train_model.py
--------------
Trains the valuation model and saves a single artifact (model.joblib).

What makes this more than a vanilla regressor:

1. MULTI-QUANTILE OBJECTIVE. One XGBoost model predicts the 10th, 50th and
   90th percentiles of price simultaneously (`reg:quantileerror` with a vector
   `quantile_alpha`). The 50th is the headline estimate; the 10th-90th band is
   an honest "we're 80% sure it's in here" interval. A used car genuinely
   doesn't have one true price, so a point estimate alone is dishonest.

2. LOG TARGET. Prices are right-skewed and multiplicative (a dent costs a %,
   not a fixed sum). We model log(price). Quantiles are preserved under the
   monotone exp() transform, so exp(q50 of log-price) is exactly the q50 of
   price - the transform is mathematically clean for intervals.

3. CALIBRATION CHECK. We verify empirically that ~80% of held-out cars really
   fall inside the predicted 80% band. An interval nobody validated is just
   decoration.

Run:  python train_model.py
"""

from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split

import feature_engineering as fe
from generate_dataset import generate

QUANTILES = np.array([0.10, 0.50, 0.90])
MEDIAN_IDX = 1  # index of 0.50 inside QUANTILES
ARTIFACT_PATH = "model.joblib"


def _metrics(y_true_price, y_pred_price):
    err = y_pred_price - y_true_price
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / y_true_price) * 100)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true_price - y_true_price.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    return {"MAE_lakh": mae, "RMSE_lakh": rmse, "MAPE_pct": mape, "R2": r2}


def train(df: pd.DataFrame | None = None, verbose: bool = True):
    if df is None:
        df = generate()

    X = fe.build_features(df)
    y_price = df[fe.TARGET].to_numpy()
    y = np.log(y_price)  # model in log space

    # Three-way split: train / val (kept for honesty, used to sanity-check) / test
    X_tr, X_tmp, y_tr, y_tmp, p_tr, p_tmp = train_test_split(
        X, y, y_price, test_size=0.30, random_state=42
    )
    X_val, X_te, y_val, y_te, p_val, p_te = train_test_split(
        X_tmp, y_tmp, p_tmp, test_size=0.50, random_state=42
    )

    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=QUANTILES,
        n_estimators=700,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=2.0,
        reg_alpha=0.1,
        tree_method="hist",
        enable_categorical=True,   # native categorical: no manual one-hot
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # ---- Evaluate on the untouched test set --------------------------------
    pred_log = np.asarray(model.predict(X_te))      # (n, 3): q10, q50, q90
    if pred_log.ndim == 1:                           # safety for odd versions
        pred_log = pred_log[:, None].repeat(3, axis=1)
    pred_price = np.exp(pred_log)
    # Quantile crossing guard: sort so q10<=q50<=q90 row-wise.
    pred_price = np.sort(pred_price, axis=1)

    q10, q50, q90 = pred_price[:, 0], pred_price[:, MEDIAN_IDX], pred_price[:, 2]
    metrics = _metrics(p_te, q50)

    # Interval calibration: target coverage is 0.80 for a [q10, q90] band.
    coverage = float(np.mean((p_te >= q10) & (p_te <= q90)))
    avg_band_width = float(np.mean(q90 - q10))
    metrics["interval_coverage_80"] = coverage
    metrics["avg_band_width_lakh"] = avg_band_width

    # Feature importance (gain) for the report.
    booster = model.get_booster()
    importance = booster.get_score(importance_type="gain")

    meta = {
        "quantiles": QUANTILES.tolist(),
        "median_idx": MEDIAN_IDX,
        "feature_columns": fe.FEATURE_COLUMNS,
        "metrics": metrics,
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "importance_gain": importance,
        # Reference sample powers the "market comparables" panel in the app.
        "reference": df.sample(min(3000, len(df)), random_state=7).reset_index(drop=True),
    }

    if verbose:
        print(f"Trained on {len(X_tr):,} cars | tested on {len(X_te):,}")
        print("-" * 46)
        print(f"  MAE   : {metrics['MAE_lakh']:.2f} lakh")
        print(f"  RMSE  : {metrics['RMSE_lakh']:.2f} lakh")
        print(f"  MAPE  : {metrics['MAPE_pct']:.1f}%")
        print(f"  R2    : {metrics['R2']:.3f}")
        print(f"  80% interval coverage : {coverage*100:.1f}%  (target ~80%)")
        print(f"  avg band width        : {avg_band_width:.2f} lakh")
        print("-" * 46)
        top = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:6]
        print("Top features by gain:")
        for k, v in top:
            print(f"    {k:18s} {v:10.1f}")

    return model, meta


def main():
    model, meta = train()
    joblib.dump({"model": model, "meta": meta}, ARTIFACT_PATH, compress=3)
    print(f"\nSaved -> {ARTIFACT_PATH}")
    # Also drop metrics as JSON so they're easy to paste into a README / resume.
    with open("metrics.json", "w") as f:
        json.dump(meta["metrics"], f, indent=2)
    print("Saved -> metrics.json")


if __name__ == "__main__":
    main()
