# 🚗 Used-Car Valuation Engine

An XGBoost used-car price model with a Streamlit front-end. Unlike most "predict
a number" demos, this one predicts an **honest price range**, **explains every
estimate**, and **anchors it to comparable listings**.

**Run it:**
```bash
pip install -r requirements.txt
streamlit run app.py        # trains the model automatically on first launch
```
That's it — the app is self-bootstrapping. On first run it generates the dataset
and trains the model (≈20–40s), caches the artifact, and serves. To train ahead
of time and see the evaluation report:
```bash
python train_model.py
```

---

## What makes this different from a typical price-predictor

Most student projects fit one regressor and print a single number. This project
is built like a small real valuation system:

**1. It predicts a price *band*, not a false-precision point.**
A single XGBoost model is trained with the multi-quantile objective
(`reg:quantileerror` with `quantile_alpha = [0.1, 0.5, 0.9]`) and predicts the
10th, 50th and 90th price percentiles in one shot. A used car doesn't have one
true price, so the output is *"best estimate ₹X, 80% sure it's ₹A–₹B"*.

**2. The intervals are calibrated and checked.**
On held-out cars we verify that ≈80% really fall inside the predicted 80% band.
An interval nobody validated is just decoration — this one is measured
(`interval_coverage_80` in the metrics).

**3. Every estimate is explained — in price terms.**
The target is `log(price)` (prices are multiplicative and right-skewed). We pull
XGBoost's native **TreeSHAP** contributions (no extra `shap` dependency) and,
because a log-space contribution `c` is a price multiplier `exp(c)`, convert them
into intuitive effects: *"Automatic → +7%", "3 owners → −13%"*. This is the
mathematically correct way to read a log-target model.

**4. Estimates are anchored to real comparables.**
The app pulls similar listings (same brand, similar age) from a reference sample
and tells you whether the car is priced below / at / above that local market.

**5. Interactive depreciation explorer.**
Sweep mileage or age and watch the model's learned price curve (with its
uncertainty band) for *your exact configuration* — a quick way to sanity-check
that the model behaves like an appraiser, not a black box.

**6. No train/serve skew.**
All feature engineering lives in one shared module (`feature_engineering.py`)
used by both training and the app, and categorical levels are pinned with a
fixed `CategoricalDtype` so XGBoost's native categorical codes are identical at
train and inference time. (This is also where a subtle real bug was fixed: a
`"None"` category silently parsed back from CSV as `NaN`.)

**7. The data has known economic structure.**
The synthetic generator (`generate_dataset.py`) prices cars with explicit
depreciation curves, brand-tier premiums, mileage wear, owner/accident penalties
and lognormal market noise. Because the ground truth is known, you can confirm
the model *recovered the right relationships* (age and mileage dominate feature
importance; the depreciation curve bends the right way), not just that error is
low.

---

## Project structure
```
feature_engineering.py   shared transforms + pinned categorical levels
generate_dataset.py      realistic synthetic market (prices in lakh INR)
train_model.py           multi-quantile XGBoost + calibration + metrics
model_service.py         inference: interval, TreeSHAP explain, comparables, sweeps
app.py                   Streamlit UI (self-bootstrapping)
requirements.txt
```

## Expected performance
Run `python train_model.py` to print the report. With native categorical
features and the full tree ensemble, typical held-out numbers land around
**MAPE ~8–12%, R² ~0.94+**, with **80% band coverage near 80%**. Exact figures
print on training and are written to `metrics.json`.

## Deploying (Streamlit Community Cloud)
Push this folder to a GitHub repo and point Streamlit Cloud at `app.py`. The
self-bootstrapping training runs once on the first cold start.

> Estimates are model output for demonstration and not a guaranteed valuation.
