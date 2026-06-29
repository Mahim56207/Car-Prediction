"""
app.py  -  Used-Car Valuation Engine (Streamlit front-end)
==========================================================
Run:  streamlit run app.py

The app is self-bootstrapping: on first launch, if `model.joblib` is missing it
generates the dataset and trains the model (cached), so a fresh deployment works
with a single command and no manual setup step.

The interesting parts live in model_service.py / train_model.py - this file is
mostly presentation, deliberately. What it surfaces that a typical demo doesn't:
  * an honest 80% price BAND, not a single number
  * a per-prediction explanation in plain "+/- % on price" terms (TreeSHAP)
  * anchoring to real comparable listings
  * an interactive depreciation curve for the exact car you configured
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import feature_engineering as fe
import model_service as ms

st.set_page_config(page_title="Used-Car Valuation Engine", page_icon="🚗",
                   layout="wide")

# ----- light cosmetic polish (the substance is in the model) ---------------
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
div[data-testid="stMetricValue"] {font-size: 2.4rem;}
.small {color:#6b7280; font-size:0.85rem;}
.tag {display:inline-block; padding:2px 10px; border-radius:999px;
      font-size:0.78rem; font-weight:600;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparing the valuation model…")
def get_model():
    if not os.path.exists("model.joblib"):
        from train_model import train
        import joblib
        model, meta = train(verbose=False)
        joblib.dump({"model": model, "meta": meta}, "model.joblib", compress=3)
        return model, meta
    return ms.load_artifact()


model, meta = get_model()
m = meta["metrics"]

st.title("🚗 Used-Car Valuation Engine")
st.markdown(
    "<span class='small'>XGBoost quantile model · predicts an honest price "
    "<b>band</b>, explains every estimate, and anchors it to comparable "
    "listings.</span>", unsafe_allow_html=True)

# ===========================================================================
# INPUTS
# ===========================================================================
left, right = st.columns([1, 1.25], gap="large")

with left:
    st.subheader("Car details")
    c1, c2 = st.columns(2)
    brand = c1.selectbox("Brand", fe.CATEGORY_LEVELS["brand"], index=1)
    year = c2.number_input("Manufacture year", 2008, fe.CURRENT_YEAR, 2019)
    mileage = c1.number_input("Mileage (km)", 0, 300_000, 55_000, step=1000)
    engine = c2.number_input("Engine (cc)", 600, 4000, 1200, step=100)
    fuel = c1.selectbox("Fuel", fe.CATEGORY_LEVELS["fuel_type"])
    trans = c2.selectbox("Transmission", fe.CATEGORY_LEVELS["transmission"])
    owners = c1.number_input("Previous owners", 1, 5, 1)
    city = c2.selectbox("City tier", fe.CATEGORY_LEVELS["city_tier"])
    service = c1.selectbox("Service record", fe.CATEGORY_LEVELS["service_history"])
    accident = c2.selectbox("Accident history", fe.CATEGORY_LEVELS["accident_history"])

car = dict(brand=brand, year=int(year), mileage_km=int(mileage),
           fuel_type=fuel, transmission=trans, engine_cc=int(engine),
           owner_count=int(owners), accident_history=accident,
           service_history=service, city_tier=city)

band = ms.predict_interval(model, meta, car)
comp = ms.comparables(meta, car)

# ===========================================================================
# RESULT + INTERVAL
# ===========================================================================
with right:
    st.subheader("Estimated value")
    st.metric("Best estimate", f"₹ {band['mid']:.2f} lakh",
              help="Median (50th percentile) of the predicted price distribution.")
    st.markdown(
        f"<span class='small'>80% confidence band: "
        f"<b>₹{band['low']:.2f}L  –  ₹{band['high']:.2f}L</b></span>",
        unsafe_allow_html=True)

    # interval visual
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[band["low"], band["high"]], y=[1, 1], mode="lines",
        line=dict(color="#94a3b8", width=10), showlegend=False,
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[band["low"], band["mid"], band["high"]], y=[1, 1, 1], mode="markers+text",
        marker=dict(size=[12, 20, 12], color=["#94a3b8", "#2563eb", "#94a3b8"]),
        text=[f"₹{band['low']:.1f}L", f"₹{band['mid']:.2f}L", f"₹{band['high']:.1f}L"],
        textposition=["bottom center", "top center", "bottom center"],
        showlegend=False, hoverinfo="skip"))
    if comp:
        fig.add_vrect(x0=comp["q25"], x1=comp["q75"], fillcolor="#10b981",
                      opacity=0.10, line_width=0,
                      annotation_text="similar listings (mid 50%)",
                      annotation_position="top left",
                      annotation_font_size=11)
    fig.update_layout(height=170, margin=dict(l=10, r=10, t=20, b=10),
                      yaxis=dict(visible=False, range=[0.5, 1.6]),
                      xaxis=dict(title="₹ lakh", showgrid=True, gridcolor="#eef2f7"),
                      plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # market positioning
    if comp:
        diff = (band["mid"] - comp["median"]) / comp["median"] * 100
        if diff < -7:
            tag, color, msg = "BELOW MARKET", "#065f46", f"~{abs(diff):.0f}% under the typical asking price for similar cars — a strong buy."
        elif diff > 7:
            tag, color, msg = "ABOVE MARKET", "#991b1b", f"~{diff:.0f}% above similar listings — priced at a premium."
        else:
            tag, color, msg = "FAIR MARKET", "#1e3a8a", "in line with comparable listings."
        st.markdown(
            f"<span class='tag' style='background:{color}1a;color:{color};'>{tag}</span> "
            f"<span class='small'>Based on <b>{comp['n']}</b> comparable cars "
            f"(median ₹{comp['median']:.2f}L), this estimate is {msg}</span>",
            unsafe_allow_html=True)

st.divider()

# ===========================================================================
# EXPLANATION + DEPRECIATION EXPLORER
# ===========================================================================
ex_col, sens_col = st.columns([1, 1], gap="large")

with ex_col:
    st.subheader("Why this price?")
    st.caption("Each factor's multiplicative effect on the estimate (TreeSHAP, "
               "read in price terms).")
    drivers = ms.explain(model, meta, car, top_k=7)
    labels = [f"{d['feature']} ({d['raw_value']})" for d in drivers][::-1]
    pcts = [d["pct"] for d in drivers][::-1]
    colors = ["#16a34a" if p >= 0 else "#dc2626" for p in pcts]
    bar = go.Figure(go.Bar(
        x=pcts, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{p:+.1f}%" for p in pcts], textposition="outside"))
    bar.update_layout(height=330, margin=dict(l=10, r=30, t=10, b=10),
                      xaxis=dict(title="effect on price (%)", zeroline=True,
                                 zerolinecolor="#9ca3af", gridcolor="#eef2f7"),
                      plot_bgcolor="white")
    st.plotly_chart(bar, use_container_width=True)

with sens_col:
    st.subheader("Depreciation explorer")
    axis = st.radio("Sweep", ["Mileage", "Age"], horizontal=True,
                    label_visibility="collapsed")
    if axis == "Mileage":
        grid = np.arange(5_000, 200_001, 10_000)
        df_s = ms.sensitivity_curve(model, meta, car, "mileage_km", grid)
        xcol, xlab, cur = "mileage_km", "mileage (km)", car["mileage_km"]
    else:
        years = np.arange(fe.CURRENT_YEAR - 17, fe.CURRENT_YEAR + 1)
        df_s = ms.sensitivity_curve(model, meta, car, "year", years)
        df_s["age"] = fe.CURRENT_YEAR - df_s["year"]
        xcol, xlab, cur = "age", "age (years)", fe.CURRENT_YEAR - car["year"]

    sc = go.Figure()
    sc.add_trace(go.Scatter(x=df_s[xcol], y=df_s["high"], mode="lines",
                            line=dict(width=0), showlegend=False, hoverinfo="skip"))
    sc.add_trace(go.Scatter(x=df_s[xcol], y=df_s["low"], mode="lines", fill="tonexty",
                            fillcolor="rgba(37,99,235,0.12)", line=dict(width=0),
                            name="80% band"))
    sc.add_trace(go.Scatter(x=df_s[xcol], y=df_s["mid"], mode="lines",
                            line=dict(color="#2563eb", width=3), name="estimate"))
    sc.add_vline(x=cur, line_dash="dot", line_color="#6b7280")
    sc.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10),
                     xaxis=dict(title=xlab, gridcolor="#eef2f7"),
                     yaxis=dict(title="₹ lakh", gridcolor="#eef2f7"),
                     plot_bgcolor="white", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(sc, use_container_width=True)
    st.caption("Dotted line = your car. The curve is the model's learned "
               "price behaviour for *this* configuration.")

# ===========================================================================
# MODEL TRANSPARENCY
# ===========================================================================
with st.expander("📊 Model quality & how it works"):
    a, b, c, d = st.columns(4)
    a.metric("Test MAPE", f"{m['MAPE_pct']:.1f}%")
    b.metric("Test R²", f"{m['R2']:.3f}")
    c.metric("MAE", f"₹{m['MAE_lakh']:.2f}L")
    d.metric("80% band coverage", f"{m['interval_coverage_80']*100:.0f}%")
    st.markdown(f"""
**Approach.** A single XGBoost model trained with the multi-quantile objective
(`reg:quantileerror`) predicts the 10th / 50th / 90th price percentiles at once.
The target is `log(price)` because used-car prices are multiplicative and
right-skewed; quantiles survive the `exp()` transform unchanged.

**Calibration.** On held-out cars, **{m['interval_coverage_80']*100:.0f}%** actually
fall inside the predicted 80% band (target 80%) — the interval is honest, not
cosmetic.

**Explanations** use XGBoost's native TreeSHAP contributions, converted from
log space into the multiplicative price effects shown above. Feature engineering
runs through one shared module for both training and inference, so there's no
train/serve skew.

*Trained {meta['trained_at']} on {meta['n_train']:,} cars · evaluated on
{meta['n_test']:,}.*
""")

st.markdown("<span class='small'>Estimates are model output for demonstration, "
            "not a guaranteed valuation.</span>", unsafe_allow_html=True)
