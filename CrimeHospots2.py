import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMAResults
import os

st.set_page_config(page_title="SA Crime Analytics", layout="wide")

st.title("Data-driven Crime Analytics – South Africa")
st.markdown("Classification of hotspots + time-series forecasts + EDA")

# =========================================================
# 1. LOAD DATA (must be done inside the app)
# =========================================================
@st.cache_data
def load_data():
    # Make sure these two files are in the same folder as this script
    # (or in a data/ folder and update the paths)
    df_crime = pd.read_csv("SouthAfricaCrimeStats_v2.csv")
    df_stations = pd.read_csv("police_stations.csv")

    # Melt to long format
    year_cols = [c for c in df_crime.columns if c.startswith("20")]
    df_long = df_crime.melt(
        id_vars=["Province", "Station", "Category"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Incidents"
    )
    df_long["Year"] = df_long["Year"].str[:4].astype(int)

    # Clean names for merging
    df_stations["name_clean"] = df_stations["name"].str.lower().str.strip()
    df_long["Station_clean"] = df_long["Station"].str.lower().str.strip()

    df = df_long.merge(
        df_stations[["name_clean", "population", "area_km2", "prov_code"]],
        left_on="Station_clean",
        right_on="name_clean",
        how="left"
    )
    return df

df = load_data()

# =========================================================
# 2. LOAD SAVED MODELS
# =========================================================
@st.cache_resource
def load_models():
    rf = joblib.load("hotspot_classifier.pkl")
    arima = ARIMAResults.load("arima_forecast_model.pkl")
    meta = joblib.load("model_metadata.pkl")
    return rf, arima, meta

try:
    rf, arima_model, meta = load_models()
    selected_category = meta.get("selected_category", "Murder")
except Exception as e:
    st.error(f"Could not load models. Make sure the .pkl files are in the root of the repository. Error: {e}")
    st.stop()

# =========================================================
# 3. SIDEBAR FILTERS
# =========================================================
st.sidebar.header("Filters")

provinces = ["All"] + sorted(df["Province"].dropna().unique().tolist())
province = st.sidebar.selectbox("Province", provinces)

categories = sorted(df["Category"].dropna().unique().tolist())
category = st.sidebar.selectbox("Crime Category", categories, 
                                index=categories.index(selected_category) if selected_category in categories else 0)

min_year = int(df["Year"].min())
max_year = int(df["Year"].max())
year_range = st.sidebar.slider("Year range", min_year, max_year, (min_year, max_year))

# =========================================================
# 4. FILTERED DATA
# =========================================================
filtered = df.copy()
if province != "All":
    filtered = filtered[filtered["Province"] == province]
filtered = filtered[filtered["Category"] == category]
filtered = filtered[(filtered["Year"] >= year_range[0]) & (filtered["Year"] <= year_range[1])]

# =========================================================
# 5. MAIN CONTENT
# =========================================================
col1, col2, col3 = st.columns(3)
col1.metric("Total Incidents", f"{filtered['Incidents'].sum():,.0f}")
col2.metric("Number of Stations", filtered["Station"].nunique())
col3.metric("Years Covered", f"{year_range[0]} – {year_range[1]}")

st.subheader("Exploratory Data Analysis")

# Trend over time
yearly = filtered.groupby("Year")["Incidents"].sum()
fig1, ax1 = plt.subplots(figsize=(10, 4))
yearly.plot(ax=ax1, marker="o")
ax1.set_title(f"{category} – Trend over time")
ax1.set_ylabel("Incidents")
st.pyplot(fig1)

# Top stations
st.subheader("Top 10 Stations by Incidents")
top_stations = (filtered.groupby("Station")["Incidents"]
                .sum()
                .nlargest(10)
                .sort_values())
fig2, ax2 = plt.subplots(figsize=(10, 5))
top_stations.plot(kind="barh", ax=ax2, color="steelblue")
ax2.set_xlabel("Total Incidents")
st.pyplot(fig2)

# =========================================================
# 6. CLASSIFICATION SECTION
# =========================================================
st.subheader("Hotspot Classification Results")
st.info("Model: Random Forest Classifier (top 25% stations = Hotspot)")

# Simple demo prediction (you can expand this)
st.write("Feature importance from the trained model:")
try:
    importances = pd.Series(rf.feature_importances_, index=meta.get("feature_columns", ["Incidents", "population", "density"]))
    st.bar_chart(importances)
except:
    st.write("Feature importances not available")

# =========================================================
# 7. FORECAST SECTION
# =========================================================
st.subheader(f"Time Series Forecast – {selected_category}")

try:
    # Show historical + forecast
    forecast = arima_model.get_forecast(steps=24)
    pred_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    # You can also plot historical data here if you want
    pred_mean.plot(ax=ax3, label="Forecast", color="red")
    ax3.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                     color="pink", alpha=0.3, label="95% CI")
    ax3.set_title("24-period Ahead Forecast with Confidence Intervals")
    ax3.legend()
    st.pyplot(fig3)

    st.write("Forecast values:")
    st.dataframe(pred_mean)
except Exception as e:
    st.warning(f"Could not generate forecast plot: {e}")

st.success("Dashboard loaded successfully!")
