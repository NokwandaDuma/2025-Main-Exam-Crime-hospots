# Save this entire cell content as app.py and run: streamlit run app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# (re-use the same loading / model code or load saved artefacts)

st.set_page_config(page_title="SA Crime Analytics Dashboard", layout="wide")
st.title("Data-driven Crime Analytics – South Africa")
st.markdown("Classification of hotspots + time-series forecasts + EDA")

# Sidebar filters
province = st.sidebar.selectbox("Province", ["All"] + sorted(df["Province"].unique().tolist()))
category = st.sidebar.selectbox("Crime Category", sorted(df["Category"].unique().tolist()))
year_range = st.sidebar.slider("Year range", int(df["Year"].min()), int(df["Year"].max()), (2010, 2015))

# Filtered data + plots
# ... (add EDA charts, confusion matrix, forecast plot with confidence intervals if you compute them, summary metrics)

st.subheader("Hotspot Classification Metrics")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
# display confusion matrix image or table

st.subheader("Forecast Preview")
# show the ARIMA forecast plot

st.success("Dashboard ready – filters update visuals in real time.")