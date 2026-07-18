import streamlit as st

from core_pipeline.data_loader import load_online_retail
from core_pipeline.helpers import dataset_summary

st.title("📈 Forecasting")
st.caption("Time Series Analysis • Sales Forecasting • Business Predictions")

st.divider()

st.markdown("""
This module predicts future business trends using machine learning and
time series forecasting techniques.
""")

# -----------------------------------
# Dashboard Summary
# -----------------------------------

df = load_online_retail()
summary = dataset_summary(df) if df is not None else None

col1, col2, col3, col4 = st.columns(4)

if summary:
    with col1:
        st.metric("Rows", f"{summary['rows']:,}")

    with col2:
        st.metric("Columns", summary["columns"])

    with col3:
        st.metric("Missing Values", f"{summary['missing_values']:,}")

    with col4:
        st.metric("Duplicate Rows", f"{summary['duplicate_rows']:,}")
else:
    with col1:
        st.metric("Rows", "--")

    with col2:
        st.metric("Columns", "--")

    with col3:
        st.metric("Missing Values", "--")

    with col4:
        st.metric("Duplicate Rows", "--")

st.divider()

# -----------------------------------
# Dataset Preview
# -----------------------------------

st.subheader("📁 Dataset Preview")

if df is not None:
    st.success("Online Retail dataset loaded successfully!")
    st.dataframe(df.head(10), use_container_width=True)
else:
    st.error("Dataset could not be loaded.")

st.divider()

# -----------------------------------
# Time Series Analysis
# -----------------------------------

st.subheader("📊 Time Series Analysis")

left, right = st.columns(2)

with left:
    st.info("Historical Sales Trend")

with right:
    st.info("Seasonality & Trend Analysis")

st.divider()

# -----------------------------------
# Forecast Results
# -----------------------------------

st.subheader("🔮 Forecast Results")

st.warning("Waiting for Member 3 integration.")

st.divider()

# -----------------------------------
# Visualizations
# -----------------------------------

st.subheader("📈 Forecast Visualizations")

chart1, chart2 = st.columns(2)

with chart1:
    st.empty()

with chart2:
    st.empty()

st.info("Forecast charts and prediction graphs will appear here.")