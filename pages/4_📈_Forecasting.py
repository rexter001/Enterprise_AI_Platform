import streamlit as st
import pandas as pd
import inspect

from forecasting_engine.time_series import (
    clean_dataset,
    create_daily_sales,
    plot_trend,
    rolling_statistics,
    seasonality_analysis,
    arima_forecast,
    prophet_forecast,
    compare_forecasts,
    plot_forecast,
    save_forecast,
)

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

left, right = st.columns(2)

st.subheader("📊 Time Series Analysis")

if "InvoiceDate" in df.columns:
    st.metric(
        "Date Range",
        f"{df['InvoiceDate'].min()} → {df['InvoiceDate'].max()}"
    )

c1, c2 = st.columns(2)

with c1:
    st.bar_chart(df.groupby("Country")["Quantity"].sum().head(10))

with c2:
    st.bar_chart(df.groupby("Country")["UnitPrice"].mean().head(10))

st.divider()

# -----------------------------------
# Forecast Results
# -----------------------------------

st.subheader("🔮 Forecasting")

forecast = st.button(
    "🚀 Run Forecasting",
    use_container_width=True
)

if forecast:

    df = clean_dataset(df)

    daily_sales = create_daily_sales(df)

    st.success("Dataset prepared successfully!")

    plot_trend(daily_sales)

    daily_sales = rolling_statistics(daily_sales)

    seasonality_analysis(daily_sales)

    arima_result, test_data = arima_forecast(daily_sales)

    prophet_result = prophet_forecast(daily_sales)

    compare_forecasts(
        test_data,
        arima_result
    )

    plot_forecast(
        daily_sales,
        arima_result,
        prophet_result
    )

    save_forecast(
        arima_result,
        prophet_result
    )

    st.success("Forecast completed successfully!")

    st.divider()

    # -----------------------------------
    # Visualizations
    # -----------------------------------

    st.subheader("📈 Forecast Visualizations")

    c1, c2 = st.columns(2)

    with c1:
        st.image(
            "outputs/graphs/trend_analysis.png",
            caption="Sales Trend",
            use_container_width=True
        )

    with c2:
        st.image(
            "outputs/graphs/rolling_statistics.png",
            caption="Rolling Statistics",
            use_container_width=True
        )

    c3, c4 = st.columns(2)

    with c3:
        st.image(
            "outputs/graphs/seasonality.png",
            caption="Seasonality",
            use_container_width=True
        )

    with c4:
        st.image(
            "outputs/graphs/forecast_comparison.png",
            caption="Forecast Comparison",
            use_container_width=True
        )

    metrics = pd.read_csv("outputs/metrics/forecast_metrics.csv")

    st.subheader("📊 Forecast Metrics")

    st.dataframe(
        metrics,
        use_container_width=True
    )

    st.subheader("📄 Forecast Results")

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(
            pd.read_csv("outputs/predictions/arima_forecast.csv"),
            use_container_width=True
        )

    with col2:
        st.dataframe(
            pd.read_csv("outputs/predictions/prophet_forecast.csv"),
            use_container_width=True
        )