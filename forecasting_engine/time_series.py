

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from statsmodels.tsa.arima.model import ARIMA

from prophet import Prophet

# -----------------------------

DATA_PATH = "datasets/online_retail.csv"

GRAPH_DIR = "outputs/graphs"

METRIC_DIR = "outputs/metrics"

PREDICTION_DIR = "outputs/predictions"

os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------

def load_dataset():

    print("Loading Dataset...")

    df = pd.read_csv(DATA_PATH)

    return df


# -----------------------------
# Clean Dataset
# -----------------------------

def clean_dataset(df):

    df = df.dropna(subset=["CustomerID"])

    df = df[df["Quantity"] > 0]

    df = df[df["UnitPrice"] > 0]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["Amount"] = df["Quantity"] * df["UnitPrice"]

    return df

# -----------------------------
# Daily Sales
# -----------------------------

def create_daily_sales(df):

    daily_sales = (

        df.groupby(df["InvoiceDate"].dt.date)["Amount"]

        .sum()

        .reset_index()

    )

    daily_sales.columns = [

        "Date",

        "Sales"

    ]

    daily_sales["Date"] = pd.to_datetime(

        daily_sales["Date"]

    )

    return daily_sales


# -----------------------------
# Trend Analysis
# -----------------------------

def plot_trend(daily_sales):

    plt.figure(figsize=(14,6))

    plt.plot(

        daily_sales["Date"],

        daily_sales["Sales"]

    )

    plt.title("Sales Trend")

    plt.xlabel("Date")

    plt.ylabel("Sales")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            GRAPH_DIR,

            "trend_analysis.png"

        )

    )

    plt.close()


# -----------------------------
# Rolling Statistics
# -----------------------------

def rolling_statistics(daily_sales):

    daily_sales["RollingMean"] = (

        daily_sales["Sales"]

        .rolling(30)

        .mean()

    )

    daily_sales["RollingStd"] = (

        daily_sales["Sales"]

        .rolling(30)

        .std()

    )

    plt.figure(figsize=(14,6))

    plt.plot(

        daily_sales["Date"],

        daily_sales["Sales"],

        label="Sales"

    )

    plt.plot(

        daily_sales["Date"],

        daily_sales["RollingMean"],

        label="Rolling Mean"

    )

    plt.plot(

        daily_sales["Date"],

        daily_sales["RollingStd"],

        label="Rolling Std"

    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            GRAPH_DIR,

            "rolling_statistics.png"

        )

    )

    plt.close()

    return daily_sales

# -----------------------------
# Seasonality
# -----------------------------

def seasonality_analysis(daily_sales):

    monthly = (

        daily_sales

        .set_index("Date")

        .resample("M")

        .sum()

    )

    plt.figure(figsize=(12,5))

    plt.plot(

        monthly.index,

        monthly["Sales"]

    )

    plt.title("Monthly Seasonality")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            GRAPH_DIR,

            "seasonality.png"

        )

    )

    plt.close()


# -----------------------------
# ARIMA Forecast
# -----------------------------

def arima_forecast(daily_sales):

    model = ARIMA(

        daily_sales["Sales"],

        order=(5,1,0)

    )

    model_fit = model.fit()

    forecast = model_fit.forecast(

        steps=30

    )

    return forecast


# -----------------------------
# Prophet Forecast
# -----------------------------

def prophet_forecast(daily_sales):

    prophet_df = daily_sales.rename(
        columns={
            "Date": "ds",
            "Sales": "y"
        }
    )

    model = Prophet()

    model.fit(prophet_df)

    future = model.make_future_dataframe(
        periods=30
    )

    forecast = model.predict(future)

    return forecast


# -----------------------------
# Forecast Comparison
# -----------------------------

def compare_forecasts(daily_sales, arima_forecast_values):

    actual = daily_sales["Sales"].tail(len(arima_forecast_values)).values

    predictions = arima_forecast_values.values

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )

    mape = np.mean(
        np.abs(
            (actual - predictions) / actual
        )
    ) * 100

    metrics = pd.DataFrame({

        "Metric": [
            "MAE",
            "RMSE",
            "MAPE"
        ],

        "Value": [
            mae,
            rmse,
            mape
        ]

    })

    metrics.to_csv(

        os.path.join(

            METRIC_DIR,

            "forecast_metrics.csv"

        ),

        index=False

    )

    print(metrics)


# -----------------------------
# Forecast Visualization
# -----------------------------

def plot_forecast(

    daily_sales,

    arima_forecast_values,

    prophet_forecast_values

):

    plt.figure(figsize=(15,6))

    plt.plot(

        daily_sales["Date"],

        daily_sales["Sales"],

        label="Actual Sales"

    )

    future_dates = pd.date_range(

        start=daily_sales["Date"].max(),

        periods=30,

        freq="D"

    )

    plt.plot(

        future_dates,

        arima_forecast_values,

        label="ARIMA Forecast"

    )

    plt.plot(

        prophet_forecast_values["ds"],

        prophet_forecast_values["yhat"],

        label="Prophet Forecast"

    )

    plt.legend()

    plt.title("Forecast Comparison")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            GRAPH_DIR,

            "forecast_comparison.png"

        )

    )

    plt.close()


# -----------------------------
# Save Forecast
# -----------------------------

def save_forecast(

    arima_forecast_values,

    prophet_forecast_values

):

    arima_df = pd.DataFrame({

        "ARIMA Forecast": arima_forecast_values

    })

    prophet_df = prophet_forecast_values[
        [
            "ds",
            "yhat"
        ]
    ]

    arima_df.to_csv(

        os.path.join(

            PREDICTION_DIR,

            "arima_forecast.csv"

        ),

        index=False

    )

    prophet_df.to_csv(

        os.path.join(

            PREDICTION_DIR,

            "prophet_forecast.csv"

        ),

        index=False

    )


# -----------------------------
# Main
# -----------------------------

def main():

    print("=" * 60)
    print("Time Series Forecasting")
    print("=" * 60)

    df = load_dataset()

    df = clean_dataset(df)

    daily_sales = create_daily_sales(df)

    plot_trend(daily_sales)

    daily_sales = rolling_statistics(daily_sales)

    seasonality_analysis(daily_sales)

    print("\nRunning ARIMA...")

    arima_result = arima_forecast(daily_sales)

    print("Running Prophet...")

    prophet_result = prophet_forecast(daily_sales)

    compare_forecasts(

        daily_sales,

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

    print("\nTime Series Completed Successfully.")

    print("Forecast CSV files saved.")

    print("Metrics saved.")

    print("Graphs saved.")


if __name__ == "__main__":

    main()