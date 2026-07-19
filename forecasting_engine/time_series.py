

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

    df.groupby(pd.Grouper(key="InvoiceDate", freq="D"))["Amount"]
      .sum()
      .reset_index()

    )

    daily_sales.columns = ["Date", "Sales"]

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

    daily_sales = daily_sales.copy()

    daily_sales["Date"] = pd.to_datetime(daily_sales["Date"])

    # Group by year and month instead of using resample
    monthly = (
        daily_sales
        .groupby(
            daily_sales["Date"].dt.to_period("M")
        )["Sales"]
        .sum()
        .reset_index()
    )

    # Convert Period -> Timestamp for plotting
    monthly["Date"] = monthly["Date"].dt.to_timestamp()

    plt.figure(figsize=(12, 5))

    plt.plot(monthly["Date"], monthly["Sales"], marker="o")

    plt.title("Monthly Seasonality")
    plt.xlabel("Month")
    plt.ylabel("Sales")

    plt.tight_layout()

    plt.savefig(
        os.path.join(GRAPH_DIR, "seasonality.png")
    )

    plt.close()


# -----------------------------
# ARIMA Forecast
# -----------------------------

def arima_forecast(daily_sales):

    sales = daily_sales["Sales"]

    train = sales[:-30]

    test = sales[-30:]

    model = ARIMA(
        train,
        order=(5,1,0)
    )

    model_fit = model.fit()

    forecast = model_fit.forecast(
        steps=30
    )

    return forecast, test


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

def compare_forecasts(actual, predictions):

    mae = mean_absolute_error(actual, predictions)

    rmse = np.sqrt(
        mean_squared_error(actual, predictions)
    )

    # Avoid division by zero
    actual = np.array(actual)
    predictions = np.array(predictions)

    mask = actual != 0

    mape = (
        np.mean(
            np.abs(
                (actual[mask] - predictions[mask]) / actual[mask]
            )
        ) * 100
    )

    metrics = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "MAPE"],
        "Value": [mae, rmse, mape]
    })

    metrics.to_csv(
        os.path.join(
            METRIC_DIR,
            "forecast_metrics.csv"
        ),
        index=False
    )

    return metrics


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

    arima_result, test_data = arima_forecast(daily_sales)

    print("Running Prophet...")

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

    print("\nTime Series Completed Successfully.")

    print("Forecast CSV files saved.")

    print("Metrics saved.")

    print("Graphs saved.")


if __name__ == "__main__":

    main()