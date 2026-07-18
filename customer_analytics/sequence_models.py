

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.ensemble import IsolationForest

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    GRU,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------

DATA_PATH = "datasets/online_retail.csv"

MODEL_DIR = "models"
GRAPH_DIR = "outputs/graphs"
METRIC_DIR = "outputs/metrics"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------

def load_dataset():

    print("Loading Dataset...")

    df = pd.read_csv(DATA_PATH)

    return df


# -----------------------------
# Cleaning
# -----------------------------

def clean_dataset(df):

    df = df.dropna(subset=["CustomerID"])

    df = df[df["Quantity"] > 0]

    df = df[df["UnitPrice"] > 0]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["CustomerID"] = df["CustomerID"].astype(int)

    df["Amount"] = df["Quantity"] * df["UnitPrice"]

    return df

# -----------------------------
# Feature Engineering (RFM)
# -----------------------------

def feature_engineering(df):

    print("Generating Customer Features...")

    latest_date = df["InvoiceDate"].max()

    customer_df = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate",
                 lambda x: (latest_date - x.max()).days),

        Frequency=("InvoiceNo", "nunique"),

        Monetary=("Amount", "sum"),

        AvgQuantity=("Quantity", "mean"),

        AvgUnitPrice=("UnitPrice", "mean")
    ).reset_index()

    return customer_df


# -----------------------------
# Create Risk Label
# -----------------------------

def create_target(customer_df):

    """
    Create binary risk label.

    Low Monetary customers
    are considered High Risk.
    """

    threshold = customer_df["Monetary"].median()

    customer_df["Risk"] = np.where(
        customer_df["Monetary"] < threshold,
        1,
        0
    )

    return customer_df


# -----------------------------
# Scale Features
# -----------------------------

def preprocess(customer_df):

    feature_cols = [
        "Recency",
        "Frequency",
        "Monetary",
        "AvgQuantity",
        "AvgUnitPrice"
    ]

    X = customer_df[feature_cols]

    y = customer_df["Risk"]

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


# -----------------------------
# Create Sequences
# -----------------------------

def create_sequences(X):

    """
    Reshape into

    (samples,
     timesteps,
     features)

    Required for LSTM/GRU.
    """

    X = np.reshape(
        X,
        (
            X.shape[0],
            1,
            X.shape[1]
        )
    )

    return X


# -----------------------------
# Train Test Split
# -----------------------------

def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


# -----------------------------
# Build LSTM Model
# -----------------------------

def build_lstm(input_shape):

    model = Sequential([
        LSTM(
            64,
            input_shape=input_shape,
            kernel_regularizer=l2(0.001),
            return_sequences=False
        ),

        BatchNormalization(),

        Dropout(0.3),

        Dense(
            32,
            activation="relu",
            kernel_regularizer=l2(0.001)
        ),

        Dropout(0.2),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# Build GRU Model
# -----------------------------

def build_gru(input_shape):

    model = Sequential([
        GRU(
            64,
            input_shape=input_shape,
            kernel_regularizer=l2(0.001),
            return_sequences=False
        ),

        BatchNormalization(),

        Dropout(0.3),

        Dense(
            32,
            activation="relu",
            kernel_regularizer=l2(0.001)
        ),

        Dropout(0.2),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# Early Stopping
# -----------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# -----------------------------
# Train Model
# -----------------------------

def train_model(model, X_train, y_train):

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    return history


# -----------------------------
# Evaluate Model
# -----------------------------

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    predictions = (predictions > 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    return accuracy, precision, recall, f1


# -----------------------------
# Accuracy Graph
# -----------------------------

def plot_accuracy(history, filename):

    plt.figure(figsize=(8,5))

    plt.plot(history.history["accuracy"])

    plt.plot(history.history["val_accuracy"])

    plt.title("Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend(["Train","Validation"])

    plt.savefig(os.path.join(GRAPH_DIR, filename))

    plt.close()


# -----------------------------
# Loss Graph
# -----------------------------

def plot_loss(history, filename):

    plt.figure(figsize=(8,5))

    plt.plot(history.history["loss"])

    plt.plot(history.history["val_loss"])

    plt.title("Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend(["Train","Validation"])

    plt.savefig(os.path.join(GRAPH_DIR, filename))

    plt.close()


    # -----------------------------
# Isolation Forest
# -----------------------------

def anomaly_detection(customer_df):

    print("Running Anomaly Detection...")

    detector = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    features = customer_df[
        [
            "Recency",
            "Frequency",
            "Monetary",
            "AvgQuantity",
            "AvgUnitPrice"
        ]
    ]

    customer_df["Anomaly"] = detector.fit_predict(features)

    return customer_df


# -----------------------------
# Risk Graph
# -----------------------------

def plot_risk(customer_df):

    plt.figure(figsize=(8,6))

    colors = customer_df["Anomaly"].map({
        1: "green",
        -1: "red"
    })

    plt.scatter(
        customer_df["Frequency"],
        customer_df["Monetary"],
        c=colors,
        alpha=0.7
    )

    plt.xlabel("Purchase Frequency")
    plt.ylabel("Monetary Value")
    plt.title("Customer Risk Analysis")

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "risk_graph.png"
        )
    )

    plt.close()


# -----------------------------
# Save Metrics
# -----------------------------

def save_metrics(lstm_metrics, gru_metrics):

    metrics = pd.DataFrame({

        "Model":[
            "LSTM",
            "GRU"
        ],

        "Accuracy":[
            lstm_metrics[0],
            gru_metrics[0]
        ],

        "Precision":[
            lstm_metrics[1],
            gru_metrics[1]
        ],

        "Recall":[
            lstm_metrics[2],
            gru_metrics[2]
        ],

        "F1 Score":[
            lstm_metrics[3],
            gru_metrics[3]
        ]

    })

    metrics.to_csv(

        os.path.join(
            METRIC_DIR,
            "sequence_metrics.csv"
        ),

        index=False

    )

    print(metrics)


# -----------------------------
# Save Models
# -----------------------------

def save_models(

    lstm_model,

    gru_model

):

    lstm_model.save(

        os.path.join(

            MODEL_DIR,

            "lstm_model.keras"

        )

    )

    gru_model.save(

        os.path.join(

            MODEL_DIR,

            "gru_model.keras"

        )

    )


# -----------------------------
# Main
# -----------------------------

def main():

    print("="*60)

    print("Sequence Models Pipeline")

    print("="*60)

    df = load_dataset()

    df = clean_dataset(df)

    customer_df = feature_engineering(df)

    customer_df = create_target(customer_df)

    X, y, scaler = preprocess(customer_df)

    X = create_sequences(X)

    X_train, X_test, y_train, y_test = split_data(X, y)

    input_shape = (

        X_train.shape[1],

        X_train.shape[2]

    )

    print("\nTraining LSTM...")

    lstm_model = build_lstm(input_shape)

    lstm_history = train_model(

        lstm_model,

        X_train,

        y_train

    )

    lstm_metrics = evaluate_model(

        lstm_model,

        X_test,

        y_test

    )

    plot_accuracy(

        lstm_history,

        "lstm_accuracy.png"

    )

    plot_loss(

        lstm_history,

        "lstm_loss.png"

    )

    print("\nTraining GRU...")

    gru_model = build_gru(input_shape)

    gru_history = train_model(

        gru_model,

        X_train,

        y_train

    )

    gru_metrics = evaluate_model(

        gru_model,

        X_test,

        y_test

    )

    plot_accuracy(

        gru_history,

        "gru_accuracy.png"

    )

    plot_loss(

        gru_history,

        "gru_loss.png"

    )

    customer_df = anomaly_detection(customer_df)

    plot_risk(customer_df)

    save_metrics(

        lstm_metrics,

        gru_metrics

    )

    save_models(

        lstm_model,

        gru_model

    )

    print("\nSequence Models Completed Successfully.")

    print("Graphs saved in outputs/graphs")

    print("Models saved in models/")

    print("Metrics saved in outputs/metrics")


if __name__ == "__main__":

    main()