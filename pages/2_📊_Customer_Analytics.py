import streamlit as st

from customer_analytics.neural_nets import (
    load_and_prepare_data,
    preprocess_data,
    build_mlp,
    train_mlp,
    evaluate_model,
    save_model,
)

from core_pipeline.data_loader import load_online_retail
from core_pipeline.helpers import dataset_summary

st.title("📊 Customer Analytics")
st.caption("Customer Segmentation • Neural Networks • Business Insights")

st.divider()

st.markdown("""
This module analyzes customer purchasing behavior using machine learning
techniques and customer segmentation.
""")

# -------------------------------
# Dashboard Summary
# -------------------------------

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

# -------------------------------
# Dataset Preview
# -------------------------------

st.subheader("📁 Dataset Preview")

if df is not None:
    st.success("Dataset loaded successfully!")
    st.dataframe(df.head(10), use_container_width=True)
else:
    st.error("Dataset could not be loaded.")

# -------------------------------
# Customer Statistics
# -------------------------------

st.subheader("📈 Customer Statistics")

left, right = st.columns(2)

left, right = st.columns(2)

with left:
    st.bar_chart(df["Country"].value_counts().head(10))

with right:
    st.bar_chart(
        df.groupby("Country")["Quantity"].sum().sort_values(ascending=False).head(10)
    )

st.divider()

# -------------------------------
# Neural Network
# -------------------------------

st.subheader("🧠 Neural Network")

cfg1, cfg2, cfg3 = st.columns(3)

with cfg1:
    activation = st.selectbox(
        "Activation",
        ["relu", "sigmoid", "tanh"]
    )

with cfg2:
    optimizer = st.selectbox(
        "Optimizer",
        ["Adam", "SGD", "RMSprop"]
    )

with cfg3:
    epochs = st.slider(
        "Epochs",
        min_value=5,
        max_value=100,
        value=20
    )

train = st.button(
    "🚀 Train Neural Network",
    use_container_width=True
)

if train:

    # Load and preprocess data
    customer_data, X, y = load_and_prepare_data(
        "datasets/online_retail.csv"
    )

    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        X,
        y
    )

    st.success("Dataset prepared successfully!")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Customers", customer_data.shape[0])

    with c2:
        st.metric("Training Samples", X_train.shape[0])

    with c3:
        st.metric("Testing Samples", X_test.shape[0])

    # Build model
    model = build_mlp(
        activation=activation,
        optimizer_name=optimizer
    )

    # Train model
    history = train_mlp(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        epochs=epochs
    )

    # Evaluate
    accuracy, report, matrix, predictions = evaluate_model(
        model,
        X_test,
        y_test
    )

    st.success("Model trained successfully!")

    m1, m2 = st.columns(2)

    with m1:
        st.metric(
            "Accuracy",
            f"{accuracy:.2%}"
        )

    with m2:
        st.metric(
            "Epochs",
            epochs
        )

    with st.expander("📄 Classification Report"):
        st.code(report)

    with st.expander("📊 Confusion Matrix"):
        st.dataframe(matrix)

    g1, g2 = st.columns(2)

    with g1:
        st.subheader("📈 Training Accuracy")
        st.line_chart(history.history["accuracy"])

        if "val_accuracy" in history.history:
            st.subheader("📈 Validation Accuracy")
            st.line_chart(history.history["val_accuracy"])

    with g2:
        st.subheader("📉 Training Loss")
        st.line_chart(history.history["loss"])

        if "val_loss" in history.history:
            st.subheader("📉 Validation Loss")
            st.line_chart(history.history["val_loss"])

    st.divider()

    # -------------------------------
    # Visualizations
    # -------------------------------

    st.subheader("📊 Visualizations")

    v1, v2 = st.columns(2)

    with v1:
        st.image(
            "analytical_reports/model_comparison.png",
            caption="Model Comparison",
            use_container_width=True
        )

    with v2:
        st.image(
            "analytical_reports/loss_curve.png",
            caption="Training Loss Curve",
            use_container_width=True
        )

    st.divider()

    # -------------------------------
    # Export Model
    # -------------------------------

    st.subheader("💾 Export Model")

    if st.button("💾 Save Model", use_container_width=True):
        save_model(model)
        st.success("Model saved successfully!")