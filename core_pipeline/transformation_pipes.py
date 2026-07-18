"""
core_pipeline/transformation_pipes.py
--------------------------------------
Member 4 - Sub-Module E: Scalable Data Transformation Pipes (Feature Engineering)

What this file does (plain English):
1. Loads the retail transactions data.
2. Builds customer-level features: lag features (time between orders),
   target encoding (avg spend per category), and handles class imbalance
   with SMOTE + class weights.
3. Wraps everything into a single sklearn Pipeline + ColumnTransformer so
   preprocessing is reproducible and reusable by the dashboard (app.py).
4. Compares model performance BEFORE vs AFTER these transformations,
   and saves a comparison chart for the report.

Expected input columns (standard "Online Retail" dataset):
    InvoiceNo, StockCode, Description, Quantity, InvoiceDate,
    UnitPrice, CustomerID, Country

If your actual CSV uses different column names, just edit the
COLUMN MAPPING section below — everything else stays the same.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("[WARN] imbalanced-learn not installed. Run: pip install imbalanced-learn")


# ----------------------------------------------------------------------
# 0. COLUMN MAPPING  -- edit this if your CSV headers differ
# ----------------------------------------------------------------------
COL = {
    "invoice": "InvoiceNo",
    "stock_code": "StockCode",
    "description": "Description",
    "quantity": "Quantity",
    "invoice_date": "InvoiceDate",
    "unit_price": "UnitPrice",
    "customer_id": "CustomerID",
    "country": "Country",
}


# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
def load_data(path="datasets/retail_transactions.csv"):
    """Load raw transactions and do minimal cleaning."""
    df = pd.read_csv(path, encoding="ISO-8859-1")

    # Basic cleaning: drop rows with no customer id, negative/zero quantity or price
    df = df.dropna(subset=[COL["customer_id"]])
    df = df[df[COL["quantity"]] > 0]
    df = df[df[COL["unit_price"]] > 0]

    df[COL["invoice_date"]] = pd.to_datetime(df[COL["invoice_date"]], errors="coerce")
    df = df.dropna(subset=[COL["invoice_date"]])

    df["TotalPrice"] = df[COL["quantity"]] * df[COL["unit_price"]]
    return df


# ----------------------------------------------------------------------
# 2. LAG FEATURES  -- "how long since this customer's last order?"
# ----------------------------------------------------------------------
def add_lag_features(df):
    """
    For each customer, sort their orders by date and compute:
      - days_since_last_order   (lag feature: gap between consecutive orders)
      - order_seq_num           (1st order, 2nd order, ...)
      - avg_gap_days_so_far     (rolling average gap = a smoothed lag feature)
    """
    df = df.sort_values([COL["customer_id"], COL["invoice_date"]]).copy()

    df["days_since_last_order"] = (
        df.groupby(COL["customer_id"])[COL["invoice_date"]]
        .diff()
        .dt.days
    )
    df["days_since_last_order"] = df["days_since_last_order"].fillna(0)

    df["order_seq_num"] = df.groupby(COL["customer_id"]).cumcount() + 1

    df["avg_gap_days_so_far"] = (
        df.groupby(COL["customer_id"])["days_since_last_order"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Weekday vs weekend ratio signal (useful behavioural feature)
    df["is_weekend"] = df[COL["invoice_date"]].dt.weekday >= 5

    return df


# ----------------------------------------------------------------------
# 3. TARGET ENCODING  -- "avg spend for this product category / country"
# ----------------------------------------------------------------------
def target_encode(df, group_col, target_col="TotalPrice", smoothing=10):
    """
    Smoothed mean target encoding.
    Instead of one-hot encoding a high-cardinality column (like Country or
    StockCode), we replace each category with a smoothed average of the
    target value for that category. Smoothing pulls rare categories toward
    the global mean so they don't overfit.
    """
    global_mean = df[target_col].mean()

    agg = df.groupby(group_col)[target_col].agg(["mean", "count"])
    smoothed = (agg["mean"] * agg["count"] + global_mean * smoothing) / (
        agg["count"] + smoothing
    )

    encoded_col = f"{group_col}_target_enc"
    df[encoded_col] = df[group_col].map(smoothed)
    df[encoded_col] = df[encoded_col].fillna(global_mean)
    return df, encoded_col


# ----------------------------------------------------------------------
# 4. BUILD A CUSTOMER-LEVEL FEATURE TABLE
#    (this is what feeds clustering AND classification)
# ----------------------------------------------------------------------
def build_customer_features(df):
    df = add_lag_features(df)
    df, country_enc_col = target_encode(df, COL["country"])

    customer_features = df.groupby(COL["customer_id"]).agg(
        total_spend=("TotalPrice", "sum"),
        avg_order_value=("TotalPrice", "mean"),
        num_orders=(COL["invoice"], "nunique"),
        avg_gap_days=("avg_gap_days_so_far", "last"),
        weekend_order_ratio=("is_weekend", "mean"),
        country_target_enc=(country_enc_col, "mean"),
    ).reset_index()

    # Simple binary target for demo classification: "high value customer"
    # (top 25% spenders = 1, rest = 0) — used only to demo SMOTE/class weights
    threshold = customer_features["total_spend"].quantile(0.75)
    customer_features["high_value"] = (
        customer_features["total_spend"] >= threshold
    ).astype(int)

    return customer_features


# ----------------------------------------------------------------------
# 5. COLUMNTRANSFORMER + PIPELINE
# ----------------------------------------------------------------------
def build_preprocessing_pipeline(numeric_features, categorical_features):
    """
    One unified preprocessing block:
      - numeric: impute missing -> scale
      - categorical: impute missing -> one-hot encode
    Combined via ColumnTransformer so it can be dropped straight into
    any sklearn Pipeline (and reused identically at inference time).
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    return preprocessor


# ----------------------------------------------------------------------
# 6. HANDLE CLASS IMBALANCE: SMOTE + class weights
# ----------------------------------------------------------------------
def apply_smote(X_train, y_train):
    if not IMBLEARN_AVAILABLE:
        print("[SKIP] SMOTE skipped — imbalanced-learn not installed.")
        return X_train, y_train
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def get_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return dict(zip(classes, weights))


# ----------------------------------------------------------------------
# 7. BEFORE vs AFTER COMPARISON
# ----------------------------------------------------------------------
def compare_before_after(customer_features, numeric_features, categorical_features,
                          target="high_value", save_path="analytical_reports"):
    os.makedirs(save_path, exist_ok=True)

    X = customer_features[numeric_features + categorical_features]
    y = customer_features[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)

    # ---- BEFORE: raw features, no scaling/encoding pipeline, no SMOTE ----
    X_train_raw = pd.get_dummies(X_train, columns=categorical_features).fillna(0)
    X_test_raw = pd.get_dummies(X_test, columns=categorical_features).fillna(0)
    X_test_raw = X_test_raw.reindex(columns=X_train_raw.columns, fill_value=0)

    model_before = RandomForestClassifier(random_state=42)
    model_before.fit(X_train_raw, y_train)
    preds_before = model_before.predict(X_test_raw)
    acc_before = accuracy_score(y_test, preds_before)
    f1_before = f1_score(y_test, preds_before)

    # ---- AFTER: full Pipeline (ColumnTransformer) + SMOTE + class weights ----
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    X_train_bal, y_train_bal = apply_smote(X_train_processed, y_train)
    weights = get_class_weights(y_train_bal)

    full_pipeline = Pipeline(steps=[
        ("classifier", RandomForestClassifier(
            random_state=42,
            class_weight=weights if not IMBLEARN_AVAILABLE else "balanced"
        )),
    ])
    full_pipeline.fit(X_train_bal, y_train_bal)
    preds_after = full_pipeline.predict(X_test_processed)
    acc_after = accuracy_score(y_test, preds_after)
    f1_after = f1_score(y_test, preds_after)

    print("BEFORE  -> Accuracy: {:.3f} | F1: {:.3f}".format(acc_before, f1_before))
    print("AFTER   -> Accuracy: {:.3f} | F1: {:.3f}".format(acc_after, f1_after))
    print("\nClassification report (AFTER):\n", classification_report(y_test, preds_after))

    # ---- chart for the report/slides ----
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = ["Accuracy", "F1-score"]
    before_vals = [acc_before, f1_before]
    after_vals = [acc_after, f1_after]
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, before_vals, width, label="Before")
    ax.bar(x + width/2, after_vals, width, label="After")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("Model Performance: Before vs After Feature Pipeline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, "before_after_comparison.png"))
    plt.close(fig)

    return {
        "before": {"accuracy": acc_before, "f1": f1_before},
        "after": {"accuracy": acc_after, "f1": f1_after},
        "preprocessor": preprocessor,
        "model": full_pipeline,
    }


# ----------------------------------------------------------------------
# 8. MAIN — run standalone for testing / to feed the dashboard
# ----------------------------------------------------------------------
if __name__ == "__main__":
    raw_df = load_data("datasets/online_retail.csv")
    customer_features = build_customer_features(raw_df)

    numeric_features = [
        "total_spend", "avg_order_value", "num_orders",
        "avg_gap_days", "weekend_order_ratio", "country_target_enc",
    ]
    categorical_features = []  # already target-encoded, so none needed here

    customer_features.to_csv("analytical_reports/customer_features.csv", index=False)
    print(f"Built feature table with {customer_features.shape[0]} customers "
          f"and {customer_features.shape[1]} columns.")

    results = compare_before_after(customer_features, numeric_features, categorical_features)
    print(results["before"], results["after"])