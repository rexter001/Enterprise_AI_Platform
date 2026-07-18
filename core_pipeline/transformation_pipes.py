import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


class DataTransformationPipeline:
    """
    Data preprocessing and feature engineering
    for the Online Retail dataset.
    """

    def preprocess_data(self, df):
        """
        Clean the dataset before feature engineering.
        """

        df = df.copy()

        # Remove rows with missing CustomerID
        df = df.dropna(subset=["CustomerID"])

        # Remove cancelled invoices
        df = df[
            ~df["InvoiceNo"].astype(str).str.startswith("C")
        ]

        # Remove invalid Quantity
        df = df[df["Quantity"] > 0]

        # Remove invalid UnitPrice
        df = df[df["UnitPrice"] > 0]

        # Convert InvoiceDate to datetime
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        return df

    def create_lag_features(self, df):
        """
        Create lag features using previous customer purchases.
        """

        df = df.copy()

        # Sort transactions
        df = df.sort_values(
            ["CustomerID", "InvoiceDate"]
        )

        # Previous Quantity
        df["PreviousQuantity"] = (
            df.groupby("CustomerID")["Quantity"]
            .shift(1)
            .fillna(0)
        )

        # Previous Unit Price
        df["PreviousUnitPrice"] = (
            df.groupby("CustomerID")["UnitPrice"]
            .shift(1)
            .fillna(0)
        )

        return df

    def build_preprocessor(self):
        """
        Create preprocessing pipeline for numerical
        and categorical features.
        """

        numeric_features = [
            "Quantity",
            "UnitPrice",
            "PreviousQuantity",
            "PreviousUnitPrice"
        ]

        categorical_features = [
            "Country"
        ]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features)
            ]
        )

        return preprocessor
    
        
    def create_high_value_customer(self, df):
        """
        Create a High_Value_Customer label based on total spending.
        """

        df = df.copy()

        # Calculate transaction spend
        df["TotalSpend"] = df["Quantity"] * df["UnitPrice"]

        # Calculate total spend per customer
        customer_spend = (
            df.groupby("CustomerID")["TotalSpend"]
            .sum()
            .reset_index()
        )

        # Median threshold
        threshold = customer_spend["TotalSpend"].median()

        # High value label
        customer_spend["High_Value_Customer"] = (
            customer_spend["TotalSpend"] >= threshold
        ).astype(int)

        # Merge back
        df = df.merge(
            customer_spend[
                ["CustomerID", "High_Value_Customer"]
            ],
            on="CustomerID",
            how="left"
        )

        return df