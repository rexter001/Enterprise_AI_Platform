import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class DataTransformationPipeline:
    """
    Handles preprocessing for the Online Retail dataset.
    """

    def preprocess_data(self, df):
        """
        Clean the Online Retail dataset before feature engineering.
        """

        # Create a copy so the original data is not modified
        df = df.copy()

        # Remove rows without a Customer ID
        df = df.dropna(subset=["CustomerID"])

        # Remove cancelled invoices (InvoiceNo starting with 'C')
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

        # Remove invalid quantities
        df = df[df["Quantity"] > 0]

        # Remove invalid prices
        df = df[df["UnitPrice"] > 0]

        # Convert InvoiceDate into datetime format
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        return df


    def create_lag_features(self, df):
        """
        Creates lag features based on previous customer purchases.
        """

        df = df.copy()

        # Sort data by CustomerID and InvoiceDate
        df = df.sort_values(["CustomerID", "InvoiceDate"])

        # Previous purchase quantity
        df["PreviousQuantity"] = (
            df.groupby("CustomerID")["Quantity"]
            .shift(1)
            .fillna(0)
        )

        # Previous purchase price
        df["PreviousUnitPrice"] = (
            df.groupby("CustomerID")["UnitPrice"]
            .shift(1)
            .fillna(0)
        )

        return df
        def build_preprocessor(self):
        """
        Build preprocessing pipeline.
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