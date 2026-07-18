import pandas as pd


def dataset_summary(df: pd.DataFrame):
    """
    Returns basic information about the dataset.
    """

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def dataset_columns(df: pd.DataFrame):
    """
    Returns all column names.
    """
    return list(df.columns)


def missing_values(df: pd.DataFrame):
    """
    Returns missing values for each column.
    """
    return df.isnull().sum()


def numerical_summary(df: pd.DataFrame):
    """
    Returns summary statistics of numerical columns.
    """
    return df.describe()


def categorical_summary(df: pd.DataFrame):
    """
    Returns summary statistics of categorical columns.
    """
    return df.describe(include="object")