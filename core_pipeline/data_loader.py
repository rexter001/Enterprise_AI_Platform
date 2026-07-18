import streamlit as st
import pandas as pd
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"


@st.cache_data
def load_online_retail():
    """
    Load the Online Retail dataset.
    """
    file_path = DATASET_DIR / "online_retail.csv"

    try:
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
        return df
    except FileNotFoundError:
        st.error("❌ online_retail.csv not found in datasets folder.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading Online Retail dataset: {e}")
        return None


@st.cache_data
def load_amazon_reviews():
    """
    Load the Amazon Reviews dataset.
    """
    file_path = DATASET_DIR / "amazon.csv"

    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error("❌ amazon.csv not found in datasets folder.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading Amazon dataset: {e}")
        return None