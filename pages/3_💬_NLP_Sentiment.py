import streamlit as st

from core_pipeline.data_loader import load_amazon_reviews
from core_pipeline.helpers import dataset_summary

st.title("💬 NLP & Sentiment Analysis")
st.caption("Text Processing • Sentiment Classification • Review Analytics")

st.divider()

st.markdown("""
This module analyzes customer reviews using Natural Language Processing (NLP)
and sentiment analysis techniques.
""")

# -----------------------------------
# Dashboard Summary
# -----------------------------------

df = load_amazon_reviews()
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
    st.success("Amazon Reviews dataset loaded successfully!")
    st.dataframe(df.head(10), use_container_width=True)
else:
    st.error("Dataset could not be loaded.")

st.divider()

# -----------------------------------
# Text Preprocessing
# -----------------------------------

st.subheader("📝 Text Preprocessing")

left, right = st.columns(2)

with left:
    st.info("Cleaning & Tokenization")

with right:
    st.info("Stopword Removal & Lemmatization")

st.divider()

# -----------------------------------
# Sentiment Analysis
# -----------------------------------

st.subheader("😊 Sentiment Analysis")

st.warning("Waiting for Member 2 integration.")

st.divider()

# -----------------------------------
# Visualizations
# -----------------------------------

st.subheader("📊 Visualizations")

chart1, chart2 = st.columns(2)

with chart1:
    st.empty()

with chart2:
    st.empty()

st.info("Sentiment charts, word clouds, and graphs will appear here.")