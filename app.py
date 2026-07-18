import streamlit as st

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Enterprise AI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Main Dashboard
# -----------------------------
st.title("🤖 Enterprise AI Platform")

st.markdown("""
### Welcome!

This platform integrates multiple Artificial Intelligence and Machine Learning
modules into a single enterprise dashboard.

Use the **sidebar** to navigate between different modules.
""")

# -----------------------------
# Project Overview
# -----------------------------
st.subheader("📌 Project Overview")

st.write("""
The Enterprise AI Platform combines customer analytics, natural language
processing, forecasting, and reporting into one unified application.

The platform is designed to demonstrate multiple AI techniques working together
through a single interactive dashboard.
""")

# -----------------------------
# Available Modules
# -----------------------------
st.subheader("📂 Available Modules")

col1, col2 = st.columns(2)

with col1:
    st.info("📊 Customer Analytics")
    st.info("💬 NLP & Sentiment")

with col2:
    st.info("📈 Forecasting")
    st.info("📑 Reports")

# -----------------------------
# Dataset Information
# -----------------------------
st.subheader("📁 Datasets")

st.markdown("""
- **online_retail.csv** — Customer analytics, segmentation, forecasting
- **amazon.csv** — NLP and sentiment analysis
""")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Enterprise AI Platform | AI/ML Team Project")