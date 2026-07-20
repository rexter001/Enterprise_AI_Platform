import streamlit as st

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Enterprise AI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Header
# ----------------------------
st.title("🤖 Enterprise AI Platform")
st.caption("Unified Enterprise AI Business Intelligence & Analytics Platform")

st.divider()

# ----------------------------
# Welcome
# ----------------------------
st.success("Welcome! Select a module from the sidebar to begin.")

# ----------------------------
# Quick Summary
# ----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Datasets", "2")

with col2:
    st.metric("Modules", "6")

with col3:
    st.metric("Status", "Ready")

st.divider()

st.markdown("""
### 🚀 Project Modules

- 📊 Customer Analytics
- 💬 NLP & Sentiment Analysis
- 📈 Forecasting
- 📑 Reports
- ℹ️ About

Use the sidebar to explore each module.
""")

st.info("✅ All AI modules have been successfully integrated. Use the sidebar to explore the platform.")