import streamlit as st

st.title("ℹ️ About")
st.caption("Enterprise AI Platform | AI/ML Team Project")

st.divider()

# -----------------------------------
# Project Overview
# -----------------------------------

st.header("🤖 Project Overview")

st.markdown("""
The **Enterprise AI Platform** is an integrated dashboard that combines
multiple Artificial Intelligence and Machine Learning modules into a single
business intelligence application.

The platform provides customer analytics, sentiment analysis, forecasting,
and reporting through an interactive Streamlit dashboard.
""")

st.divider()

# -----------------------------------
# Project Objectives
# -----------------------------------

st.header("🎯 Project Objectives")

col1, col2 = st.columns(2)

with col1:
    st.success("✔ Customer Analytics")
    st.success("✔ Sentiment Analysis")
    st.success("✔ Sales Forecasting")

with col2:
    st.success("✔ Business Insights")
    st.success("✔ Interactive Dashboard")
    st.success("✔ Unified AI Platform")

st.divider()

# -----------------------------------
# Technology Stack
# -----------------------------------

st.header("🛠 Technology Stack")

tech1, tech2, tech3 = st.columns(3)

with tech1:
    st.info("🐍 Python")
    st.info("📈 Streamlit")

with tech2:
    st.info("🧠 Scikit-Learn")
    st.info("🤖 TensorFlow")

with tech3:
    st.info("🐼 Pandas")
    st.info("🔢 NumPy")

st.divider()

# -----------------------------------
# Project Modules
# -----------------------------------

st.header("📂 Project Modules")

st.markdown("""
- 📊 Customer Analytics
- 💬 NLP & Sentiment Analysis
- 📈 Forecasting
- 📑 Reports & Business Insights
""")

st.divider()

# -----------------------------------
# Team Responsibilities
# -----------------------------------

st.header("👥 Team Responsibilities")

st.markdown("""
| Team Member | Responsibility |
|------------|----------------|
| Member 1 | Neural Networks |
| Member 2 | NLP & Sentiment Analysis |
| Member 3 | Time Series Forecasting |
| Member 4 | Customer Segmentation & Feature Engineering |
| Team Lead | Dashboard Integration, Testing & Deployment |
""")

st.divider()

# -----------------------------------
# Workflow
# -----------------------------------

st.header("🔄 Project Workflow")

st.info("""
Dataset → Data Processing → Machine Learning Models →
Visualization → Business Reports
""")

st.divider()

# -----------------------------------
# Future Enhancements
# -----------------------------------

st.header("🚀 Future Enhancements")

st.markdown("""
- Real-time analytics
- Cloud deployment
- User authentication
- Interactive dashboards
- Automated report generation
- AI-powered recommendations
""")

st.divider()

st.success("Enterprise AI Platform © 2026")