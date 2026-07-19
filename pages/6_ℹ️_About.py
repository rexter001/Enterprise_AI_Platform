import streamlit as st

st.title("ℹ️ About")
st.caption("Enterprise AI Platform | AI/ML Team Project")

st.divider()

# -----------------------------------
# Project Overview
# -----------------------------------

st.header("🤖 Project Overview")

st.markdown("""
The **Enterprise AI Platform** is an integrated Artificial Intelligence and
Machine Learning dashboard designed to provide business intelligence through
data-driven analysis.

The platform combines customer analytics, Natural Language Processing,
sentiment analysis, time series forecasting, and automated reporting into a
single interactive Streamlit application.
""")

st.divider()

# -----------------------------------
# Project Objectives
# -----------------------------------

st.header("🎯 Project Objectives")

col1, col2 = st.columns(2)

with col1:
    st.success("✔ Customer Behavior Analysis")
    st.success("✔ Customer Segmentation")
    st.success("✔ Sentiment Analysis")

with col2:
    st.success("✔ Sales Forecasting")
    st.success("✔ Business Intelligence")
    st.success("✔ AI-Based Decision Support")

st.divider()

# -----------------------------------
# Dataset
# -----------------------------------

st.header("📂 Dataset")

st.info("""
**Dataset Used:** Online Retail Transaction Dataset

The dataset contains customer purchase information including transactions,
products, quantities, prices, and customer details.

It is used for:
• Customer Analytics  
• Feature Engineering  
• Sales Forecasting  
• Business Insights Generation
""")

st.divider()

# -----------------------------------
# Technology Stack
# -----------------------------------

st.header("🛠 Technology Stack")

tech1, tech2, tech3 = st.columns(3)

with tech1:
    st.info("🐍 Python")
    st.info("📈 Streamlit")
    st.info("🐼 Pandas")

with tech2:
    st.info("🧠 Scikit-Learn")
    st.info("🤖 TensorFlow")
    st.info("📚 NLTK")

with tech3:
    st.info("📊 Prophet")
    st.info("📉 ARIMA")
    st.info("📈 Matplotlib")

st.divider()

# -----------------------------------
# Project Modules
# -----------------------------------

st.header("📂 Project Modules")

st.markdown("""
- 📊 **Customer Analytics**
    - Customer behavior analysis
    - Feature engineering
    - Neural network-based insights

- 💬 **NLP & Sentiment Analysis**
    - Text preprocessing
    - Review analysis
    - Sentiment classification

- 📈 **Forecasting**
    - Time series analysis
    - ARIMA forecasting
    - Prophet forecasting

- 📑 **Reports & Business Insights**
    - Model summaries
    - Performance metrics
    - Business recommendations
""")

st.divider()

# -----------------------------------
# Team Responsibilities
# -----------------------------------

st.header("👥 Team Responsibilities")

st.markdown("""
| Role | Responsibility |
|---|---|
| AI/ML Engineer | Neural Networks & Model Development |
| NLP Engineer | Text Processing & Sentiment Analysis |
| Forecasting Engineer | Time Series Analysis & Prediction |
| Analytics Engineer | Customer Segmentation & Feature Engineering |
| Team Lead | Dashboard Integration, Testing & Deployment |
""")

st.divider()

# -----------------------------------
# Workflow
# -----------------------------------

st.header("🔄 Project Workflow")

st.info("""
Dataset Collection  
↓  
Data Cleaning & Preprocessing  
↓  
Feature Engineering  
↓  
Machine Learning / NLP Models  
↓  
Forecasting & Prediction  
↓  
Visualization Dashboard  
↓  
Business Reports & Insights
""")

st.divider()

# -----------------------------------
# Future Enhancements
# -----------------------------------

st.header("🚀 Future Enhancements")

st.markdown("""
- Real-time data processing
- Cloud deployment
- User authentication
- Automated report generation
- AI-powered recommendations
- Real-time business dashboards
""")

st.divider()

st.success(
    "Enterprise AI Platform • AI/ML Internship Project • 2026"
)