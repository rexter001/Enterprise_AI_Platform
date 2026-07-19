import streamlit as st

st.title("🏠 Home")
st.caption("Enterprise AI Platform Dashboard")

st.divider()

st.markdown("""
## Welcome!

The **Enterprise AI Platform** integrates Customer Analytics, NLP Sentiment Analysis, 
Time Series Forecasting and Business Reporting into one interactive dashboard for business intelligence.

Navigate through the modules using the sidebar.
""")

st.divider()

st.subheader("🎯 Project Objectives")

col1, col2 = st.columns(2)

with col1:
    st.success("✔ Customer Analytics")
    st.success("✔ NLP & Sentiment Analysis")
    st.success("✔ Time Series Forecasting")

with col2:
    st.success("✔ Business Reporting")
    st.success("✔ Interactive Dashboard")
    st.success("✔ AI Model Integration")

st.divider()

st.subheader("📊 Project Summary")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Datasets", "2")

with c2:
    st.metric("Modules", "5")

with c3:
    st.metric("Project Status", "Completed")

st.divider()

st.subheader("🛠 Technology Stack")

tech1, tech2, tech3 = st.columns(3)

with tech1:
    st.info("🐍 Python")
    st.info("📈 Streamlit")

with tech2:
    st.info("🤖 TensorFlow")
    st.info("🧠 Scikit-Learn")

with tech3:
    st.info("🐼 Pandas")
    st.info("🔢 NumPy")

st.divider()

st.info("🚀 Explore Customer Analytics, NLP, Forecasting, Reports and About using the sidebar.")