import streamlit as st

st.set_page_config(
    page_title="Enterprise AI Platform",
    page_icon="🤖",
    layout="wide"
)

st.title("Integrated AI Business Intelligence Platform")

st.markdown("---")

st.write("Project initialized successfully!")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Home",
        "Customer Analytics",
        "Natural Language",
        "Forecasting",
        "Reports"
    ]
)

if page == "Home":
    st.header("Welcome")
    st.write("This is the main dashboard.")

elif page == "Customer Analytics":
    st.header("Customer Analytics Module")

elif page == "Natural Language":
    st.header("NLP Module")

elif page == "Forecasting":
    st.header("Forecasting Module")

elif page == "Reports":
    st.header("Reports")