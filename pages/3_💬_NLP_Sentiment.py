import streamlit as st

from natural_language.nlp_pipelines import (
    build_nlp_features,
    sentiment_analysis,
    sentiment_distribution,
    most_common_words,
    tfidf_embeddings,
    get_top_tfidf_features,
)

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

with st.spinner("Running NLP preprocessing..."):
    nlp_df = build_nlp_features(df)

    st.write("Step 2: Preprocessing completed")

    st.success("NLP preprocessing completed successfully!")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Reviews", len(nlp_df))

    with c2:
        st.metric(
            "Processed Reviews",
            nlp_df["clean_review"].notna().sum()
        )

    with c3:
        st.metric(
            "Tokenized Reviews",
            nlp_df["tokens"].notna().sum()
        )

    st.subheader("📄 Processed Reviews")

    st.dataframe(
        nlp_df[
            [
                "review_content",
                "clean_review"
            ]
        ].head(10),
        use_container_width=True
    )

st.divider()

# -----------------------------------
# Sentiment Analysis
# -----------------------------------

st.subheader("😊 Sentiment Analysis")

sentiment_df = sentiment_analysis(nlp_df)

sentiment_counts = sentiment_distribution(sentiment_df)
st.success("Sentiment analysis completed successfully!")

c1, c2, c3 = st.columns(3)

with c1:
        st.metric(
            "Positive",
            sentiment_counts.get("Positive", 0)
        )

with c2:
        st.metric(
            "Neutral",
            sentiment_counts.get("Neutral", 0)
        )

with c3:
        st.metric(
            "Negative",
            sentiment_counts.get("Negative", 0)
        )

st.subheader("📄 Reviews with Sentiment")

st.dataframe(
        sentiment_df[
            [
                "review_content",
                "rating",
                "sentiment"
            ]
        ].head(10),
        use_container_width=True
    )

st.divider()

# -------------------------------
# Visualizations
# -------------------------------

st.subheader("📊 Visualizations")

v1, v2 = st.columns(2)

with v1:
    st.image(
        "analytical_reports/model_comparison.png",
        caption="Model Comparison",
        use_container_width=True
    )

with v2:
    st.image(
        "analytical_reports/loss_curve.png",
        caption="Training Loss Curve",
        use_container_width=True
    )

st.divider()

# -------------------------------
# Export Model
# -------------------------------

st.subheader("💾 Export Model")

if st.button("💾 Save Model", use_container_width=True):
    save_model(model)
    st.success("Model saved successfully!")