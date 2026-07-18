"""
===========================================================
Enterprise AI Business Intelligence Platform
Natural Language Processing Pipeline
===========================================================

This module provides:
1. Text Preprocessing
2. Tokenization
3. Stopword Removal
4. Stemming
5. Lemmatization
6. POS Tagging
7. TF-IDF Vectorization
8. Word2Vec Embeddings
9. GloVe Embeddings
10. Named Entity Recognition
11. Sentiment Analysis
12. WordCloud Generation

Author: Team AI Ascenders
"""

# =========================================================
# IMPORTS
# =========================================================

import os
import re
import string
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from wordcloud import WordCloud

# ===========================
# NLTK
# ===========================

import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# ===========================
# Scikit Learn
# ===========================

from sklearn.feature_extraction.text import TfidfVectorizer

# ===========================
# Gensim
# ===========================

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# ===========================
# spaCy
# ===========================

import spacy

# =========================================================
# WARNINGS
# =========================================================

warnings.filterwarnings("ignore")

# =========================================================
# DOWNLOAD NLTK DATA
# =========================================================

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("omw-1.4", quiet=True)

# =========================================================
# LOAD SPACY MODEL
# =========================================================

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    print("SpaCy model not found.")
    print("Run:")
    print("python -m spacy download en_core_web_sm")

# =========================================================
# GLOBAL OBJECTS
# =========================================================

STOP_WORDS = set(stopwords.words("english"))

STEMMER = PorterStemmer()

LEMMATIZER = WordNetLemmatizer()

# =========================================================
# DATASET PATH
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "..",
    "datasets",
    "amazon.csv"
)

# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(path=DATASET_PATH):
    """
    Load Amazon Reviews Dataset.

    Returns:
        pandas.DataFrame
    """

    df = pd.read_csv(path)

    return df


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    """
    Clean raw text.

    Steps
    -----
    1. Lowercase
    2. Remove URLs
    3. Remove HTML
    4. Remove punctuation
    5. Remove numbers
    6. Remove extra spaces
    """

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.lower()

    # remove urls
    text = re.sub(r"https?://\S+", "", text)

    # remove html
    text = re.sub(r"<.*?>", "", text)

    # remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # remove numbers
    text = re.sub(r"\d+", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# PREPROCESS DATASET
# =========================================================

def preprocess_dataframe(df):
    """
    Create cleaned review column.
    """

    df = df.copy()

    df["clean_review"] = (
        df["review_content"]
        .fillna("")
        .apply(clean_text)
    )

    return df

# =========================================================
# TOKENIZATION
# =========================================================

def tokenize_text(text):
    """
    Tokenize text into words.
    """

    if not text:
        return []

    return word_tokenize(text)


# =========================================================
# STOPWORD REMOVAL
# =========================================================

def remove_stopwords(tokens):
    """
    Remove English stopwords.
    """

    return [
        word
        for word in tokens
        if word not in STOP_WORDS
    ]


# =========================================================
# STEMMING
# =========================================================

def stem_tokens(tokens):
    """
    Apply Porter Stemming.
    """

    return [
        STEMMER.stem(word)
        for word in tokens
    ]


# =========================================================
# LEMMATIZATION
# =========================================================

def lemmatize_tokens(tokens):
    """
    Lemmatize words.
    """

    return [
        LEMMATIZER.lemmatize(word)
        for word in tokens
    ]


# =========================================================
# POS TAGGING
# =========================================================

def pos_tagging(tokens):
    """
    Perform Part-of-Speech tagging.
    """

    return pos_tag(tokens)


# =========================================================
# COMPLETE NLP PREPROCESSING
# =========================================================

def process_text(text):
    """
    Complete NLP preprocessing pipeline.

    Returns dictionary containing
    cleaned text,
    tokens,
    filtered tokens,
    stemmed words,
    lemmatized words,
    POS tags.
    """

    cleaned = clean_text(text)

    tokens = tokenize_text(cleaned)

    filtered = remove_stopwords(tokens)

    stemmed = stem_tokens(filtered)

    lemmatized = lemmatize_tokens(filtered)

    pos_tags = pos_tagging(lemmatized)

    return {
        "clean_text": cleaned,
        "tokens": tokens,
        "filtered_tokens": filtered,
        "stemmed_tokens": stemmed,
        "lemmatized_tokens": lemmatized,
        "pos_tags": pos_tags
    }


# =========================================================
# APPLY PIPELINE TO DATAFRAME
# =========================================================

def build_nlp_features(df):
    """
    Apply NLP preprocessing to all reviews.
    """

    df = preprocess_dataframe(df)

    processed = df["clean_review"].apply(process_text)

    df["tokens"] = processed.apply(
        lambda x: x["tokens"]
    )

    df["filtered_tokens"] = processed.apply(
        lambda x: x["filtered_tokens"]
    )

    df["stemmed_tokens"] = processed.apply(
        lambda x: x["stemmed_tokens"]
    )

    df["lemmatized_tokens"] = processed.apply(
        lambda x: x["lemmatized_tokens"]
    )

    df["pos_tags"] = processed.apply(
        lambda x: x["pos_tags"]
    )

    return df


# =========================================================
# TF-IDF VECTORIZATION
# =========================================================

def tfidf_embeddings(df, max_features=500):
    """
    Generate TF-IDF vectors using stopword-free text.
    """

    corpus = df["filtered_tokens"].apply(lambda x: " ".join(x))

    vectorizer = TfidfVectorizer(
        max_features=max_features
    )

    tfidf_matrix = vectorizer.fit_transform(corpus)

    return vectorizer, tfidf_matrix
# =========================================================
# WORD2VEC EMBEDDINGS
# =========================================================

def train_word2vec(df,
                   vector_size=100,
                   window=5,
                   min_count=2,
                   workers=4):
    """
    Train Word2Vec model using lemmatized tokens.
    """

    sentences = df["lemmatized_tokens"].tolist()

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=42
    )

    return model


def get_word_vector(model, word):
    """
    Return embedding vector for a word.
    """

    if word in model.wv:
        return model.wv[word]

    return None


# =========================================================
# GLOVE MODEL
# =========================================================
def load_glove_model(glove_path):

    if not os.path.exists(glove_path):
        print(f"GloVe file not found: {glove_path}")
        return None

    return KeyedVectors.load_word2vec_format(
        glove_path,
        binary=False,
        no_header=True
    )


def get_glove_vector(model, word):

    if word in model:

        return model[word]

    return None


# =========================================================
# NAMED ENTITY RECOGNITION
# =========================================================

def extract_named_entities(text):
    """
    Extract entities using spaCy.
    """

    doc = nlp(text)

    entities = []

    for ent in doc.ents:

        entities.append(
            {
                "text": ent.text,
                "label": ent.label_
            }
        )

    return entities


def add_named_entities(df):
    """
    Add named entities column.
    """

    df = df.copy()

    df["named_entities"] = df["review_content"].apply(
        extract_named_entities
    )

    return df


# =========================================================
# SENTIMENT ANALYSIS
# =========================================================

def rating_to_sentiment(rating):
    """
    Convert Amazon rating into sentiment.
    """

    try:

        rating = float(rating)

    except:

        return "Unknown"

    if rating >= 3.5:
        return "Positive"
    elif rating >= 2.5:
        return "Neutral"
    else:
        return "Negative"


def sentiment_analysis(df):
    """
    Add sentiment column using ratings.
    """

    df = df.copy()

    df["sentiment"] = df["rating"].apply(
        rating_to_sentiment
    )

    return df


# =========================================================
# SENTIMENT SUMMARY
# =========================================================

def sentiment_distribution(df):
    """
    Return sentiment counts.
    """

    return df["sentiment"].value_counts()


# =========================================================
# WORD FREQUENCY
# =========================================================

def most_common_words(df, top_n=20):

    words = []

    for tokens in df["lemmatized_tokens"]:

        words.extend(tokens)

    freq = pd.Series(words).value_counts()

    return freq.head(top_n)

# =========================================================
# WORD CLOUD
# =========================================================

def generate_wordcloud(df, save_path=None):
    """
    Generate WordCloud from cleaned reviews.
    """

    text = " ".join(
    df["filtered_tokens"]
      .apply(lambda x: " ".join(x))
)

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white"
    ).generate(text)

    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Amazon Reviews")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# =========================================================
# TOP TF-IDF FEATURES
# =========================================================

def get_top_tfidf_features(vectorizer, tfidf_matrix, top_n=20):
    """
    Return top TF-IDF features.
    """

    scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()

    features = vectorizer.get_feature_names_out()

    feature_df = pd.DataFrame({
        "Feature": features,
        "Score": scores
    })

    feature_df = feature_df.sort_values(
        by="Score",
        ascending=False
    )

    return feature_df.head(top_n)


# =========================================================
# SENTIMENT VISUALIZATION
# =========================================================

def plot_sentiment_distribution(df):

    counts = df["sentiment"].value_counts()

    plt.figure(figsize=(6,4))

    counts.plot(kind="bar")

    plt.title("Sentiment Distribution")

    plt.xlabel("Sentiment")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.show()


# =========================================================
# MOST COMMON WORDS VISUALIZATION
# =========================================================

def plot_common_words(df, top_n=20):

    words = most_common_words(df, top_n)

    plt.figure(figsize=(10,5))

    words.plot(kind="bar")

    plt.title("Most Common Words")

    plt.xlabel("Words")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.show()


# =========================================================
# COMPLETE PIPELINE
# =========================================================

def run_complete_pipeline():

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    df = load_dataset()

    df = build_nlp_features(df)

    df = sentiment_analysis(df)

    print("\nDataset Shape:", df.shape)

    print("\nSentiment Distribution")

    print(sentiment_distribution(df))

    print("\nTraining Word2Vec...")

    model = train_word2vec(df)

    print("Vocabulary Size:", len(model.wv.index_to_key))

    vectorizer, tfidf = tfidf_embeddings(df)

    print("\nTop TF-IDF Features")

    print(get_top_tfidf_features(vectorizer, tfidf))

    return df


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    run_complete_pipeline()