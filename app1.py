import streamlit as st
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.corpus import sentiwordnet as swn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import train_test_split

# Import and download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Function to get sentiment from TextBlob
def get_sentiment_textblob(text):
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return pd.Series([0.0, 'neutral'])
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            sentiment = 'positive'
        elif polarity < 0:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        return pd.Series([polarity, sentiment])
    except Exception as e:
        return pd.Series([np.nan, 'error'])

# Function to get sentiment from VADER
def get_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    if vs['compound'] >= 0.05:
        sentiment = 'positive'
    elif vs['compound'] <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    return vs['compound'], sentiment

# Function to get sentiment from SentiWordNet
def get_sentiment_sentiwordnet(text):
    senti_val = []
    postagging = nltk.pos_tag(nltk.word_tokenize(text))  # Tokenize and POS tag the string

    for word, tag in postagging:
        pos_tag = get_wordnet_pos(tag)
        if pos_tag:  # Check if a valid WordNet POS tag was found
            synsets = list(swn.senti_synsets(word, pos_tag))
            if not synsets:
                # Try with lemmatized word if no synsets found for the original word
                lemmatized_word = lemmatizer.lemmatize(word, pos_tag)
                synsets = list(swn.senti_synsets(lemmatized_word, pos_tag))

            if synsets:
                score = synsets[0].pos_score() - synsets[0].neg_score()
                senti_val.append(score)

    # Calculate the average sentiment score for the review
    avg_score = sum(senti_val) / len(senti_val) if senti_val else 0

    # Classify sentiment based on the score for SentiWordNet
    if avg_score >= 0.05:
        sentiment = 'Positive'
    elif avg_score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return avg_score, sentiment

# Sidebar elements
st.sidebar.title("Sentiment Analysis Dashboard")
sentiment_method = st.sidebar.selectbox("Choose Sentiment Analysis Method", ["TextBlob", "VADER", "SentiWordNet"])
export_button = st.sidebar.button("Ekspor Data")

# Main title
st.title("Sentiment Analysis of Bitcoin Comments Dashboard")

# Load dataset
df_raw = pd.read_csv("twitter_data_labeled.csv")

# Sentiment Analysis methods
if sentiment_method == "TextBlob":
    st.header("TextBlob Sentiment Analysis")
    df_raw[['polarity', 'sentiment_textblob']] = df_raw['sentiment_textblob_preprocessed_content'].apply(get_sentiment_textblob)
elif sentiment_method == "VADER":
    st.header("VADER Sentiment Analysis")
    df_raw[['polarity', 'sentiment_vader']] = df_raw['sentiment_vader_preprocessed_content'].apply(get_sentiment_vader)
elif sentiment_method == "SentiWordNet":
    st.header("SentiWordNet Sentiment Analysis")
    df_raw[['polarity', 'sentiment_sentiwordnet']] = df_raw['sentiment_sentiwordnet_preprocessed_content'].apply(get_sentiment_sentiwordnet)

# Overview section
st.subheader("Overview")

# Sentiment distribution
sentiment_counts = df_raw['sentiment_textblob'].value_counts() if sentiment_method == "TextBlob" else (
    df_raw['sentiment_vader'].value_counts() if sentiment_method == "VADER" else df_raw['sentiment_sentiwordnet'].value_counts()
)
st.write(sentiment_counts)

# Display a pie chart for sentiment distribution
fig, ax = plt.subplots()
sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax)
ax.set_ylabel('')
st.pyplot(fig)

# Display 5 example comments per sentiment
st.subheader("5 Example Comments Based on Sentiment")
sentiment = st.radio("Select Sentiment", ["positive", "neutral", "negative"])
example_comments = df_raw[df_raw['sentiment_textblob'] == sentiment].head(5) if sentiment_method == "TextBlob" else (
    df_raw[df_raw['sentiment_vader'] == sentiment].head(5) if sentiment_method == "VADER" else df_raw[df_raw['sentiment_sentiwordnet'] == sentiment].head(5)
)
st.write(example_comments[['Content', 'sentiment_textblob']].head() if sentiment_method == "TextBlob" else (
    example_comments[['Content', 'sentiment_vader']].head() if sentiment_method == "VADER" else example_comments[['Content', 'sentiment_sentiwordnet']].head()
))

# Text Analysis Section - WordClouds
st.subheader("Text Analysis: WordCloud")

sentiment_words = {'positive': Counter(), 'neutral': Counter(), 'negative': Counter()}
for idx, row in df_raw.iterrows():
    sentiment = row['sentiment_textblob'] if sentiment_method == "TextBlob" else (
        row['sentiment_vader'] if sentiment_method == "VADER" else row['sentiment_sentiwordnet']
    )
    sentiment_words[sentiment].update(row['Content'].split())

# Create wordclouds for each sentiment
for sentiment in ['positive', 'neutral', 'negative']:
    wc = WordCloud(width=800, height=400, background_color='white', collocations=False).generate_from_frequencies(sentiment_words[sentiment])
    st.subheader(f"WordCloud for {sentiment.capitalize()} Sentiment")
    st.image(wc.to_array())

# Export data
if export_button:
    df_raw.to_csv("exported_data.csv", index=False)
    st.success("Data has been exported successfully!")
