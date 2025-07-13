import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
import nltk
from io import BytesIO
import requests
import numpy as np

# --- NLTK Data Downloads ---
@st.cache_resource
def download_nltk_data():
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)

download_nltk_data()

# --- Streamlit App Starts Here ---
st.title("🤖 Sentiment Analysis of Bitcoin Comments Dashboard")
st.sidebar.title("Navigation & Filters")
st.markdown("""
    ### Interactive Bitcoin Comment Analysis Dashboard
    Jelajahi pola sentimen dari komentar terkait Bitcoin dengan visualisasi interaktif.
    """)
st.sidebar.markdown("Filter data dan konfigurasikan visualisasi:")

# --- Definisi Nama Kolom ---
TEXT_COL = 'Content'  # Kolom komentar asli
PREPROCESSED_TEXT_COL = 'preprocessed_content'  # Kolom komentar yang sudah diproses
SENTIMENT_COL = 'sentiment_preprocessed_content'  # Kolom label sentimen

# --- Fungsi untuk Memuat Data ---
@st.cache_data(persist=True)
def load_data():
    try:
        # Ganti dengan path file Anda atau pastikan file diupload di Streamlit
        data = pd.read_csv("twitter_data_labeled.csv")
    except FileNotFoundError:
        st.error("ERROR: File 'twitter_data_labeled.csv' tidak ditemukan.")
        st.stop()

    if PREPROCESSED_TEXT_COL not in data.columns:
        st.error(f"Kolom '{PREPROCESSED_TEXT_COL}' tidak ditemukan di dataset.")
        st.stop()
    if SENTIMENT_COL not in data.columns:
        st.error(f"Kolom '{SENTIMENT_COL}' tidak ditemukan di dataset.")
        st.stop()

    data[PREPROCESSED_TEXT_COL] = data[PREPROCESSED_TEXT_COL].astype(str)
    data[SENTIMENT_COL] = data[SENTIMENT_COL].astype(str)
    
    # Filter 'error' sentiment
    original_rows = len(data)
    data = data[data[SENTIMENT_COL] != 'error']
    if len(data) < original_rows:
        st.warning(f"Menghapus {original_rows - len(data)} baris dengan sentimen 'error'.")
    
    return data

data = load_data()

# --- Struktur Tab Dashboard  ---
tab1, tab2 = st.tabs(["Overview", "Text Analysis"])

# ========== Tab 1: Overview ==========

with tab1:
    st.header("Statistik Umum")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Sentimen Komentar")
        viz_type = st.selectbox('Pilih jenis visualisasi', ['Bar plot', 'Pie chart'])
        sentiment_count = data[SENTIMENT_COL].value_counts().reset_index()
        sentiment_count.columns = ['Sentimen', 'Jumlah']
        
        if viz_type == 'Bar plot':
            fig = px.bar(sentiment_count, x='Sentimen', y='Jumlah', color='Sentimen', height=400, title="Distribusi Sentimen Keseluruhan")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.pie(sentiment_count, values='Jumlah', names='Sentimen', title="Distribusi Sentimen Keseluruhan")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Contoh Komentar Asli per Sentimen")
        selected_sentiment_for_comments = st.selectbox("Pilih sentimen untuk melihat komentar:", data[SENTIMENT_COL].unique())
        num_comments_to_show = st.slider("Jumlah komentar yang akan ditampilkan", 1, 10, 5)

        if selected_sentiment_for_comments:
            top_comments_df = data[data[SENTIMENT_COL] == selected_sentiment_for_comments][TEXT_COL].head(num_comments_to_show).reset_index(drop=True)
            if not top_comments_df.empty:
                for i, comment in enumerate(top_comments_df):
                    st.write(f"**{i+1}.** {comment}")
            else:
                st.info("Tidak ada komentar ditemukan untuk sentimen yang dipilih setelah filter.")
        else:
            st.info("Pilih sentimen untuk menampilkan komentar.")

# ========== Tab 2: Text Analysis ==========

with tab2:
    st.header("Analisis Tekstual")
    
    st.subheader("Word Cloud (Kata Unik per Sentimen)")
    top_n_words_wc = st.slider("Jumlah kata teratas yang akan dipertimbangkan untuk word cloud unik", 50, 200, 100)
    colormap_wc = st.selectbox("Pilih tema warna untuk word cloud", ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])

    all_sentiment_labels = ['positive', 'neutral', 'negative']
    sentiment_words = {label: Counter() for label in all_sentiment_labels}

    # Mengumpulkan frekuensi kata untuk setiap sentimen
    for label in all_sentiment_labels:
        text_list = data[data[SENTIMENT_COL] == label][PREPROCESSED_TEXT_COL].dropna().tolist()
        if text_list:
            full_text = ' '.join(text_list)
            sentiment_words[label].update(full_text.split())

    top_words_per_sentiment = {label: [word for word, count in counter.most_common(top_n_words_wc)]
                               for label, counter in sentiment_words.items()}

    unique_words_for_wc = {label: set() for label in all_sentiment_labels}
    unique_words_for_wc['positive'].update(top_words_per_sentiment['positive'])

    for word in top_words_per_sentiment['negative']:
        if word not in unique_words_for_wc['positive']:
            unique_words_for_wc['negative'].add(word)

    for word in top_words_per_sentiment['neutral']:
        if word not in unique_words_for_wc['positive'] and word not in unique_words_for_wc['negative']:
            unique_words_for_wc['neutral'].add(word)

    for label in all_sentiment_labels:
        st.subheader(f"Word Cloud untuk Sentimen {label.capitalize()}")
        words_to_use = unique_words_for_wc[label]
        filtered_counter = Counter({word: count for word, count in sentiment_words[label].items() if word in words_to_use})
        
        if not filtered_counter:
            st.info(f"Tidak ada kata unik yang tersisa untuk sentimen '{label}'.")
            continue

        wc = WordCloud(width=800, height=400, background_color='white',
                       collocations=False, min_font_size=10,
                       colormap=colormap_wc).generate_from_frequencies(filtered_counter)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        st.download_button(
            label=f"Unduh Word Cloud {label.capitalize()}",
            data=buf.getvalue(),
            file_name=f"{label}_wordcloud.png",
            mime="image/png",
            key=f"download_wc_{label}"
        )
        plt.close(fig)

# --- Penambahan Sidebar ---
st.sidebar.markdown("---")
st.sidebar.header("Ekspor Data")
if st.sidebar.button("Unduh Komentar Berlabel sebagai CSV"):
    csv = data.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Unduh CSV",
        data=csv,
        file_name="bitcoin_comments_labeled.csv",
        mime="text/csv"
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Tentang Penulis")

image_url = "https://avatars.githubusercontent.com/u/98022263?v=4"
try:
    response = requests.get(image_url)
    response.raise_for_status()
    image = response.content
    st.sidebar.image(image, caption="Maulana Imanulhaq (Fancyyy21)")
except requests.exceptions.RequestException as e:
    st.sidebar.error(f"Gagal memuat gambar profil: {e}")

st.sidebar.markdown(
    """
    Aplikasi ini dibangun dengan ❤️ oleh **Maulana Imanulhaq (Fancyyy21)**. 
    Anda dapat terhubung dengan saya di: [LinkedIn](https://www.linkedin.com/in/maulana-imanulhaq-38992a201/)
    """
)
st.sidebar.info("""
    **Fitur Dashboard:**
    - Plot distribusi sentimen interaktif
    - Word cloud yang dapat disesuaikan dengan kata-kata unik per sentimen
    - Kemampuan ekspor data
    - Menampilkan contoh komentar asli teratas per sentimen
""")
