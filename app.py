import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from io import BytesIO
import nltk
import requests

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
TEXT_COL = 'cleaned_raw_content'  # Kolom komentar asli
PREPROCESSED_TEXT_COL = 'preprocessed_content'  # Kolom komentar yang sudah diproses
SENTIMENT_COL_TEXBLOB = 'sentiment_textblob_preprocessed_content'  # Kolom sentiment untuk TextBlob
SENTIMENT_COL_VADER = 'sentiment_vader_preprocessed_content'  # Kolom sentiment untuk Vader
SENTIMENT_COL_SENTIWORDNET = 'sentiment_sentiwordnet_preprocessed_content'  # Kolom sentiment untuk SentiWordNet

# --- Fungsi untuk Memuat Data ---
@st.cache_data(persist=True)
def load_data(dataset_type='texblob'):
    """
    Function to load the dataset based on the selected sentiment analysis method.
    """
    try:
        if dataset_type == 'texblob':
            data = pd.read_csv("twitter_data_labeled.csv")  # Replace with actual path for TexBlob
        elif dataset_type == 'vader':
            data = pd.read_csv("labeled_data.csv")  # Replace with actual path for Vader
        elif dataset_type == 'sentiwordnet':
            data = pd.read_csv("labeled_data.csv")  # Replace with actual path for SentiWordNet
    except FileNotFoundError:
        st.error(f"ERROR: File for {dataset_type} not found.")
        st.stop()
    
    return data

# --- Struktur Tab Dashboard ---
tab1, tab2 = st.tabs(["Overview", "Text Analysis"])

# ========== Tab 1: Overview ==========
with tab1:
    st.header("Statistik Umum")
    
    # --- Pilihan untuk distribusi sentimen ---
    st.subheader("Distribusi Sentimen Bitcoin Comments")
    
    # --- Sentiment Analysis Method Selection in Main Content ---
    method = st.selectbox("Pilih Metode Analisis Sentimen", ['TexBlob', 'Vader', 'SentiWordNet'])

    # Load dataset based on the sentiment method selected
    data = load_data(dataset_type=method.lower())

    # Choose the appropriate sentiment column based on selected method
    if method == 'TexBlob':
        sentiment_column = SENTIMENT_COL_TEXBLOB
    elif method == 'Vader':
        sentiment_column = SENTIMENT_COL_VADER
    else:  # SentiWordNet
        sentiment_column = SENTIMENT_COL_SENTIWORDNET
    
    # Group by sentiment and count
    sentiment_count = data[sentiment_column].value_counts().reset_index()
    sentiment_count.columns = ['Sentiment', 'Count']

    # Pilihan untuk visualisasi
    chart_type = st.selectbox("Pilih jenis visualisasi", ['Bar chart', 'Pie chart'])

    if chart_type == 'Bar chart':
        # Plotly bar chart for sentiment distribution with hover information
        fig = px.bar(sentiment_count, 
                     x='Sentiment', 
                     y='Count', 
                     color='Sentiment', 
                     hover_data={'Sentiment': True, 'Count': True}, 
                     labels={'Sentiment': 'Sentiment Type', 'Count': 'Number of Comments'}, 
                     title=f"Distribusi Sentimen {method} Bitcoin Comments",
                     height=400)

        fig.update_layout(xaxis_title='Sentiment', yaxis_title='Number of Comments')
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == 'Pie chart':
        # Plotly pie chart for sentiment distribution with hover information
        fig = px.pie(sentiment_count, 
                     values='Count', 
                     names='Sentiment', 
                     hover_data={'Sentiment': True, 'Count': True}, 
                     title=f"Distribusi Sentimen {method} Bitcoin Comments")

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Contoh Komentar Asli per Sentimen")
    selected_sentiment_for_comments = st.selectbox("Pilih sentimen untuk melihat komentar:", data[sentiment_column].unique())
    num_comments_to_show = st.slider("Jumlah komentar yang akan ditampilkan", 1, 10, 5)

    if selected_sentiment_for_comments:
        top_comments_df = data[data[sentiment_column] == selected_sentiment_for_comments][TEXT_COL].head(num_comments_to_show).reset_index(drop=True)
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
    
    # No need for the sentiment method selection here anymore
    st.subheader("Word Cloud (Kata Unik)")

    top_n_words_wc = st.slider("Jumlah kata teratas yang akan dipertimbangkan untuk word cloud unik", 50, 200, 100)
    colormap_wc = st.selectbox("Pilih tema warna untuk word cloud", ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])

    # Load the dataset independently for text analysis
    data_text_analysis = load_data(dataset_type='texblob')  # Load default method or based on preference

    all_sentiment_labels = ['positive', 'neutral', 'negative']
    sentiment_words = {label: Counter() for label in all_sentiment_labels}

    # Collect word frequencies for each sentiment in the text analysis tab (independent of method)
    for label in all_sentiment_labels:
        text_list = data_text_analysis[data_text_analysis[SENTIMENT_COL_TEXBLOB] == label][PREPROCESSED_TEXT_COL].dropna().tolist()
        if text_list:
            full_text = ' '.join(text_list)
            sentiment_words[label].update(full_text.split())

    # Get top N words for each sentiment label
    top_words_per_sentiment = {label: [word for word, count in counter.most_common(top_n_words_wc)]
                               for label, counter in sentiment_words.items()}

    # Create unique words per sentiment (to avoid duplicates across labels)
    unique_words_for_wc = {label: set() for label in all_sentiment_labels}
    unique_words_for_wc['positive'].update(top_words_per_sentiment['positive'])

    for word in top_words_per_sentiment['negative']:
        if word not in unique_words_for_wc['positive']:
            unique_words_for_wc['negative'].add(word)

    for word in top_words_per_sentiment['neutral']:
        if word not in unique_words_for_wc['positive'] and word not in unique_words_for_wc['negative']:
            unique_words_for_wc['neutral'].add(word)

    # Generate WordCloud for each sentiment
    for label in all_sentiment_labels:
        st.subheader(f"Word Cloud untuk Sentimen {label.capitalize()}")  # Removed method from title
        
        # Ensure there is data for this sentiment
        words_to_use = unique_words_for_wc[label]
        filtered_counter = Counter({word: count for word, count in sentiment_words[label].items() if word in words_to_use})

        if len(filtered_counter) == 0:
            st.warning(f"Tidak ada kata untuk ditampilkan pada word cloud untuk sentimen {label.capitalize()}.")
            st.write(f"Sentiment data for {label}: {sentiment_words[label]}")  # Debugging line
            continue  # Skip to the next sentiment

        # Generate WordCloud for all available words
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
    - Plot distribusi sentimen interaktif dengan pilihan metode analisis (TexBlob, Vader, SentiWordNet)
    - Kemampuan ekspor data
    - Menampilkan contoh komentar asli teratas per sentimen
""")
