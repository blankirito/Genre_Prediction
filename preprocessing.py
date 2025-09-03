import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def feature_engineering(df):
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['release_year'] = df['date_added'].dt.year
    df['title_length'] = df['title'].apply(lambda x: len(str(x)))
    df['description_word_count'] = df['description'].apply(lambda x: len(str(x).split()))
    df['release_year'].fillna(df['release_year'].mode()[0], inplace=True)
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def apply_clean_text(df):
    df['clean_description'] = df['description'].apply(clean_text)
    return df

def filter_and_encode_genres(df, min_genre_count=50):
    df['main_genre'] = df['listed_in'].apply(lambda x: x.split(',')[0].strip())
    genre_counts = df['main_genre'].value_counts()
    valid_genres = genre_counts[genre_counts >= min_genre_count].index
    df_filtered = df[df['main_genre'].isin(valid_genres)].copy().reset_index(drop=True)

    le = LabelEncoder()
    df_filtered['genre_label'] = le.fit_transform(df_filtered['main_genre'])

    return df_filtered, le

def generate_tfidf_matrix(df_filtered):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_filtered['clean_description'])
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    return tfidf_matrix, tfidf
