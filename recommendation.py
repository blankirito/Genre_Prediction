import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_cosine_similarity_matrix(tfidf_matrix):
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print("Cosine Similarity Matrix Created")
    return cosine_sim


def get_recommendation(title, df, cosine_sim):
    indices = pd.Series(df.index, index=df['title'].str.lower())
    title = title.lower()
    
    if title not in indices:
        return f"'{title}' not found in the dataset."
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

    recommended_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[recommended_indices]


def get_recommendation_filtered(title, df, cosine_sim, top_n=10, genre_filter=None, type_filter=None):
    indices = pd.Series(df.index, index=df['title'].str.lower())
    title = title.lower()
    
    if title not in indices:
        return f"'{title}' not found in the dataset."
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]

    sim_df = pd.DataFrame(sim_scores, columns=['index', 'score'])
    sim_df = df.loc[sim_df['index']].copy()
    sim_df['similarity'] = sim_df.index.map(lambda i: cosine_sim[idx][i])

    if genre_filter:
        sim_df = sim_df[sim_df['main_genre'].str.lower() == genre_filter.lower()]
    if type_filter:
        sim_df = sim_df[sim_df['type'].str.lower() == type_filter.lower()]

    return sim_df[['title', 'type', 'main_genre', 'similarity']].head(top_n)


def predict_genre_from_text(text, model, vectorizer, label_encoder):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return label_encoder.inverse_transform(prediction)[0]
