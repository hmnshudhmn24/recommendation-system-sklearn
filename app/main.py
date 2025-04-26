import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Load data
movies = pd.read_csv('data/movies.csv')

# TF-IDF vectorization of genres (content-based)
tfidf = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations based on your favorite title and genre similarity!")

movie_titles = movies['title'].tolist()
selected_movie = st.selectbox("Choose a movie you like:", movie_titles)

# Recommendation function
def recommend(movie_title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Show recommendations
if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.subheader("📽️ Recommended Movies:")
    for rec in recommendations:
        st.write(f"- {rec}")
