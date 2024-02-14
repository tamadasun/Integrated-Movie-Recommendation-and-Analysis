import streamlit as st
import pandas as pd

from movie_recommendation import recommend_based_on_user_preference, get_movie_recommendations, \
    interactive_movie_recommendation, get_wordnet_pos, lemmatize_text, apply_tfidf_vectorizer, \
    calculate_similarity, get_similar_movies, item_similarity_matrix, interaction_data_pivot

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import nltk
from nltk.corpus import wordnet


tmdb_df = pd.read_csv('../data/tmdb_data.csv')

def display_movie_info(movie_data):
    if movie_data.empty:
        st.warning("Movie not found in the dataset. Try using advanced preferences.")
        return
    
    for index, movie in movie_data.iterrows():
        st.write(f"Title: {movie['title']}")
        st.write(f"Release Year: {movie['release_year']}")
        st.write(f"Genres: {movie['genre_names']}")
        st.write(f"Overview: {movie['overview']}")
        st.write(f"Popularity: {movie['popularity']}")
        
        poster_url = f"https://image.tmdb.org/t/p/original{movie['poster_path']}"
        
        st.image(poster_url, caption="Poster", use_column_width=True)
        
        st.write("---")

def app():
    st.title("The Cinematic Nexus: Unveiling the Future of Movie Recommendations and Analysis")

    # Welcome Section
    st.write(
        "Welcome to The Cinematic Nexus! Discover your next favorite movie with our advanced recommendation system."
    )

    # Sidebar
    st.sidebar.header("User Preferences")
    user_input_movie = st.sidebar.text_input("Enter your favorite movie:")

    if st.sidebar.button("Get Recommendations"):
        if user_input_movie:
            try:
                recommended_movies = get_movie_recommendations(user_input_movie, item_similarity_matrix, interaction_data_pivot)
                st.subheader(f"Recommended Movies based on '{user_input_movie}':")
                display_movie_info(tmdb_df[tmdb_df['title'].isin(recommended_movies[:5])])
            except KeyError:
                st.warning("Movie not found in the dataset. Please enter a new movie.")
                return

    st.sidebar.header("User Preferences (Advanced)")
    user_preference = st.sidebar.text_input("Enter your favorite genre, director, or actor/actress:")
    start_year = st.sidebar.number_input("Enter the starting year:", 1900, 2022, 1900)
    end_year = st.sidebar.number_input("Enter the ending year:", 1900, 2022, 2022)

    if st.sidebar.button("Advanced Recommendations"):
        if user_preference:
            try:
                filtered_movies = tmdb_df[
                    (tmdb_df['genre_names'].str.lower().str.contains(user_preference.lower())) |
                    (tmdb_df['directors'].str.lower().str.contains(user_preference.lower())) |
                    (tmdb_df['cast'].str.lower().str.contains(user_preference.lower())) &
                    (tmdb_df['release_year'].between(start_year, end_year))
                ]

                if not filtered_movies.empty:
                    sorted_movies = filtered_movies.sort_values(by='popularity_normalized', ascending=False)
                    recommended_movies = sorted_movies['title'].tolist()
                    st.subheader("Recommended Movies based on Advanced Preferences:")
                    display_movie_info(tmdb_df[tmdb_df['title'].isin(recommended_movies[:5])])
                else:
                    st.warning("No movies found based on your advanced preferences.")
            except KeyError:
                st.warning("Movie not found in the dataset. Please enter a new movie.")
                return

if __name__ == "__main__":
    app()
