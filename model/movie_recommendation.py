import pandas as pd
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

#make functions reusable as modules 

def recommend_based_on_user_preference(user_preference, tmdb_df):
    # Split user input
    preferences = [preference.strip() for preference in user_preference.split(',')]

    # Filter movies based on user preferences in directors and cast columns
    filtered_movies = tmdb_df[
        tmdb_df['directors'].apply(lambda x: any(pref.lower() in str(x).lower() for pref in preferences)) |
        tmdb_df['cast'].apply(lambda x: any(pref.lower() in str(x).lower() for pref in preferences))
    ]

    # Sort movies by popularity (can change for other metric)
    sorted_movies = filtered_movies.sort_values(by='popularity_normalized', ascending=False)

    # Extract recommended movie titles
    recommended_movies = sorted_movies['title'].tolist()

    return recommended_movies

item_similarity = cosine_similarity(sparse_interaction_matrix.T, dense_output=False)
# interaction_data_pivot is user interaction data
item_similarity_matrix = cosine_similarity(interaction_data_pivot.fillna(0))

# Function to get movie recommendations based on item similarity
def get_movie_recommendations(movie_title, item_similarity_matrix, interaction_data_pivot):
    """
    This function takes a movie title, an item similarity matrix, and a interaction_data_pivot as input.
    It returns a list of movie recommendations based on the item similarity of the input movie.
    
    - movie_title (str): The title of the movie for which recommendations are requested.
    - item_similarity_matrix (numpy.ndarray): The item similarity matrix, computed using collaborative filtering
      (e.g., cosine similarity on the interaction_data_pivot ).
    - interaction_data_pivot (pd.DataFrame): The user-item interaction matrix where rows represent movies, and columns represent various features
      like 'director_avg_rating', 'lead_actor_avg_rating', 'popularity_normalized', 'vote_average_normalized', 'vote_count_normalized', 'release_year', and genre indicators.
      The values represent movie features or characteristics.
    
    Returns:
    List[str]: A list of recommended movies based on item similarity. 
    The list is sorted in descending order of similarity.
      
    """
    if movie_title in interaction_data_pivot.index:
        similar_scores = item_similarity_matrix[interaction_data_pivot.index.get_loc(movie_title)]
        similar_movies = list(interaction_data_pivot.index[np.argsort(similar_scores)[::-1]])
        return similar_movies[1:]  # Exclude the input movie itself
    else:
        print(f"Movie '{movie_title}' not found in the dataset.")
        user_preference = input("Enter your preferred actor, genre, or other relevant information: ")
        # Perform recommendation based on user's additional input
        recommendations = recommend_based_on_user_preference(user_preference, tmdb_df)
        return recommendations
    
def interactive_movie_recommendation(tmdb_df):
    """
    This function allows users to input their favorite movie and receive recommendations
    based on their preferences,including genre, director, actor/actress, and release year range.
    
    Parameters:
    - tmdb_df (pd.DataFrame): DataFrame containing movie data, including columns like 'title', 'genre_names', 'directors',
    'cast', 'release_year', 'popularity_normalized', and others.
    
    Returns:
    None
    
    Note: The function utlizes the 'get_movie_recommendations' function, and the 'item_similarity_matrix' computed
    using collaborative filtering (e.g., cosine similarity on the item interaction matrix).
    
    Example usage:
    interactive_movie_recommendation(tmdb_df)
    """
    while True:
        # Prompt the user to enter their favorite movie
        user_input_movie = input("Enter your favorite movie: ")

        # Check if the movie exists in the dataset
        matching_movies = tmdb_df[tmdb_df['title'].str.lower() == user_input_movie.lower()]

        if not matching_movies.empty:
            # If the movie is found, recommend similar movies
            recommended_movies = get_movie_recommendations(user_input_movie, item_similarity_matrix, interaction_data_pivot)
            print(f"\nHere are some recommendations based on '{user_input_movie}':")
            print(recommended_movies[:5])
        else:
            print(f"Movie '{user_input_movie}' not found in the dataset.")
            print("Let's try to find recommendations based on your preferences.")

            while True:
                # Prompt the user for their favorite genre, director, or actor/actress
                user_preference = input("Enter your favorite genre, director, or actor/actress: ")

                # Prompt the user for the desired release year range
                start_year = int(input("Enter the starting year: "))
                end_year = int(input("Enter the ending year: "))

                # Filter movies based on user preferences and release year range
                filtered_movies = tmdb_df[
                    (tmdb_df['genre_names'].apply(lambda x: user_preference.lower() in str(x).lower())) |
                    (tmdb_df['directors'].apply(lambda x: user_preference.lower() in str(x).lower())) |
                    (tmdb_df['cast'].apply(lambda x: user_preference.lower() in str(x).lower())) &
                    (tmdb_df['release_year'].between(start_year, end_year))
                ]
                print("Filtered Movies:")
                print(filtered_movies[['title', 'release_year']])


                # Sort movies by popularity
                sorted_movies = filtered_movies.sort_values(by='popularity_normalized', ascending=False)

                # Extract recommended movie titles
                recommended_movies = sorted_movies['title'].tolist()

                if not recommended_movies:
                    print("No movies found based on your preferences.")
                    break

                print(f"\nHere are some recommendations based on your preferences:")
                print(recommended_movies[:5])

                # user feedback
                user_feedback = input("Do these movies appeal to you? (yes/no): ").lower()

                if user_feedback == 'yes':
                    print("Great! Enjoy watching.")
                    return
                elif user_feedback == 'no':
                    print("Let's try refining your preferences.")
                    continue
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")
                    continue

        #user feedback
        user_feedback = input("Do these movies appeal to you? (yes/no): ").lower()

        if user_feedback == 'yes':
            print("Great! Enjoy watching.")
            break
        elif user_feedback == 'no':
            print("Sorry to hear that. Let's try refining your preferences.")
            continue
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
            continue
            
            
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV, "J": wordnet.ADJ}
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_text(text):
    
    if pd.isnull(text):  # Check for NaN values
        return ''
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    tokens = word_tokenize(text)
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]

    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)

tmdb_df['preprocessed_overview'] = tmdb_df['overview'].apply(lemmatize_text)


def apply_tfidf_vectorizer(data, text_column='preprocessed_overview'):
    """
    Apply TfidfVectorizer to convert preprocessed text data in the specified column of the dataframe to a TF-IDF matrix.
    
    Parameters:
    - data (pd.DataFrame): The input dataframe containing the column with preprocessed text data.
    - text_column (str): The name of the column containing preprocessed text data. Default is 'preprocessed_overview'.
    
    Returns:
    - tuple: A tuple showing the shape of the resulting TF-IDF matrix.
    """
    # Extract the preprocessed text data
    text_data = data[text_column].astype(str)
    
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit and transform the preprocessed text data to TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
    
    #feature_names = tfidf_vectorizer.get_feature_names_out()
    
    return tfidf_matrix, tfidf_vectorizer

tfidf_matrix, tfidf_vectorizer = apply_tfidf_vectorizer(tmdb_df)
# Combine features
combined_features_matrix = combine_features(tmdb_df, tfidf_matrix)

def calculate_similarity(feature_matrix):
    """
    Calculate cosine similarity between movies based on their feature matrix.

    Parameters:
    - feature_matrix (scipy.sparse.csr_matrix): The feature matrix containing TF-IDF and other features.

    Returns:
    - similarity_matrix (numpy.ndarray): The cosine similarity matrix.
    """
    similarity_matrix = cosine_similarity(feature_matrix, dense_output=False)
    return similarity_matrix

#contain the pairwise cosine similarity scores between movies based on feature matrices
similarity_matrix = calculate_similarity(combined_features_matrix)


def get_similar_movies(movie_title, similarity_matrix, data):
    """
    Get a list of similar movies based on a given movie title.

    Parameters:
    - movie_title (str): The title of the movie.
    - similarity_matrix (numpy.ndarray or scipy.sparse.csr_matrix): The cosine similarity matrix.
    - data (pd.DataFrame): The input dataframe containing movie information.

    Returns:
    - similar_movies (list): A list of similar movies.
    """
    movie_index = data[data['title'].str.lower() == movie_title.lower()].index[0]

    if isinstance(similarity_matrix, csr_matrix):
        similarity_matrix = similarity_matrix.toarray()

    similar_scores = similarity_matrix[movie_index]
    similar_movie_indices = similar_scores.argsort()[::-1][1:]  # Exclude the input movie itself
    similar_movies = data.iloc[similar_movie_indices]['title'].tolist()
    return similar_movies

