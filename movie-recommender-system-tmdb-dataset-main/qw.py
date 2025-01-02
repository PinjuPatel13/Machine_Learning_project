import pickle
import requests
import streamlit as st
import time

# Function to fetch poster for the movie
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    attempt = 0
    retries = 3
    delay = 2
    max_delay = 16
    while attempt < retries:
        try:
            time.sleep(delay)
            response = requests.get(url, timeout=10)  # Timeout to avoid hanging indefinitely
            response.raise_for_status()  # Raise an error for bad status codes (e.g., 404 or 500)
            data = response.json()
            poster_path = data.get('poster_path', None)
            if poster_path:
                full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
                return full_path
            else:
                return "https://via.placeholder.com/500"  # Fallback if poster_path is None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching poster for movie {movie_id}: {e}")
            attempt += 1
            if attempt < retries:
                print(f"Retrying in {delay} seconds... (Attempt {attempt}/{retries})")
                delay = min(delay * 2, max_delay)  # Exponential backoff
                time.sleep(delay)
            else:
                return "https://via.placeholder.com/500"  # Fallback image after max retries

# Function to fetch detailed movie information
def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    response = requests.get(url)
    data = response.json()
    title = data['title']
    overview = data['overview']
    release_date = data['release_date']
    poster_path = "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    return title, overview, release_date, poster_path

# Function to fetch the movie's recommendation
def recommend(movie, movies, similarity):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    recommended_movie_links = []  # To store the movie links
    for i in distances[1:6]:
        # fetch the movie ID and other details
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)
        recommended_movie_links.append(f"https://www.themoviedb.org/movie/{movie_id}")  # Get the movie link

    return recommended_movie_names, recommended_movie_posters, recommended_movie_links


# Streamlit interface
st.header('Movie Recommender System')

# Load movie data and similarity matrix
movies = pickle.load(open('movie_list.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

# List of movies for selection
movie_list = movies['title'].values

# Movie selection
selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

# Display selected movie's details
if selected_movie:
    movie_id = movies[movies['title'] == selected_movie].movie_id.values[0]
    title, overview, release_date, poster_url = fetch_movie_details(movie_id)
    
    st.subheader(f"Movie Details: {title}")
    st.image(poster_url, width=150)  # Display image with fixed width (150px)
    st.write(f"**Overview:** {overview}")
    st.write(f"**Release Date:** {release_date}")

    # Button to show movie recommendations
    if st.button('Show Recommendation'):
        recommended_movie_names, recommended_movie_posters, recommended_movie_links = recommend(selected_movie, movies, similarity)
        
        # Display recommended movies in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        for i, col in enumerate([col1, col2, col3, col4, col5]):
            with col:
                st.text(recommended_movie_names[i])
                st.markdown(f"[![{recommended_movie_names[i]}]({recommended_movie_posters[i]})]({recommended_movie_links[i]})", unsafe_allow_html=True)
