import time
import requests
import pickle
import streamlit as st

# Function to fetch poster with error handling and retry mechanism
def fetch_poster(movie_id, retries=3, delay=2):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    attempt = 0
    while attempt < retries:
        try:
            time.sleep(delay)  # Add a 2-second delay between requests
            response = requests.get(url, timeout=10)  # Add timeout to avoid hanging indefinitely
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
                time.sleep(delay)
            else:
                return "https://via.placeholder.com/500"  # Fallback image after max retries

# Function to fetch movie link
def fetch_movie_link(movie_id):
    return f"https://www.themoviedb.org/movie/{movie_id}"

# Function to recommend movies based on similarity
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    recommended_movie_links = []  # To store movie links
    for i in distances[1:6]:  # Get top 5 similar movies
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)
        recommended_movie_links.append(fetch_movie_link(movie_id))  # Get the movie link

    return recommended_movie_names, recommended_movie_posters, recommended_movie_links



# Streamlit code to display the recommendation system
st.header('Movie Recommender System')

# Load the movies data and similarity matrix from pickle files
movies = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# List of movie titles for the dropdown
movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

# Static image URL (just for display purposes)
static_image_url = "https://img.freepik.com/free-photo/abstract-luxury-plain-blur-grey-black-gradient-used-as-background-studio-wall-display-your-p_1258-112144.jpg?t=st=1735639467~exp=1735643067~hmac=083dfa816a1b0947c413c6f4c6caccb079f68aac15641f11814217c81ce80a7a&w=1380"

# Button to show recommendations
if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters, recommended_movie_links = recommend(selected_movie)
    
    # Create columns to display movie recommendations
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Make the image a clickable link
        st.text(recommended_movie_names[0])
        st.markdown(f"[![{recommended_movie_names[0]}]({recommended_movie_posters[0]})]({recommended_movie_links[0]})", unsafe_allow_html=True)
    
    with col2:
        st.text(recommended_movie_names[1])
        st.markdown(f"[![{recommended_movie_names[1]}]({recommended_movie_posters[1]})]({recommended_movie_links[1]})", unsafe_allow_html=True)
    
    with col3:
        st.text(recommended_movie_names[2])
        st.markdown(f"[![{recommended_movie_names[2]}]({recommended_movie_posters[2]})]({recommended_movie_links[2]})", unsafe_allow_html=True)
    
    with col4:
        st.text(recommended_movie_names[3])
        st.markdown(f"[![{recommended_movie_names[3]}]({recommended_movie_posters[3]})]({recommended_movie_links[3]})", unsafe_allow_html=True)
    
    with col5:
        st.text(recommended_movie_names[4])
        st.markdown(f"[![{recommended_movie_names[4]}]({recommended_movie_posters[4]})]({recommended_movie_links[4]})", unsafe_allow_html=True)
