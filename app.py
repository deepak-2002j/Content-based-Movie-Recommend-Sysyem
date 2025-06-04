import pickle

import pandas as pd
import streamlit as st
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend_chat(movie):
    idx = movies_new[movies_new["title"] == movie].index[0]
    similarity_scores = list(enumerate(similarity[idx]))
    top_matches = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]

    titles = [movies_new.iloc[i[0]].title for i in top_matches]
    posters = [fetch_poster(movies_new.iloc[i[0]].movie_id) for i in top_matches]

    return titles,posters


st.header('Movie Recommender System')
movies_app = pickle.load(open('movie_list_dict.pkl','rb'))
movies_new = pd.DataFrame(movies_app)
# similarity = pickle.load(open('similarity.pkl','rb'))
# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies_new["tags"]).toarray()

# Similarity Matrix
similarity = cosine_similarity(vectors)
movie_list = movies_new['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names,recommended_movie_posters = recommend_chat(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(recommended_movie_posters[0])
        st.text(recommended_movie_names[0])
    with col2:
        st.image(recommended_movie_posters[1])
        st.text(recommended_movie_names[1])
    with col3:
        st.image(recommended_movie_posters[2])
        st.text(recommended_movie_names[2])
    with col4:
        st.image(recommended_movie_posters[3])
        st.text(recommended_movie_names[3])
    with col5:
        st.image(recommended_movie_posters[4])
        st.text(recommended_movie_names[4])
        
