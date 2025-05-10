# main.py

import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer

# Load and process data
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")
movies = movies.merge(credits)
movies = movies[["genres", "movie_id", "keywords", "overview", "title", "cast", "crew"]]
movies.dropna(inplace=True)

def convert(obj):
    return [i["name"] for i in ast.literal_eval(obj)]

def get_cast(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i["name"])
        if len(l) == 3:
            break
    return l

def get_director(obj):
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            return [i["name"]]
    return []

def remove_space(obj):
    return [i.replace(" ", "") for i in obj]

def stemmer(text):
    snow = SnowballStemmer("english")
    return " ".join([snow.stem(word) for word in text.split()])

# Apply transformations
movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(get_cast)
movies["crew"] = movies["crew"].apply(get_director)
movies["overview"] = movies["overview"].apply(lambda x: x.split())

movies["cast"] = movies["cast"].apply(remove_space)
movies["genres"] = movies["genres"].apply(remove_space)
movies["crew"] = movies["crew"].apply(remove_space)
movies["keywords"] = movies["keywords"].apply(remove_space)

movies["tags"] = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]
newdf = movies[["movie_id", "title", "tags"]].copy()

newdf["tags"] = newdf["tags"].apply(lambda x: " ".join(x))
newdf["tags"] = newdf["tags"].apply(lambda x: x.lower())
newdf["tags"] = newdf["tags"].apply(stemmer)

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(newdf["tags"]).toarray()

# Similarity Matrix
similarity = cosine_similarity(vectors)

import requests

# To get the posters of the movies based on the movie_id
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

# Recommendation function
def recommend_chat(movie):
    idx = newdf[newdf["title"] == movie].index[0]
    similarity_scores = list(enumerate(similarity[idx]))
    top_matches = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]

    titles = [newdf.iloc[i[0]].title for i in top_matches]
    # posters = [fetch_poster(newdf.iloc[i[0]].movie_id) for i in top_matches]

    return titles

import pickle
pickle.dump(newdf.to_dict(),open('movie_list_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))
