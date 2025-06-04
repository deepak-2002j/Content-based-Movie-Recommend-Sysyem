🎬 Content-Based Movie Recommender System
A content-based movie recommender system built using NLP and cosine similarity that suggests movies similar to the user's input based on metadata like genres, cast, crew, keywords, and overview.

🚀 Features
Extracts features like genres, cast, crew, keywords, and overview from TMDB dataset

Processes text with stemming and vectorization (CountVectorizer)

Computes movie similarity using cosine similarity

Fetches movie posters using the TMDB API

Recommends top 10 similar movies based on user input

🧠 Tech Stack
Python, Pandas, NumPy

scikit-learn for vectorization and similarity calculation

NLTK for stemming

TMDB API for dynamic poster fetching

📁 Dataset
tmdb_5000_movies.csv

tmdb_5000_credits.csv

📌 How It Works
Extract and clean features like genres, keywords, cast, crew, and overview

Combine them into a single tags field

Apply stemming and vectorization

Compute cosine similarity between movies

Return top 10 similar movies with posters using the TMDB API

📦 Output
Recommended movie titles

Movie posters for better UI integration

🧪 Sample Function Call
python
Copy
Edit
titles, posters = recommend_chat("Avatar")
📌 Note
Requires a TMDB API key for poster fetching. Replace the key in fetch_poster() if needed.

