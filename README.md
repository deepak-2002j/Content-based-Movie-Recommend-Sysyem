# Movie Recommender System

A content-based movie recommendation system built with Python and Streamlit that suggests similar movies based on genres, keywords, cast, crew, and plot overview.

## Features

- **Content-Based Filtering**: Recommends movies based on similarity in genres, keywords, cast, crew, and plot
- **Interactive Web Interface**: Built with Streamlit for easy movie selection
- **Movie Posters**: Fetches and displays movie posters from TMDB API
- **Text Processing**: Uses stemming and TF-IDF vectorization for improved accuracy

## Dataset

Uses TMDB 5000 Movie Dataset:
- `tmdb_5000_movies.csv` - Movie details (genres, keywords, overview)
- `tmdb_5000_credits.csv` - Cast and crew information

## Installation

```bash
pip install pandas numpy scikit-learn nltk streamlit requests
```

## Usage

1. **Data Preprocessing** (run once):
   ```bash
   python preprocessing.py
   ```

2. **Run Streamlit App**:
   ```bash
   streamlit run app.py
   ```

3. Select a movie from the dropdown and click "Show Recommendation" to get 5 similar movies.

## How It Works

1. **Feature Extraction**: Combines genres, keywords, cast, crew, and overview into tags
2. **Text Processing**: Applies stemming and removes stop words
3. **Vectorization**: Converts text to numerical vectors using CountVectorizer
4. **Similarity Calculation**: Uses cosine similarity to find similar movies
5. **Recommendation**: Returns top 5 most similar movies with posters

## Files

- `preprocessing.py` - Data preprocessing and model training
- `app.py` - Streamlit web application
- `movie_list_dict.pkl` - Processed movie data
- `similarity.pkl` - Precomputed similarity matrix

## API Key

Requires TMDB API key for fetching movie posters. Replace the API key in the `fetch_poster()` function with your own key from [TMDB](https://www.themoviedb.org/settings/api).
