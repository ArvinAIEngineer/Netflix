# streamlit_app_unified.py - Complete Netflix Recommender in One File
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

# --- Configuration ---
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
NO_POSTER_IMAGE = "https://via.placeholder.com/200x300?text=No+Poster"
NO_BACKDROP_IMAGE = "https://via.placeholder.com/600x338?text=No+Backdrop"
ARTIFACTS_DIR = "artifacts"
DATA_PATH = os.path.join(ARTIFACTS_DIR, "netflix_recommender_data.pkl")

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Netflix Content Recommender",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Netflix-style CSS ---
st.markdown("""
<style>
.stApp { background-color: #141414; }
.block-container { max-width: 1400px; padding-top: 2rem; }
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stImage"] {
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.6);
}
.similarity-badge {
    background: linear-gradient(135deg, #E50914, #B20710);
    color: white;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 600;
    display: inline-block;
    margin: 0.5rem 0;
}
.genre-tag {
    background-color: #2F2F2F;
    color: #B3B3B3;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 0.9em;
    display: inline-block;
    margin: 2px;
}
.section-header {
    font-size: 1.6em;
    font-weight: 600;
    color: #FFFFFF;
    margin: 1rem 0;
    border-bottom: 3px solid #E50914;
}
</style>
""", unsafe_allow_html=True)

# --- Recommender Class ---
class Recommender:
    def __init__(self, data_path):
        self.df = pd.read_pickle(data_path)

        self.df['overview'] = self.df['overview'].fillna('')
        self.df['keywords'] = self.df['keywords'].apply(lambda x: x if isinstance(x, list) else [])
        self.df['genre_names'] = self.df['genre_names'].apply(lambda x: x if isinstance(x, list) else [])

        self.df['combined_features'] = self.df.apply(
            lambda r: ' '.join(r['genre_names']) + ' ' +
                      ' '.join(r['keywords']) + ' ' +
                      r['overview'],
            axis=1
        ).str.lower()

        self.vectorizer = TfidfVectorizer(stop_words="english", max_df=0.85, min_df=2)
        self.tfidf = self.vectorizer.fit_transform(self.df['combined_features'])

        self.id_to_index = pd.Series(self.df.index, index=self.df['id']).to_dict()

    def _format(self, row):
        return {
            "id": int(row['id']),
            "title": row.get('title') or row.get('name'),
            "media_type": row.get('media_type'),
            "poster_path": row.get('poster_path'),
            "overview": row.get('overview'),
            "release_date": row.get('release_date') or row.get('first_air_date'),
            "vote_average": row.get('vote_average'),
            "popularity": row.get('popularity'),
            "genre_names": row.get('genre_names'),
            "keywords": row.get('keywords')
        }

    def search_item(self, query):
        q = query.lower()
        results = self.df[
            self.df['title'].str.lower().str.contains(q, na=False) |
            self.df['name'].str.lower().str.contains(q, na=False)
        ].head(10)
        return [self._format(r) for _, r in results.iterrows()]

    def get_recommendations(self, item_id, top_n=10):
        idx = self.id_to_index.get(item_id)
        if idx is None:
            return None, []

        scores = linear_kernel(self.tfidf[idx:idx+1], self.tfidf).flatten()
        indices = scores.argsort()[-top_n-1:-1][::-1]

        source = self._format(self.df.iloc[idx])
        recs = []

        for i in indices:
            item = self._format(self.df.iloc[i])
            item["similarity_score"] = round(scores[i], 4)
            item["reasoning"] = ["Similar content based on genres, keywords, and overview"]
            recs.append(item)

        return source, recs

# --- Load Model ---
@st.cache_resource
def load_recommender():
    if not os.path.exists(DATA_PATH):
        st.error(f"Missing data file: {DATA_PATH}")
        st.stop()
    return Recommender(DATA_PATH)

recommender = load_recommender()

# --- UI ---
st.markdown("<h2 style='color:#E50914;text-align:center;'>🎬 Netflix Content Recommender</h2>", unsafe_allow_html=True)

query = st.text_input("Search", placeholder="Search for a movie or TV show...")

if query:
    results = recommender.search_item(query)

    if results:
        options = [
            f"{'🎬' if r['media_type']=='movie' else '📺'} {r['title']} ||| {r['id']}"
            for r in results
        ]
        choice = st.radio("Select a title", options)

        item_id = int(choice.split("|||")[1])
        source, recs = recommender.get_recommendations(item_id)

        st.markdown('<p class="section-header">Your Selection</p>', unsafe_allow_html=True)
        cols = st.columns([1,4])

        with cols[0]:
            poster = TMDB_IMAGE_BASE_URL + source['poster_path'] if source['poster_path'] else NO_POSTER_IMAGE
            st.image(poster, use_column_width=True)

        with cols[1]:
            st.markdown(f"### {source['title']}")
            st.markdown(source['overview'])

        st.markdown('<p class="section-header">Recommended for You</p>', unsafe_allow_html=True)

        grid = st.columns(5)
        for i, rec in enumerate(recs):
            with grid[i % 5]:
                poster = TMDB_IMAGE_BASE_URL + rec['poster_path'] if rec['poster_path'] else NO_POSTER_IMAGE
                st.image(poster, use_column_width=True)
                st.markdown(f"**{rec['title']}**")
                st.markdown(
                    f"<div class='similarity-badge'>{rec['similarity_score']*100:.0f}% Match</div>",
                    unsafe_allow_html=True
                )
    else:
        st.info("No results found.")
