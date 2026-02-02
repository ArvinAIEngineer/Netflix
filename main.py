# streamlit_app_unified.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

# ---------------- CONFIG ----------------
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
NO_POSTER_IMAGE = "https://via.placeholder.com/200x300?text=No+Poster"
ARTIFACTS_DIR = "artifacts"
DATA_PATH = os.path.join(ARTIFACTS_DIR, "netflix_recommender_data.pkl")

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="Netflix Content Recommender",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- GLOBAL CSS (ALL TEXT WHITE) ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #141414;
}

/* Remove Streamlit chrome */
#MainMenu, footer, header {
    visibility: hidden;
}

/* FORCE ALL TEXT TO WHITE */
html, body, p, span, div, label,
h1, h2, h3, h4, h5, h6,
li, ul, ol, small, strong, em {
    color: #FFFFFF !important;
}

/* Text input */
.stTextInput input {
    background-color: #1f1f1f !important;
    color: #FFFFFF !important;
    border: 2px solid #E50914;
}

/* Radio buttons */
div[data-testid="stRadio"] label {
    color: #FFFFFF !important;
    font-size: 1.05em;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label > div {
    color: #FFFFFF !important;
}

/* Section headers */
.section-header {
    font-size: 1.6em;
    font-weight: 600;
    margin: 1rem 0;
    border-bottom: 3px solid #E50914;
}

/* Cards */
.content-card {
    background: linear-gradient(to bottom, #1a1a1a, #0d0d0d);
    border-radius: 8px;
    padding: 1.5rem;
    border: 1px solid #333;
}

/* Images */
div[data-testid="stImage"] {
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.6);
}

/* Similarity badge */
.similarity-badge {
    background: linear-gradient(135deg, #E50914, #B20710);
    color: #FFFFFF !important;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 600;
    display: inline-block;
    margin: 0.5rem 0;
}

/* Genre tags */
.genre-tag {
    background-color: #2F2F2F;
    color: #FFFFFF !important;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 0.9em;
    margin: 2px;
    display: inline-block;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: #2F2F2F;
    color: #FFFFFF !important;
    font-weight: 500;
}

/* Info / warning boxes */
.stInfo, .stWarning {
    background-color: #2F2F2F !important;
    color: #FFFFFF !important;
    border-left: 4px solid #E50914;
}

</style>
""", unsafe_allow_html=True)

# ---------------- RECOMMENDER ----------------
class Recommender:
    def __init__(self, path):
        self.df = pd.read_pickle(path)

        self.df['overview'] = self.df['overview'].fillna('')
        self.df['keywords'] = self.df['keywords'].apply(lambda x: x if isinstance(x, list) else [])
        self.df['genre_names'] = self.df['genre_names'].apply(lambda x: x if isinstance(x, list) else [])

        self.df['combined'] = (
            self.df['overview'] +
            self.df['keywords'].apply(lambda x: " ".join(x)) +
            self.df['genre_names'].apply(lambda x: " ".join(x))
        ).str.lower()

        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
        self.matrix = self.vectorizer.fit_transform(self.df['combined'])

        self.id_to_index = pd.Series(self.df.index, index=self.df['id']).to_dict()

    def search(self, query):
        q = query.lower()
        results = self.df[
            self.df['title'].str.lower().str.contains(q, na=False) |
            self.df['name'].str.lower().str.contains(q, na=False)
        ].head(10)

        return results

    def recommend(self, item_id, top_n=10):
        idx = self.id_to_index.get(item_id)
        scores = linear_kernel(self.matrix[idx:idx+1], self.matrix).flatten()
        indices = scores.argsort()[-top_n-1:-1][::-1]

        source = self.df.iloc[idx]
        recs = self.df.iloc[indices].copy()
        recs['score'] = scores[indices]

        return source, recs

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(DATA_PATH):
        st.error("Data file missing")
        st.stop()
    return Recommender(DATA_PATH)

rec = load_model()

# ---------------- UI ----------------
st.markdown("<h2 style='color:#E50914;text-align:center;'>🎬 Netflix Content Recommender</h2>", unsafe_allow_html=True)

query = st.text_input("Search", placeholder="Search for a movie or TV show...")

if query:
    results = rec.search(query)

    if not results.empty:
        options = [
            f"{'🎬' if r.media_type=='movie' else '📺'} {r.title or r.name} ||| {r.id}"
            for _, r in results.iterrows()
        ]

        choice = st.radio("Select a title", options)
        item_id = int(choice.split("|||")[1])

        source, recs = rec.recommend(item_id)

        st.markdown('<p class="section-header">Your Selection</p>', unsafe_allow_html=True)

        cols = st.columns([1, 4])
        with cols[0]:
            poster = TMDB_IMAGE_BASE_URL + source.poster_path if source.poster_path else NO_POSTER_IMAGE
            st.image(poster, use_column_width=True)

        with cols[1]:
            st.markdown(f"### {source.title or source.name}")
            st.markdown(source.overview)

        st.markdown('<p class="section-header">Recommended For You</p>', unsafe_allow_html=True)

        grid = st.columns(5)
        for i, (_, r) in enumerate(recs.iterrows()):
            with grid[i % 5]:
                poster = TMDB_IMAGE_BASE_URL + r.poster_path if r.poster_path else NO_POSTER_IMAGE
                st.image(poster, use_column_width=True)
                st.markdown(f"**{r.title or r.name}**")
                st.markdown(
                    f"<div class='similarity-badge'>{int(r.score*100)}% Match</div>",
                    unsafe_allow_html=True
                )
    else:
        st.info("No results found")
