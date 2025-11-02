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

# Enhanced Netflix-style CSS
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #141414;
    }
    
    .main {
        background-color: #141414;
        padding-top: 1rem;
    }
    
    /* Remove default spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Search Box Styling */
    .stTextInput > div > div > input {
        background-color: #2F2F2F;
        color: #FFFFFF;
        border: 2px solid #404040;
        border-radius: 4px;
        padding: 12px 20px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #E50914;
        box-shadow: 0 0 10px rgba(229, 9, 20, 0.3);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8em;
        font-weight: 600;
        color: #FFFFFF;
        margin: 1rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #E50914;
    }
    
    /* Radio Button Styling */
    .stRadio > div {
        background-color: #2F2F2F;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #404040;
    }
    
    .stRadio > div > label {
        color: #E0E0E0 !important;
        font-size: 1.1em;
        padding: 0.5rem 0;
    }
    
    .stRadio > div > label:hover {
        background-color: #404040;
        border-radius: 4px;
        cursor: pointer;
    }
    
    /* Card Styling */
    .content-card {
        background: linear-gradient(to bottom, #1a1a1a, #0d0d0d);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.4);
    }
    
    /* Movie/Show Cards */
    div[data-testid="column"] {
        background-color: transparent;
    }
    
    div[data-testid="stImage"] {
        border-radius: 4px;
        overflow: hidden;
        transition: transform 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.6);
    }
    
    div[data-testid="stImage"]:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 16px rgba(229, 9, 20, 0.4);
    }
    
    /* Info Boxes */
    .stAlert {
        background-color: #2F2F2F;
        border-left: 4px solid #E50914;
        color: #E0E0E0;
        border-radius: 4px;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #2F2F2F;
        color: #FFFFFF;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #404040;
    }
    
    /* Text Colors */
    p, span, label {
        color: #E0E0E0;
    }
    
    h1, h2, h3, h4 {
        color: #FFFFFF;
    }
    
    /* Markdown overrides */
    .stMarkdown {
        color: #E0E0E0;
        margin-bottom: 0.5rem;
    }
    
    .stMarkdown p {
        margin-bottom: 0.5rem;
    }
    
    /* Reduce spacing in main container */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #E50914 !important;
    }
    
    /* Similarity Badge */
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
    
    /* Genre Tags */
    .genre-tag {
        background-color: #2F2F2F;
        color: #B3B3B3;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.9em;
        display: inline-block;
        margin: 2px;
    }
    
    /* Divider */
    hr {
        border: none;
        margin: 0;
        padding: 0;
        display: none;
    }
    
    /* Remove extra spacing from elements */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem;
    }
    
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    
    /* Warning/Info boxes */
    .stWarning, .stInfo {
        background-color: #2F2F2F;
        color: #E0E0E0;
        border-left: 4px solid #E50914;
    }
    
    /* Small text */
    small {
        color: #B3B3B3;
    }
</style>
""", unsafe_allow_html=True)

# --- Recommender Class (Embedded) ---
class Recommender:
    def __init__(self, data_path):
        """Initialize the recommender with data from pickle file"""
        try:
            self.df = pd.read_pickle(data_path)
        except Exception as e:
            st.error(f"Failed to load data from {data_path}: {e}")
            raise

        # Prepare features
        self.df['overview'] = self.df['overview'].fillna('')
        self.df['keywords'] = self.df['keywords'].apply(lambda x: x if isinstance(x, list) else [])
        self.df['genre_names'] = self.df['genre_names'].apply(lambda x: x if isinstance(x, list) else [])
        
        self.df['combined_features'] = self.df.apply(
            lambda row: ' '.join(row['genre_names']) + ' ' + ' '.join(row['keywords']) + ' ' + row['overview'],
            axis=1
        )
        self.df['combined_features'] = self.df['combined_features'].str.lower()
        
        # Initialize and fit TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2)
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['combined_features'])
        except Exception as e:
            st.error(f"Failed to fit TF-IDF vectorizer: {e}")
            raise
        
        # Create mappings
        self.id_to_index = pd.Series(self.df.index, index=self.df['id']).to_dict()
        self.title_to_id = pd.Series(self.df['id'].values, index=self.df['title'].str.lower()).to_dict()
        self.name_to_id = pd.Series(self.df['id'].values, index=self.df['name'].str.lower()).to_dict()

    def _prepare_item_for_output(self, item_row):
        """Convert DataFrame row to dictionary with proper types"""
        output = {
            "id": int(item_row['id']),
            "title": item_row.get('title') or item_row.get('name'),
            "media_type": item_row.get('media_type'),
            "poster_path": item_row.get('poster_path'),
            "backdrop_path": item_row.get('backdrop_path'),
            "overview": item_row.get('overview'),
            "release_date": item_row.get('release_date') or item_row.get('first_air_date'),
            "popularity": float(item_row['popularity']) if pd.notna(item_row.get('popularity')) else None,
            "vote_average": float(item_row['vote_average']) if pd.notna(item_row.get('vote_average')) else None,
            "genre_names": item_row.get('genre_names'),
            "keywords": item_row.get('keywords')
        }
        return output

    def search_item(self, query):
        """Search for items by title"""
        query_lower = query.lower()
        
        # Try exact match first
        exact_match = self.df[(self.df['title'].str.lower() == query_lower) | 
                              (self.df['name'].str.lower() == query_lower)]
        if not exact_match.empty:
            results = exact_match.head(5) 
        else:
            # Partial match
            results = self.df[(self.df['title'].str.lower().str.contains(query_lower, na=False)) | 
                              (self.df['name'].str.lower().str.contains(query_lower, na=False))].head(10)

        return [self._prepare_item_for_output(row) for _, row in results.iterrows()]

    def get_recommendations(self, item_id, top_n=10):
        """Generate content-based recommendations"""
        idx = self.id_to_index.get(item_id)
        if idx is None:
            return None, []

        source_vector = self.tfidf_matrix[idx:idx+1]
        if source_vector.nnz == 0:
            return None, []
            
        cosine_similarities = linear_kernel(source_vector, self.tfidf_matrix).flatten()
        related_indices = cosine_similarities.argsort()[-top_n*2-1:-1][::-1]

        source_item = self.df.iloc[idx]
        source_genres = source_item.get('genre_names', [])
        source_keywords = source_item.get('keywords', [])
        
        recommendations = []
        for i in related_indices:
            if i == idx:
                continue
            
            rec_item_row = self.df.iloc[i]
            rec_genres = rec_item_row.get('genre_names', [])
            rec_keywords = rec_item_row.get('keywords', [])

            shared_genres = list(set(source_genres) & set(rec_genres))
            shared_keywords = list(set(source_keywords) & set(rec_keywords))
            
            reasoning = []
            if shared_genres:
                reasoning.append(f"Shares genres: {', '.join(shared_genres)}")
            if shared_keywords:
                reasoning.append(f"Shares keywords: {', '.join(shared_keywords)}")
            if not reasoning:
                reasoning.append("Similar based on description.")

            output_rec_item = self._prepare_item_for_output(rec_item_row)
            output_rec_item['similarity_score'] = round(cosine_similarities[i], 4)
            output_rec_item['reasoning'] = reasoning
            
            recommendations.append(output_rec_item)

            if len(recommendations) >= top_n:
                break
        
        return self._prepare_item_for_output(source_item), recommendations

# --- Load Recommender (Cached) ---
@st.cache_resource
def load_recommender():
    """Load recommender model once and cache it"""
    if not os.path.exists(DATA_PATH):
        st.error(f"⚠️ Data file not found at `{DATA_PATH}`")
        st.info("Please ensure the artifacts folder contains `netflix_recommender_data.pkl`")
        st.stop()
    
    with st.spinner("🔄 Loading Netflix content database..."):
        try:
            recommender = Recommender(DATA_PATH)
            return recommender
        except Exception as e:
            st.error(f"❌ Failed to initialize recommender: {e}")
            st.stop()

# Initialize recommender
recommender = load_recommender()

# --- Main App Logic ---

# Welcome message at the top
st.markdown("""
<div class="content-card" style="text-align: center; padding: 2rem; margin-bottom: 1.5rem;">
    <h2 style="color: #E50914; margin-bottom: 1rem;">👋 Welcome to Netflix Recommender!</h2>
    <p style="font-size: 1.2em; color: #B3B3B3;">
        Start by searching for your favorite movie or TV show below.<br>
        We'll find similar content you'll love based on genres, themes, and more!
    </p>
</div>
""", unsafe_allow_html=True)

# Search Input
st.markdown("### 🔍 What are you in the mood for?")
search_query = st.text_input("Search for a movie or TV show", "", label_visibility="collapsed", 
                              placeholder="Search for a movie or TV show...")

if search_query:
    with st.spinner(f"🔎 Searching for '{search_query}'..."):
        search_results = recommender.search_item(search_query)

    if search_results:
        st.markdown(f'<p class="section-header">📺 Found {len(search_results)} Result(s)</p>', 
                    unsafe_allow_html=True)
        
        display_options = []
        for result in search_results:
            title = result.get('title') or result.get('name')
            media_type = result.get('media_type')
            release_year = (result.get('release_date') or result.get('first_air_date'))
            release_year = release_year[:4] if release_year else 'N/A'
            
            icon = "🎬" if media_type == "movie" else "📺"
            display_options.append(f"{icon} {title} ({media_type.upper()}) • {release_year} ||| {result['id']}")

        selected_option = st.radio("**Select a title to get personalized recommendations:**", 
                                   display_options, label_visibility="visible")

        if selected_option:
            selected_item_id = int(selected_option.split('|||')[1].strip())
            
            with st.spinner("✨ Generating personalized recommendations..."):
                source_item, recommendations = recommender.get_recommendations(selected_item_id, top_n=10)

            if source_item and recommendations:
                # --- Display Source Item Details ---
                st.markdown('<div class="content-card">', unsafe_allow_html=True)
                st.markdown('<p class="section-header">📌 Your Selection</p>', unsafe_allow_html=True)
                
                source_cols = st.columns([1, 4])
                
                with source_cols[0]:
                    poster_url = TMDB_IMAGE_BASE_URL + source_item['poster_path'] if source_item.get('poster_path') else NO_POSTER_IMAGE
                    st.image(poster_url, use_container_width=True)
                
                with source_cols[1]:
                    media_icon = "🎬" if source_item.get('media_type') == 'movie' else "📺"
                    st.markdown(f"## {media_icon} {source_item.get('title')}")
                    
                    info_cols = st.columns(3)
                    with info_cols[0]:
                        st.markdown(f"**Type:** {source_item.get('media_type', '').upper()}")
                        st.markdown(f"**Released:** {source_item.get('release_date', 'N/A')}")
                    with info_cols[1]:
                        st.markdown(f"**⭐ Rating:** {source_item.get('vote_average', 'N/A'):.1f}/10")
                        st.markdown(f"**🔥 Popularity:** {source_item.get('popularity', 'N/A'):.0f}")
                    with info_cols[2]:
                        if source_item.get('genre_names'):
                            st.markdown("**Genres:**")
                            genres_html = ' '.join([f'<span class="genre-tag">{g}</span>' 
                                                   for g in source_item['genre_names'][:4]])
                            st.markdown(genres_html, unsafe_allow_html=True)
                    
                    if source_item.get('overview'):
                        st.markdown(f"**Overview:** {source_item['overview']}")
                    
                    if source_item.get('keywords'):
                        keywords_html = ' '.join([f'<span class="genre-tag">#{k}</span>' 
                                                 for k in source_item['keywords'][:6]])
                        st.markdown(keywords_html, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

                # --- Recommendations Section ---
                st.markdown(f'<p class="section-header" style="margin-top: 1.5rem;">✨ Because You Selected "{source_item.get("title")}"</p>', 
                           unsafe_allow_html=True)
                st.markdown(f"<p style='color: #B3B3B3; margin-bottom: 1rem; margin-top: 0.5rem;'>Here are {len(recommendations)} personalized recommendations just for you</p>", 
                           unsafe_allow_html=True)

                # Display recommendations in a grid
                cols_per_row = 5
                for i in range(0, len(recommendations), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < len(recommendations):
                            rec = recommendations[i + j]
                            with cols[j]:
                                rec_poster_url = TMDB_IMAGE_BASE_URL + rec['poster_path'] if rec.get('poster_path') else NO_POSTER_IMAGE
                                st.image(rec_poster_url, use_container_width=True)
                                
                                media_icon = "🎬" if rec.get('media_type') == 'movie' else "📺"
                                st.markdown(f"**{media_icon} {rec.get('title')}**")
                                
                                similarity_pct = rec.get('similarity_score', 0) * 100
                                st.markdown(f'<div class="similarity-badge">{similarity_pct:.0f}% Match</div>', 
                                           unsafe_allow_html=True)
                                
                                if rec.get('reasoning'):
                                    reasoning_text = " • ".join(rec['reasoning'][:2])
                                    st.info(f"💡 {reasoning_text}", icon="✨")
                                
                                with st.expander("📋 Full Details"):
                                    st.markdown(f"**Overview:** {rec.get('overview', 'N/A')}")
                                    st.markdown(f"**Released:** {rec.get('release_date', 'N/A')}")
                                    st.markdown(f"**⭐ Rating:** {rec.get('vote_average', 0):.1f}/10")
                                    st.markdown(f"**🔥 Popularity:** {rec.get('popularity', 0):.0f}")
                                    
                                    if rec.get('reasoning'):
                                        st.markdown("**Why recommended:**")
                                        for reason in rec['reasoning']:
                                            st.markdown(f"• {reason}")
            else:
                st.warning("⚠️ No recommendations found for this title. It might be too obscure or not have enough metadata for similarity matching.")
    else:
        st.info(f"🔍 No results found for '{search_query}'. Try a different search term or check your spelling.")

# Footer
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0; margin-top: 3rem;">
    <p>Project by Dr Arvin Subramanian</p>
""", unsafe_allow_html=True)
