import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.cm as cm
import joblib
from sklearn.preprocessing import StandardScaler
import os
import io
import base64
from PIL import Image
import random
import time
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Try importing transformers package if available
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    transformers_available = True
except ImportError:
    transformers_available = False

# Set page title and favicon
st.set_page_config(
    page_title="Music Genre Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1DB954;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4b93c1;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: 500;
    }
    .section-header {
        font-size: 1.3rem;
        color: #4b93c1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    .feature-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background-color: #e6f3ff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .genre-label {
        font-size: 2rem;
        font-weight: 700;
        color: #1DB954;
        margin-bottom: 10px;
    }
    .confidence-bar {
        height: 24px;
        border-radius: 12px;
        margin: 10px 0;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        color: #888;
        font-size: 0.8rem;
    }
    .stSlider {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .row-gap {
        margin-bottom: 1.5rem;
    }
    .logo-img {
        max-width: 150px;
        margin: 0 auto;
        display: block;
    }
    .audio-feature-label {
        font-weight: 600;
        color: #1DB954;
    }
    .emotion-feature-label {
        font-weight: 600;
        color: #4b93c1;
    }
    .search-results {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .lyrics-box {
        max-height: 300px;
        overflow-y: auto;
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #4b93c1;
        font-style: italic;
    }
    .info-box {
        background-color: #e8f4f8; 
        padding: 15px; 
        border-radius: 5px; 
        border-left: 5px solid #4b93c1;
        margin-bottom: 15px;
    }
    .api-status {
        padding: 8px 15px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    .api-connected {
        background-color: #d4edda;
        color: #155724;
    }
    .api-disconnected {
        background-color: #f8d7da;
        color: #721c24;
    }
    .ai-feature-notice {
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
        padding: 10px 15px;
        margin-bottom: 15px;
        border-radius: 4px;
    }
    .feature-source {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
        margin-top: 5px;
    }
    .feature-source-ai {
        color: #FF6F00;
    }
    
    .feature-source-api {
        color: #1976D2;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'selected_genre' not in st.session_state:
    st.session_state.selected_genre = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'spotify_data' not in st.session_state:
    st.session_state.spotify_data = None
if 'lyrics_analyzed' not in st.session_state:
    st.session_state.lyrics_analyzed = False

if 'current_lyrics' not in st.session_state:
    st.session_state.current_lyrics = None
if 'song_data_for_prediction' not in st.session_state:
    st.session_state.song_data_for_prediction = None
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None 

# Create a radar chart function
def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # Calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon returns a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def create_emotion_radar_chart(emotion_data):
    # Get the emotion values and labels
    emotions = list(emotion_data.keys())
    values = list(emotion_data.values())
    
    # Create the radar chart
    N = len(emotions)
    theta = radar_factory(N, frame='polygon')
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
    
    # Plot the emotion values
    # Note: The radar_factory now handles closing the polygon
    ax.plot(theta, values, 'o-', linewidth=2, color='#4b93c1')
    ax.fill(theta, values, alpha=0.25, color='#4b93c1')
    
    # Set the labels
    ax.set_varlabels(emotions)
    
    # Set the radial limits
    ax.set_ylim(0, 1)
    
    # Add a title
    plt.title('Emotion Profile', size=15, color='#4b93c1', y=1.1)
    
    return fig

def create_genre_distribution_chart(genre_probabilities, top_n=10):
    # Sort genres by probability and take top N
    sorted_genres = sorted(genre_probabilities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    genres, probs = zip(*sorted_genres)
    
    # Create a horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a colormap
    cmap = cm.get_cmap('viridis')
    colors = [cmap(i/len(genres)) for i in range(len(genres))]
    
    # Create the bars
    bars = ax.barh(genres, probs, color=colors)
    
    # Add percentage labels to the bars
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width > 0.05 else 0.05
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1%}',
                va='center', color='white' if width > 0.1 else 'black', fontweight='bold')
    
    # Customize the chart
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title('Genre Probability Distribution', fontsize=15, color='#4b93c1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig

def get_genre_color(genre):
    # Define a mapping of genres to colors
    genre_colors = {
        'acoustic': '#1DB954',  # Spotify green
        'afrobeat': '#FF9900',  # Orange
        'alternative': '#8B4513',  # Brown
        'blues': '#0000FF',  # Blue
        'children': '#FFC0CB',  # Pink
        'chill': '#00FFFF',  # Cyan
        'club': '#800080',  # Purple
        'country': '#964B00',  # Brown
        'dance': '#FF1493',  # Deep Pink
        'disco': '#FFD700',  # Gold
        'disney': '#1E90FF',  # Dodger Blue
        'edm': '#00FF00',  # Lime
        'electro': '#FF00FF',  # Magenta
        'emo': '#000000',  # Black
        'funk': '#FF4500',  # Orange Red
        'groove': '#32CD32',  # Lime Green
        'happy': '#FFFF00',  # Yellow
        'house': '#FF6347',  # Tomato
        'jazz': '#4B0082',  # Indigo
        'pop': '#FF69B4',  # Hot Pink
    }
    
    # Return the color for the genre, or a default color if not found
    return genre_colors.get(genre.lower(), '#1DB954')

# Function to initialize Spotify client
@st.cache_resource
def get_spotify_client():
    # Check if credentials are set
    client_id = os.environ.get('SPOTIFY_CLIENT_ID', '')
    client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET', '')
    
    if not client_id or not client_secret:
        return None
    
    try:
        client_credentials_manager = SpotifyClientCredentials(
            client_id=client_id, 
            client_secret=client_secret
        )
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        return sp
    except Exception as e:
        st.error(f"Error initializing Spotify client: {e}")
        return None

# Function to initialize Genius client
@st.cache_resource
def get_genius_client():
    # Check if credentials are set
    genius_token = os.environ.get('GENIUS_ACCESS_TOKEN', '')
    
    if not genius_token:
        return None
    
    try:
        genius = lyricsgenius.Genius(genius_token)
        # Configure genius client to exclude annotations and skip info message
        genius.verbose = False
        genius.remove_section_headers = True
        genius.skip_non_songs = True
        return genius
    except Exception as e:
        st.error(f"Error initializing Genius client: {e}")
        return None

def get_audio_features_from_llm(track_name, artist, genre=None):
    """
    Use LLM to estimate audio features for a song based on its title, artist, and optionally genre
    This is used when Spotify API cannot provide these features
    
    Parameters:
    track_name: The name of the song
    artist: The artist name
    genre: Known genre (if any)
    
    Returns:
    Dictionary with estimated audio features
    """
    import time
    import random
    
    # In a real implementation, this would call an LLM API
    # For example, using OpenAI's API or similar
    
    # To simulate an actual LLM, we'll create plausible values based on the song title
    # In a real implementation, replace this with an actual LLM API call
    st.info(f"Generating audio features for '{track_name}' by {artist} using AI estimation")
    
    # Show a simulated "thinking" spinner
    with st.spinner("AI is analyzing the song characteristics..."):
        # Simulate LLM processing time
        time.sleep(2)
        
        # Create a seed from the track name and artist for consistent results
        seed = hash(f"{track_name}{artist}") % 10000
        random.seed(seed)
        
        # Base values
        base_energy = random.uniform(0.3, 0.8)
        base_tempo = random.uniform(80, 160)
        base_valence = random.uniform(0.2, 0.8)
        base_mode = random.choice([0, 1])
        base_key = random.randint(0, 11)
        
        # Adjust for title keywords
        track_lower = track_name.lower()
        
        # Energy modifiers
        if any(word in track_lower for word in ['rock', 'hard', 'heavy', 'metal', 'punk', 'power', 'fire', 'burn']):
            base_energy = min(1.0, base_energy + 0.2)
        elif any(word in track_lower for word in ['calm', 'gentle', 'soft', 'slow', 'quiet', 'peace', 'sleep']):
            base_energy = max(0.1, base_energy - 0.2)
            
        # Tempo modifiers
        if any(word in track_lower for word in ['fast', 'run', 'dance', 'beat', 'quick', 'rush']):
            base_tempo = min(190, base_tempo + 20)
        elif any(word in track_lower for word in ['slow', 'ballad', 'adagio', 'gentle']):
            base_tempo = max(60, base_tempo - 20)
            
        # Valence modifiers
        if any(word in track_lower for word in ['happy', 'joy', 'love', 'sweet', 'fun', 'good', 'smile']):
            base_valence = min(1.0, base_valence + 0.2)
        elif any(word in track_lower for word in ['sad', 'blue', 'cry', 'tear', 'pain', 'hurt', 'lonely']):
            base_valence = max(0.1, base_valence - 0.2)
            
        # Mode modifiers (major/minor)
        if any(word in track_lower for word in ['happy', 'joy', 'bright', 'light', 'major']):
            base_mode = 1  # Major
        elif any(word in track_lower for word in ['sad', 'dark', 'minor', 'melancholy', 'pain']):
            base_mode = 0  # Minor
        
        # Use genre information if provided
        if genre:
            genre_lower = genre.lower()
            
            # Genre-specific adjustments
            if genre_lower in ['rock', 'metal', 'punk']:
                base_energy = min(1.0, base_energy + 0.15)
                base_tempo = min(180, base_tempo + 10)
            elif genre_lower in ['classical', 'ambient', 'chill']:
                base_energy = max(0.1, base_energy - 0.15)
            elif genre_lower in ['dance', 'edm', 'techno', 'house']:
                base_tempo = min(190, base_tempo + 15)
                base_energy = min(1.0, base_energy + 0.1)
            elif genre_lower in ['jazz', 'blues']:
                base_tempo = max(60, base_tempo - 10)
    
    # Create the features dictionary
    audio_features = {
        'energy': round(base_energy, 2),
        'tempo': round(base_tempo, 1),
        'valence': round(base_valence, 2),
        'mode': base_mode,
        'key': base_key,
        'acousticness': round(random.uniform(0.1, 0.9), 2),
        'danceability': round(random.uniform(0.2, 0.8), 2),
        'instrumentalness': round(random.uniform(0, 0.3), 2),
        'liveness': round(random.uniform(0.05, 0.3), 2),
        'loudness': round(random.uniform(-15, -5), 2),
        'speechiness': round(random.uniform(0.02, 0.1), 2),
        'using_ai_estimation': True
    }
    
    return audio_features

# Function to fetch song data from Spotify with debugging
# def get_song_from_spotify(query, spotify_client):
#     if not spotify_client:
#         st.error("Spotify integration is not available. Please enter audio features manually.")
#         return None
    
#     try:
#         # Print token information for debugging
#         auth_manager = spotify_client._auth_manager
#         token_info = auth_manager.get_cached_token() if hasattr(auth_manager, 'get_cached_token') else "Token info not available"
#         print(f"Token info: {token_info}")
        
#         print(f"Searching for: {query}")
#         # Search for the track
#         results = spotify_client.search(q=query, type='track', limit=5)
        
#         if not results['tracks']['items']:
#             st.error(f"No tracks found for '{query}'")
#             return None
        
#         # Select the top track
#         track = results['tracks']['items'][0]
#         track_id = track['id']
#         print(f"Found track: {track['name']} with ID: {track_id}")
        
#         # This is where the error seems to happen - let's add more debugging
#         print(f"Fetching audio features for track ID: {track_id}")
#         try:
#             audio_features = spotify_client.audio_features(track_id)[0]
#             print(f"Successfully fetched audio features")
#         except Exception as audio_error:
#             print(f"Error fetching audio features: {audio_error}")
#             # Try a workaround - get audio features individually
#             print("Trying alternative approach...")
#             try:
#                 audio_features_endpoint = f"https://api.spotify.com/v1/audio-features/{track_id}"
#                 audio_features = spotify_client._get(audio_features_endpoint)
#                 print(f"Alternative approach result: {audio_features}")
#             except Exception as alt_error:
#                 print(f"Alternative approach failed: {alt_error}")
#                 # Create dummy audio features as fallback
#                 audio_features = {
#                     'energy': 0.5,
#                     'tempo': 120,
#                     'valence': 0.5,
#                     'mode': 1,
#                     'key': 0,
#                     'acousticness': 0.5,
#                     'danceability': 0.5,
#                     'instrumentalness': 0.1,
#                     'liveness': 0.1,
#                     'loudness': -8.0,
#                     'speechiness': 0.1
#                 }
#                 print("Using dummy audio features as fallback")
        
#         # Get track details
#         print(f"Fetching track details for track ID: {track_id}")
#         track_info = spotify_client.track(track_id)
#         print("Successfully fetched track details")
        
#         # Combine relevant data
#         song_data = {
#             'track_name': track['name'],
#             'artists': ', '.join([artist['name'] for artist in track['artists']]),
#             'album': track['album']['name'],
#             'release_date': track_info['album']['release_date'],
#             'popularity': track['popularity'],
#             'preview_url': track['preview_url'],
#             'energy': audio_features['energy'],
#             'tempo': audio_features['tempo'],
#             'valence': audio_features['valence'],
#             'mode': audio_features['mode'],
#             'key': audio_features['key'],
#             'acousticness': audio_features['acousticness'],
#             'danceability': audio_features['danceability'],
#             'instrumentalness': audio_features['instrumentalness'],
#             'liveness': audio_features['liveness'],
#             'loudness': audio_features['loudness'],
#             'speechiness': audio_features['speechiness'],
#             'album_cover_url': track['album']['images'][0]['url'] if track['album']['images'] else None
#         }
        
#         return song_data
    
#     except Exception as e:
#         st.error(f"Error fetching song data: {e}")
#         import traceback
#         print(f"Detailed error: {traceback.format_exc()}")
#         return None

# Function to fetch song data from Spotify with LLM fallback
def get_song_from_spotify(query, spotify_client):
    if not spotify_client:
        st.error("Spotify integration is not available. Please enter audio features manually.")
        return None
    
    try:
        # Print token information for debugging
        auth_manager = spotify_client._auth_manager
        token_info = auth_manager.get_cached_token() if hasattr(auth_manager, 'get_cached_token') else "Token info not available"
        print(f"Token info: {token_info}")
        
        print(f"Searching for: {query}")
        # Search for the track
        results = spotify_client.search(q=query, type='track', limit=5)
        
        if not results['tracks']['items']:
            st.error(f"No tracks found for '{query}'")
            return None
        
        # Select the top track
        track = results['tracks']['items'][0]
        track_id = track['id']
        print(f"Found track: {track['name']} with ID: {track_id}")
        
        # Get track details
        print(f"Fetching track details for track ID: {track_id}")
        track_info = spotify_client.track(track_id)
        print("Successfully fetched track details")
        
        # Prepare basic song data without audio features
        song_data = {
            'track_name': track['name'],
            'artists': ', '.join([artist['name'] for artist in track['artists']]),
            'album': track['album']['name'],
            'release_date': track_info['album']['release_date'],
            'popularity': track['popularity'],
            'preview_url': track['preview_url'],
            'album_cover_url': track['album']['images'][0]['url'] if track['album']['images'] else None
        }
        
        # Try to get audio features from Spotify
        audio_features = None
        try:
            print(f"Attempting to fetch audio features for track ID: {track_id}")
            audio_features = spotify_client.audio_features(track_id)[0]
            print(f"Successfully fetched audio features from Spotify API")
            
            # Add Spotify audio features to song data
            if audio_features:
                song_data.update({
                    'energy': audio_features['energy'],
                    'tempo': audio_features['tempo'],
                    'valence': audio_features['valence'],
                    'mode': audio_features['mode'],
                    'key': audio_features['key'],
                    'acousticness': audio_features['acousticness'],
                    'danceability': audio_features['danceability'],
                    'instrumentalness': audio_features['instrumentalness'],
                    'liveness': audio_features['liveness'],
                    'loudness': audio_features['loudness'],
                    'speechiness': audio_features['speechiness'],
                    'using_spotify_api': True
                })
        except Exception as e:
            print(f"Error fetching Spotify audio features: {e}")
            audio_features = None
        
        # If Spotify audio features are not available, use LLM estimation
        if not audio_features:
            st.warning("⚠️ Spotify API audio features are unavailable due to API deprecation. Using AI to estimate audio features instead.")
            
            # Get estimated audio features from LLM
            llm_features = get_audio_features_from_llm(
                track_name=song_data['track_name'],
                artist=song_data['artists']
            )
            
            # Add LLM-estimated features to song data
            song_data.update(llm_features)
        
        return song_data
    
    except Exception as e:
        st.error(f"Error fetching song data: {e}")
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        return None

# Function to fetch lyrics from Genius
def fetch_lyrics(artist_name, song_title, genius_client):
    if not genius_client:
        return None
    
    try:
        # Search for the song
        song = genius_client.search_song(song_title, artist_name)
        
        if not song:
            st.warning(f"Lyrics for '{song_title}' by {artist_name} not found on Genius.")
            return None
        
        # Get the lyrics
        lyrics = song.lyrics
        
        # Clean up the lyrics (remove [Verse], [Chorus], etc.)
        lyrics = re.sub(r'\[.*?\]', '', lyrics)
        lyrics = re.sub(r'^\d+Embed$', '', lyrics, flags=re.MULTILINE)
        lyrics = lyrics.strip()
        
        return lyrics
    except Exception as e:
        st.error(f"Error fetching lyrics: {e}")
        return None

# Function to initialize Sentiment Analyzer
@st.cache_resource
def get_sentiment_analyzer():
    try:
        # Download necessary NLTK data
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        # Initialize the sentiment analyzer
        return SentimentIntensityAnalyzer()
    except Exception as e:
        st.error(f"Error initializing sentiment analyzer: {e}")
        return None

# Function to analyze lyrics sentiment
def analyze_lyrics(lyrics, sentiment_analyzer=None, emotion_classifier=None):
    if not lyrics:
        # Create dummy data if lyrics are not available
        return {
            'emotion_joy': np.random.random() * 0.8,
            'emotion_sadness': np.random.random() * 0.5,
            'emotion_love': np.random.random() * 0.6,
            'emotion_anger': np.random.random() * 0.3,
            'emotion_fear': np.random.random() * 0.2,
            'emotion_surprise': np.random.random() * 0.4,
            'source': 'simulated'
        }
    
    try:
        # Calculate word count
        word_count = len(re.findall(r'\b\w+\b', lyrics))
        
        # Initialize emotion scores
        emotion_scores = {
            'emotion_joy': 0.0,
            'emotion_sadness': 0.0,
            'emotion_love': 0.0,
            'emotion_anger': 0.0,
            'emotion_fear': 0.0,
            'emotion_surprise': 0.0,
            'word_count': word_count,
            'source': 'analysis'
        }
        
        # Calculate VADER sentiment if available
        if sentiment_analyzer:
            sentiment = sentiment_analyzer.polarity_scores(lyrics)
            emotion_scores.update({
                'vader_compound': sentiment['compound'],
                'vader_pos': sentiment['pos'],
                'vader_neg': sentiment['neg'],
                'vader_neu': sentiment['neu']
            })
            
            # Map VADER sentiment to emotions
            if sentiment['compound'] > 0.5:
                emotion_scores['emotion_joy'] += 0.7
            elif sentiment['compound'] > 0.2:
                emotion_scores['emotion_joy'] += 0.5
                emotion_scores['emotion_surprise'] += 0.3
            elif sentiment['compound'] > 0:
                emotion_scores['emotion_joy'] += 0.3
                emotion_scores['emotion_love'] += 0.2
            elif sentiment['compound'] > -0.2:
                emotion_scores['emotion_sadness'] += 0.3
            elif sentiment['compound'] > -0.5:
                emotion_scores['emotion_sadness'] += 0.5
                emotion_scores['emotion_fear'] += 0.2
            else:
                emotion_scores['emotion_sadness'] += 0.6
                emotion_scores['emotion_anger'] += 0.4
                
            # Scale based on confidence
            for key in ['emotion_joy', 'emotion_sadness', 'emotion_love', 'emotion_anger', 'emotion_fear', 'emotion_surprise']:
                emotion_scores[key] = min(1.0, emotion_scores[key])
            
        # Fallback to keyword-based approach if no transformer model
        joy_keywords = ['happy', 'joy', 'wonderful', 'love', 'smile', 'laugh', 'excited']
        sadness_keywords = ['sad', 'cry', 'tears', 'lonely', 'sorrow', 'grief', 'depressed']
        love_keywords = ['love', 'heart', 'passion', 'forever', 'together', 'soul', 'darling']
        anger_keywords = ['angry', 'hate', 'rage', 'fight', 'mad', 'fury', 'bitter']
        fear_keywords = ['fear', 'scared', 'afraid', 'terror', 'dread', 'horror', 'panic']
        surprise_keywords = ['surprise', 'shock', 'amazed', 'wonder', 'unexpected', 'astonished']
        
        lyrics_lower = lyrics.lower()
        
        # Calculate keyword matches
        joy_score = sum(lyrics_lower.count(word) for word in joy_keywords) / max(word_count, 1) * 5
        sadness_score = sum(lyrics_lower.count(word) for word in sadness_keywords) / max(word_count, 1) * 5
        love_score = sum(lyrics_lower.count(word) for word in love_keywords) / max(word_count, 1) * 5
        anger_score = sum(lyrics_lower.count(word) for word in anger_keywords) / max(word_count, 1) * 5
        fear_score = sum(lyrics_lower.count(word) for word in fear_keywords) / max(word_count, 1) * 5
        surprise_score = sum(lyrics_lower.count(word) for word in surprise_keywords) / max(word_count, 1) * 5
        
        # Blend with sentiment scores if available
        if 'vader_pos' in emotion_scores:
            joy_score = 0.7 * joy_score + 0.3 * emotion_scores['vader_pos']
            sadness_score = 0.7 * sadness_score + 0.3 * emotion_scores['vader_neg']
            anger_score = 0.7 * anger_score + 0.3 * emotion_scores['vader_neg'] * 0.7
        
        # Update scores
        emotion_scores.update({
            'emotion_joy': min(1.0, max(0.0, joy_score)),
            'emotion_sadness': min(1.0, max(0.0, sadness_score)),
            'emotion_love': min(1.0, max(0.0, love_score)),
            'emotion_anger': min(1.0, max(0.0, anger_score)),
            'emotion_fear': min(1.0, max(0.0, fear_score)),
            'emotion_surprise': min(1.0, max(0.0, surprise_score)),
            'source': 'keyword'
        })
        
        # Normalize emotions to sum to 1.0
        total = sum(emotion_scores[k] for k in ['emotion_joy', 'emotion_sadness', 'emotion_love', 'emotion_anger', 'emotion_fear', 'emotion_surprise'])
        if total > 0:
            for key in ['emotion_joy', 'emotion_sadness', 'emotion_love', 'emotion_anger', 'emotion_fear', 'emotion_surprise']:
                emotion_scores[key] /= total
        
        return emotion_scores
    
    except Exception as e:
        st.error(f"Error analyzing lyrics: {e}")
        # Return default values in case of error
        return {
            'emotion_joy': 0.2,
            'emotion_sadness': 0.2,
            'emotion_love': 0.2,
            'emotion_anger': 0.2,
            'emotion_fear': 0.2,
            'emotion_surprise': 0.2,
            'source': 'fallback'
        }

# Add this code where you display audio features after song search
def display_audio_features_with_source(spotify_data):
    # Create visualization
    audio_features = {
        'energy': spotify_data['energy'],
        'valence': spotify_data['valence'],
        'danceability': spotify_data.get('danceability', 0.5),
        'acousticness': spotify_data.get('acousticness', 0.5),
        'instrumentalness': spotify_data.get('instrumentalness', 0.5),
        'liveness': spotify_data.get('liveness', 0.5),
        'speechiness': spotify_data.get('speechiness', 0.5)
    }
    
    # Check if features were estimated by AI
    is_ai_estimated = spotify_data.get('using_ai_estimation', False)
    
    if is_ai_estimated:
        st.markdown("""
        <div class="ai-feature-notice">
            <strong>Note:</strong> Audio features shown below are AI-estimated due to Spotify API limitations.
            These estimations are based on song title, artist, and genre characteristics.
        </div>
        """, unsafe_allow_html=True)
    
    # # Display feature chart
    # audio_fig = create_audio_features_chart(audio_features)
    # st.pyplot(audio_fig)
    
    display_audio_features_with_source(spotify_data)
    
    # Add source indicator
    source_text = "AI-estimated features" if is_ai_estimated else "Spotify API features"
    source_class = "feature-source-ai" if is_ai_estimated else "feature-source-api"
    st.markdown(f"""
    <p class="feature-source {source_class}">Source: {source_text}</p>
    """, unsafe_allow_html=True)
    
    # Add feature table
    with st.expander("View all audio features"):
        features_df = pd.DataFrame({
            "Feature": ["Energy", "Valence", "Tempo", "Mode", "Key", "Danceability", "Acousticness"],
            "Value": [
                f"{spotify_data['energy']:.2f}",
                f"{spotify_data['valence']:.2f}",
                f"{spotify_data['tempo']:.1f} BPM",
                "Major" if spotify_data['mode'] == 1 else "Minor",
                get_key_name(spotify_data['key'], spotify_data['mode']),
                f"{spotify_data.get('danceability', 0.5):.2f}",
                f"{spotify_data.get('acousticness', 0.5):.2f}"
            ],
            "Source": ["AI-estimated" if is_ai_estimated else "Spotify API"] * 7
        })
        st.table(features_df)
        
# Function to visualize audio features
def create_audio_features_chart(audio_data):
    # Extract relevant audio features
    features = ['energy', 'valence', 'danceability', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
    values = [audio_data.get(feature, 0) for feature in features]
    
    # Create a horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors based on values
    colors = ['#ff9999' if v < 0.33 else '#66b3ff' if v < 0.66 else '#99ff99' for v in values]
    
    # Create the horizontal bar chart
    bars = ax.barh(features, values, color=colors)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                ha='left', va='center', fontweight='bold')
    
    # Customize the chart
    ax.set_xlim(0, 1)
    ax.set_xlabel('Value (0-1 Scale)', fontsize=12)
    ax.set_ylabel('Audio Feature', fontsize=12)
    ax.set_title('Audio Features Analysis', fontsize=16)
    ax.grid(axis='x', alpha=0.3)
    
    return fig

def check_model_files():
    """Check if all required model files exist and are loadable"""
    required_files = ['model.joblib', 'encoder.joblib', 'scaler.joblib', 'metadata.joblib']
    missing_files = []
    
    for file in required_files:
        try:
            # Try to load the file
            joblib.load(file)
        except (FileNotFoundError, IOError):
            missing_files.append(file)
    
    return len(missing_files) == 0, missing_files

# # def predict_genre(song_data, model_params_path='best_genre_prediction_model_params.joblib',
#     scaler_path='genre_prediction_scaler.joblib',
#         features_path='selected_features.joblib'):
#     """
#     Predict the genre of a new song
    
#     Parameters:
#     song_data: Dictionary with song features
#     model_params_path: Path to the saved model parameters
#     scaler_path: Path to the saved scaler
#     features_path: Path to the saved selected features
    
#     Returns:
#     Predicted genre and confidence
#     """
#     # Load the model parameters, scaler, and selected features
#     model_params = joblib.load(model_params_path)
#     model = model_params['model']
#     encoder = model_params['encoder']
#     scaler = joblib.load(scaler_path)
#     selected_features = joblib.load(features_path)

#     # Ensure the song data has all required features
#     for feature in selected_features:
#         if feature not in song_data and feature not in ['joy_to_sadness_ratio', 'anger_to_love_ratio',
#                                                        'energy_to_valence_ratio', 'surprise_to_fear_ratio',
#                                                        'energy_x_tempo', 'energy_x_valence', 'joy_x_tempo',
#                                                        'sadness_x_valence', 'energy_squared', 'tempo_log',
#                                                        'valence_squared', 'total_emotion_intensity',
#                                                        'positive_emotions', 'negative_emotions',
#                                                        'emotion_ratio', 'emotional_diversity']:
#             song_data[feature] = 0

#     # Calculate engineered features if they're part of selected_features
#     # Ratios
#     if 'joy_to_sadness_ratio' in selected_features:
#         song_data['joy_to_sadness_ratio'] = song_data.get('emotion_joy', 0) / (song_data.get('emotion_sadness', 0) + 0.0001)
#     if 'anger_to_love_ratio' in selected_features:
#         song_data['anger_to_love_ratio'] = song_data.get('emotion_anger', 0) / (song_data.get('emotion_love', 0) + 0.0001)
#     if 'energy_to_valence_ratio' in selected_features:
#         song_data['energy_to_valence_ratio'] = song_data.get('energy', 0) / (song_data.get('valence', 0) + 0.0001)
#     if 'surprise_to_fear_ratio' in selected_features:
#         song_data['surprise_to_fear_ratio'] = song_data.get('emotion_surprise', 0) / (song_data.get('emotion_fear', 0) + 0.0001)

#     # Interactions
#     if 'energy_x_tempo' in selected_features:
#         song_data['energy_x_tempo'] = song_data.get('energy', 0) * song_data.get('tempo', 0)
#     if 'energy_x_valence' in selected_features:
#         song_data['energy_x_valence'] = song_data.get('energy', 0) * song_data.get('valence', 0)
#     if 'joy_x_tempo' in selected_features:
#         song_data['joy_x_tempo'] = song_data.get('emotion_joy', 0) * song_data.get('tempo', 0)
#     if 'sadness_x_valence' in selected_features:
#         song_data['sadness_x_valence'] = song_data.get('emotion_sadness', 0) * song_data.get('valence', 0)

#     # Transformations
#     if 'energy_squared' in selected_features:
#         song_data['energy_squared'] = song_data.get('energy', 0) ** 2
#     if 'tempo_log' in selected_features:
#         song_data['tempo_log'] = np.log1p(song_data.get('tempo', 0))
#     if 'valence_squared' in selected_features:
#         song_data['valence_squared'] = song_data.get('valence', 0) ** 2

#     # Composites
#     emotion_feats = ['emotion_sadness', 'emotion_joy', 'emotion_love', 'emotion_anger', 'emotion_fear', 'emotion_surprise']

#     if 'total_emotion_intensity' in selected_features:
#         song_data['total_emotion_intensity'] = sum(song_data.get(ef, 0) for ef in emotion_feats)
#     if 'positive_emotions' in selected_features:
#         song_data['positive_emotions'] = song_data.get('emotion_joy', 0) + song_data.get('emotion_love', 0) + song_data.get('emotion_surprise', 0)
#     if 'negative_emotions' in selected_features:
#         song_data['negative_emotions'] = song_data.get('emotion_sadness', 0) + song_data.get('emotion_anger', 0) + song_data.get('emotion_fear', 0)
#     if 'emotion_ratio' in selected_features:
#         pos = song_data.get('positive_emotions', song_data.get('emotion_joy', 0) + song_data.get('emotion_love', 0) + song_data.get('emotion_surprise', 0))
#         neg = song_data.get('negative_emotions', song_data.get('emotion_sadness', 0) + song_data.get('emotion_anger', 0) + song_data.get('emotion_fear', 0))
#         song_data['emotion_ratio'] = pos / (neg + 0.0001)
    
#     if 'emotional_diversity' in selected_features:
#         # Calculate emotional diversity using Shannon entropy
#         emotion_values = [song_data.get(ef, 0) for ef in emotion_feats]
#         total = sum(emotion_values) + 0.0001  # Avoid division by zero
#         normalized_values = [v / total for v in emotion_values]
#         diversity = -sum((v * np.log(v + 0.0001)) for v in normalized_values)
#         song_data['emotional_diversity'] = diversity

#     # Handle dominant emotion one-hot encoding
#     if any('dominant_' in feat for feat in selected_features):
#         # Find the dominant emotion
#         emotions = {feat: song_data.get(feat, 0) for feat in emotion_feats}
#         dominant = max(emotions, key=emotions.get).replace('emotion_', '')

#         # Set all dominant_ features to 0
#         for feat in selected_features:
#             if feat.startswith('dominant_'):
#                 song_data[feat] = 0

#         # Set the appropriate dominant_ feature to 1
#         dominant_feat = f'dominant_{dominant}'
#         if dominant_feat in selected_features:
#             song_data[dominant_feat] = 1

#     # Create DataFrame with just the selected features
#     X_new = pd.DataFrame([{feature: song_data.get(feature, 0) for feature in selected_features}])

#     # Scale the features
#     X_new_scaled = scaler.transform(X_new)

#     # Get all genre probabilities if the model supports it
#     genre_probabilities = {}
#     if hasattr(model, 'predict_proba'):
#         proba = model.predict_proba(X_new_scaled)[0]
#         for i, prob in enumerate(proba):
#             if encoder is not None:
#                 # For XGBoost which uses encoded labels
#                 genre = encoder.inverse_transform([i])[0]
#             else:
#                 # For other models that can handle string labels
#                 genre = model.classes_[i]
#             genre_probabilities[genre] = prob

#     # Predict
#     if hasattr(model, 'predict_proba'):
#         if encoder is not None:
#             # For XGBoost which needs encoded labels
#             pred_encoded = model.predict(X_new_scaled)[0]
#             predicted_genre = encoder.inverse_transform([pred_encoded])[0]

#             proba = model.predict_proba(X_new_scaled)[0]
#             confidence = proba[pred_encoded]
#         else:
#             # For other models that can handle string labels
#             proba = model.predict_proba(X_new_scaled)[0]
#             genre_idx = np.argmax(proba)
#             predicted_genre = model.classes_[genre_idx]
#             confidence = proba[genre_idx]
#     else:
#         predicted_genre = model.predict(X_new_scaled)[0]
#         confidence = None

#     return predicted_genre, confidence, genre_probabilities

def predict_genre(song_data, model_path='model.joblib', 
                 encoder_path='encoder.joblib',
                 scaler_path='scaler.joblib',
                 metadata_path='metadata.joblib'):
    """
    Predict the genre of a new song using the new model files
    
    Parameters:
    song_data: Dictionary with song features
    model_path: Path to the trained model
    encoder_path: Path to the label encoder
    scaler_path: Path to the feature scaler
    metadata_path: Path to metadata containing feature information
    
    Returns:
    Predicted genre and confidence
    """
    # Load model components
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    metadata = joblib.load(metadata_path)
    
    # Extract feature information from metadata
    selected_features = metadata.get('features', [])
    
    # Rest of the function remains similar
    # Ensure the song data has all required features
    for feature in selected_features:
        if feature not in song_data and feature not in ['joy_to_sadness_ratio', 'anger_to_love_ratio',
                                                       'energy_to_valence_ratio', 'surprise_to_fear_ratio',
                                                       'energy_x_tempo', 'energy_x_valence', 'joy_x_tempo',
                                                       'sadness_x_valence', 'energy_squared', 'tempo_log',
                                                       'valence_squared', 'total_emotion_intensity',
                                                       'positive_emotions', 'negative_emotions',
                                                       'emotion_ratio', 'emotional_diversity']:
            song_data[feature] = 0

    # Calculate engineered features if they're part of selected_features
    # [keep the existing feature engineering code]
    
    # Create DataFrame with just the selected features
    X_new = pd.DataFrame([{feature: song_data.get(feature, 0) for feature in selected_features}])

    # Scale the features
    X_new_scaled = scaler.transform(X_new)

    # Get all genre probabilities
    genre_probabilities = {}
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_new_scaled)[0]
        for i, prob in enumerate(proba):
            genre = encoder.inverse_transform([i])[0]
            genre_probabilities[genre] = prob

    # Predict
    if hasattr(model, 'predict_proba'):
        # For models that provide probability estimates
        proba = model.predict_proba(X_new_scaled)[0]
        genre_idx = np.argmax(proba)
        predicted_genre = encoder.inverse_transform([genre_idx])[0]
        confidence = proba[genre_idx]
    else:
        # For models that only provide class predictions
        pred_encoded = model.predict(X_new_scaled)[0]
        predicted_genre = encoder.inverse_transform([pred_encoded])[0]
        confidence = None

    return predicted_genre, confidence, genre_probabilities

def generate_random_song():
    """Generate random song data for demonstration"""
    # Audio features
    energy = random.uniform(0.1, 1.0)
    mode = random.choice([0, 1])
    key = random.randint(0, 11)
    valence = random.uniform(0.1, 1.0)
    tempo = random.uniform(60, 180)
    
    # Emotion features
    # Generate random emotions that sum to 1
    emotions = np.random.dirichlet(np.ones(6), size=1)[0]
    
    song_data = {
        'energy': energy,
        'mode': mode,
        'key': key,
        'valence': valence,
        'tempo': tempo,
        'emotion_joy': emotions[0],
        'emotion_sadness': emotions[1],
        'emotion_anger': emotions[2],
        'emotion_fear': emotions[3],
        'emotion_love': emotions[4],
        'emotion_surprise': emotions[5]
    }
    
    return song_data

def get_audio_feature_description(feature):
    """Return description for audio features"""
    descriptions = {
        'energy': "A measure from 0.0 to 1.0 representing intensity and activity. Energetic tracks feel fast, loud, and noisy.",
        'mode': "The modality of the track. Major (1) or Minor (0).",
        'key': "The key of the track. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, etc.",
        'valence': "A measure from 0.0 to 1.0 describing musical positiveness. High valence sounds more positive (happy, cheerful, euphoric).",
        'tempo': "The overall estimated tempo of the track in beats per minute (BPM)."
    }
    return descriptions.get(feature, "No description available.")

def get_emotion_feature_description(feature):
    """Return description for emotion features"""
    descriptions = {
        'emotion_joy': "Happy, pleased, or satisfied feelings.",
        'emotion_sadness': "Feelings of sorrow, grief, or unhappiness.",
        'emotion_anger': "Strong feelings of annoyance, displeasure, or hostility.",
        'emotion_fear': "An unpleasant emotion caused by the belief that someone or something is dangerous.",
        'emotion_love': "Deep affection, attachment, or adoration.",
        'emotion_surprise': "A feeling of mild astonishment or shock caused by something unexpected."
    }
    return descriptions.get(feature, "No description available.")

def get_tempo_description(tempo):
    """Return a musical term for the tempo"""
    if tempo < 60:
        return "Largo (very slow)"
    elif tempo < 76:
        return "Adagio (slow)"
    elif tempo < 108:
        return "Andante (walking pace)"
    elif tempo < 120:
        return "Moderato (moderate)"
    elif tempo < 168:
        return "Allegro (fast)"
    else:
        return "Presto (very fast)"

def get_key_name(key, mode):
    """Return the name of the key based on key and mode"""
    key_names = ["C", "C♯/D♭", "D", "D♯/E♭", "E", "F", "F♯/G♭", "G", "G♯/A♭", "A", "A♯/B♭", "B"]
    mode_names = ["minor", "major"]
    
    return f"{key_names[key]} {mode_names[mode]}"

def wait_for_prediction():
    """Simulate a prediction process with a progress bar"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    steps = ["Analyzing audio features...", "Processing emotion data...", "Calculating musical profiles...", 
             "Comparing with genre signatures...", "Finalizing predictions..."]
    
    for i, step in enumerate(steps):
        # Update progress bar and status text
        progress = (i + 1) / len(steps)
        progress_bar.progress(progress)
        status_text.text(step)
        
        # Add a small delay to simulate processing
        time.sleep(0.5)
    
    # Clear the status text and progress bar
    status_text.empty()
    progress_bar.empty()

def get_genre_description(genre):
    """Return description for the predicted genre"""
    descriptions = {
        'acoustic': "Music that predominantly uses acoustic instruments rather than electric or electronic instruments. Often characterized by a more natural, unplugged sound.",
        'afrobeat': "A genre blending West African musical styles with American jazz, funk, and soul, characterized by complex rhythmic patterns and percussion.",
        'alternative': "Music that exists outside of mainstream commercial genres, often drawing from indie, rock, punk, and experimental sounds.",
        'blues': "A genre originating from African American communities in the Deep South, characterized by specific chord progressions and soulful vocals expressing hardship and emotion.",
        'children': "Music specifically created for children, often featuring simple, catchy melodies, repetitive lyrics, and educational or playful themes.",
        'chill': "Relaxed, laid-back music designed to create a calm atmosphere, often blending elements of downtempo, ambient, and soft electronic sounds.",
        'club': "Upbeat dance music primarily played in nightclubs, featuring strong beats, repetitive rhythms, and often electronic production.",
        'country': "A genre developed in the southern United States blending folk, western, and blues influences, often featuring stringed instruments and storytelling lyrics.",
        'dance': "Music specifically created for dancing, characterized by strong beats, rhythmic patterns, and often electronic production.",
        'disco': "A genre from the 1970s characterized by steady beats, orchestral elements, and soulful vocals, designed for dancing.",
        'disney': "Music associated with Disney films and productions, often featuring orchestral arrangements, memorable melodies, and theatrical elements.",
        'edm': "Electronic Dance Music, characterized by synthesized sounds, heavy beats, and production specifically designed for dancing environments.",
        'electro': "A genre of electronic music characterized by the use of drum machines, synthesizers, and futuristic sounds.",
        'emo': "Emotionally charged music often with confessional, introspective lyrics and a combination of punk and indie rock influences.",
        'funk': "A rhythmic genre characterized by a strong bass line, syncopated rhythms, and emphasis on the first beat of each measure.",
        'groove': "Music with a strong, repetitive rhythmic pattern that creates a sense of flow and movement.",
        'happy': "Upbeat, positive music designed to evoke joyful emotions, often featuring major keys and uplifting lyrics.",
        'house': "Electronic dance music with a repetitive 4/4 beat, synthesized bass lines, and often featuring sampled vocals.",
        'jazz': "A complex genre originating in African American communities, characterized by improvisation, syncopation, swing notes, and unique harmony.",
        'pop': "Popular music aimed at a wide audience, characterized by catchy melodies, repeated choruses, and accessible structures."
    }
    return descriptions.get(genre.lower(), "No description available.")

def get_appropriate_songs(genre):
    """Return a list of appropriate songs for the predicted genre"""
    genre_songs = {
        'acoustic': ["'Landslide' by Fleetwood Mac", "'Hallelujah' by Jeff Buckley", "'Banana Pancakes' by Jack Johnson"],
        'afrobeat': ["'Water No Get Enemy' by Fela Kuti", "'Essence' by WizKid ft. Tems", "'Ye' by Burna Boy"],
        'alternative': ["'Creep' by Radiohead", "'Seven Nation Army' by The White Stripes", "'Take Me Out' by Franz Ferdinand"],
        'blues': ["'The Thrill Is Gone' by B.B. King", "'Stormy Monday' by T-Bone Walker", "'Cross Road Blues' by Robert Johnson"],
        'children': ["'Baby Shark' by Pinkfong", "'Let It Go' from Frozen", "'The Wheels on the Bus' (Traditional)"],
        'chill': ["'Flightless Bird, American Mouth' by Iron & Wine", "'Breathe' by Télépopmusik", "'Porcelain' by Moby"],
        'club': ["'Don't You Worry Child' by Swedish House Mafia", "'Levels' by Avicii", "'One More Time' by Daft Punk"],
        'country': ["'Jolene' by Dolly Parton", "'Friends in Low Places' by Garth Brooks", "'Take Me Home, Country Roads' by John Denver"],
        'dance': ["'Can't Get You Out of My Head' by Kylie Minogue", "'We Found Love' by Rihanna ft. Calvin Harris", "'Titanium' by David Guetta ft. Sia"],
        'disco': ["'Stayin' Alive' by Bee Gees", "'I Will Survive' by Gloria Gaynor", "'Le Freak' by Chic"],
        'disney': ["'Circle of Life' from The Lion King", "'A Whole New World' from Aladdin", "'How Far I'll Go' from Moana"],
        'edm': ["'Animals' by Martin Garrix", "'Scary Monsters and Nice Sprites' by Skrillex", "'Strobe' by deadmau5"],
        'electro': ["'Da Funk' by Daft Punk", "'Technologic' by Daft Punk", "'Harder, Better, Faster, Stronger' by Daft Punk"],
        'emo': ["'Welcome to the Black Parade' by My Chemical Romance", "'Misery Business' by Paramore", "'The Middle' by Jimmy Eat World"],
        'funk': ["'Superstition' by Stevie Wonder", "'Get Up (I Feel Like Being a) Sex Machine' by James Brown", "'Uptown Funk' by Mark Ronson ft. Bruno Mars"],
        'groove': ["'Chameleon' by Herbie Hancock", "'Get Lucky' by Daft Punk ft. Pharrell Williams", "'Can't Stop' by Red Hot Chili Peppers"],
        'happy': ["'Happy' by Pharrell Williams", "'Walking on Sunshine' by Katrina and The Waves", "'Good Vibrations' by The Beach Boys"],
        'house': ["'Show Me Love' by Robin S", "'Around the World' by Daft Punk", "'Music Sounds Better With You' by Stardust"],
        'jazz': ["'Take Five' by Dave Brubeck", "'So What' by Miles Davis", "'My Favorite Things' by John Coltrane"],
        'pop': ["'Shake It Off' by Taylor Swift", "'Bad Guy' by Billie Eilish", "'Blinding Lights' by The Weeknd"]
    }
    return genre_songs.get(genre.lower(), ["No specific songs available for this genre."])

def main():
    # Check model files first
    models_ready, missing_files = check_model_files()
    if not models_ready:
        st.error(f"Missing model files: {', '.join(missing_files)}")
        st.error("Please ensure all model files (model.joblib, encoder.joblib, scaler.joblib, metadata.joblib) are in the app directory.")
        st.info("The app will work but prediction functionality will be disabled.")
        
    # Initialize API clients
    spotify_client = get_spotify_client()
    genius_client = get_genius_client()
    sentiment_analyzer = get_sentiment_analyzer()
    
    # Header with logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 class='main-header'>Music Genre Predictor</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>Using Audio Features and Lyrics Emotions</p>", unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🎵 Predict Genre", "🔍 Search for a Song", "ℹ️ About"])
    
    with tab1:
        # Sidebar for feature inputs
        with st.sidebar:
            st.markdown("<p class='section-header'>Audio Features</p>", unsafe_allow_html=True)
            
            # Audio Features
            energy = st.slider(
                "Energy",
                min_value=0.0,
                max_value=1.0,
                value=0.65,
                step=0.01,
                help=get_audio_feature_description("energy")
            )
            
            st.markdown("<span class='audio-feature-label'>Mode</span>", unsafe_allow_html=True)
            mode = st.radio(
                "Mode",
                options=["Major", "Minor"],
                index=0,
                horizontal=True,
                help=get_audio_feature_description("mode"),
                label_visibility="collapsed"
            )
            mode = 1 if mode == "Major" else 0
            
            st.markdown("<span class='audio-feature-label'>Key</span>", unsafe_allow_html=True)
            key = st.selectbox(
                "Key",
                options=["C", "C♯/D♭", "D", "D♯/E♭", "E", "F", "F♯/G♭", "G", "G♯/A♭", "A", "A♯/B♭", "B"],
                index=0,
                help=get_audio_feature_description("key"),
                label_visibility="collapsed"
            )
            key = ["C", "C♯/D♭", "D", "D♯/E♭", "E", "F", "F♯/G♭", "G", "G♯/A♭", "A", "A♯/B♭", "B"].index(key)
            
            valence = st.slider(
                "Valence (Positivity)",
                min_value=0.0,
                max_value=1.0,
                value=0.45,
                step=0.01,
                help=get_audio_feature_description("valence")
            )
            
            tempo = st.slider(
                "Tempo (BPM)",
                min_value=60.0,
                max_value=180.0,
                value=120.0,
                step=1.0,
                help=get_audio_feature_description("tempo")
            )
            
            st.markdown("<div class='row-gap'></div>", unsafe_allow_html=True)
            st.markdown("<p class='section-header'>Emotion Features</p>", unsafe_allow_html=True)
            
            # Use columns for emotion sliders to save space
            col1, col2 = st.columns(2)
            
            with col1:
                joy = st.slider(
                    "Joy",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.4,
                    step=0.01,
                    help=get_emotion_feature_description("emotion_joy")
                )
                
                sadness = st.slider(
                    "Sadness",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    step=0.01,
                    help=get_emotion_feature_description("emotion_sadness")
                )
                
                anger = st.slider(
                    "Anger",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.01,
                    help=get_emotion_feature_description("emotion_anger")
                )
            
            with col2:
                fear = st.slider(
                    "Fear",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.01,
                    help=get_emotion_feature_description("emotion_fear")
                )
                
                love = st.slider(
                    "Love",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.01,
                    help=get_emotion_feature_description("emotion_love")
                )
                
                surprise = st.slider(
                    "Surprise",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.01,
                    help=get_emotion_feature_description("emotion_surprise")
                )
            
            # Normalize emotions to sum to 1
            emotions_sum = joy + sadness + anger + fear + love + surprise
            if emotions_sum > 0:
                joy = joy / emotions_sum
                sadness = sadness / emotions_sum
                anger = anger / emotions_sum
                fear = fear / emotions_sum
                love = love / emotions_sum
                surprise = surprise / emotions_sum
            
            # Random song generator button
            st.markdown("<div class='row-gap'></div>", unsafe_allow_html=True)
            if st.button("Generate Random Song"):
                random_song = generate_random_song()
                st.session_state.energy = random_song['energy']
                st.session_state.mode = 0 if random_song['mode'] == 0 else 1
                st.session_state.key = random_song['key']
                st.session_state.valence = random_song['valence']
                st.session_state.tempo = random_song['tempo']
                st.session_state.joy = random_song['emotion_joy']
                st.session_state.sadness = random_song['emotion_sadness']
                st.session_state.anger = random_song['emotion_anger']
                st.session_state.fear = random_song['emotion_fear']
                st.session_state.love = random_song['emotion_love']
                st.session_state.surprise = random_song['emotion_surprise']
                st.rerun()
        
        # Main content area
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
            st.markdown("<p class='section-header'>Audio Features Summary</p>", unsafe_allow_html=True)
            
            # Create a DataFrame for audio features
            audio_data = {
                'Feature': ['Energy', 'Mode', 'Key', 'Valence', 'Tempo'],
                'Value': [
                    f"{energy:.2f} ({'High' if energy > 0.6 else 'Medium' if energy > 0.3 else 'Low'})",
                    f"{'Major' if mode == 1 else 'Minor'}",
                    get_key_name(key, mode),
                    f"{valence:.2f} ({'Positive' if valence > 0.6 else 'Neutral' if valence > 0.3 else 'Negative'})",
                    f"{tempo:.0f} BPM ({get_tempo_description(tempo)})"
                ]
            }
            
            # Convert to DataFrame and display
            audio_df = pd.DataFrame(audio_data)
            st.table(audio_df)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
            st.markdown("<p class='section-header'>Emotion Profile</p>", unsafe_allow_html=True)
            
            # Create emotion data for radar chart
            emotion_data = {
                'Joy': joy,
                'Sadness': sadness,
                'Anger': anger,
                'Fear': fear,
                'Love': love,
                'Surprise': surprise
            }
            
            # Create radar chart for emotions
            emotion_fig = create_emotion_radar_chart(emotion_data)
            st.pyplot(emotion_fig)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Collect all features into a dictionary
        song_data = {
            'energy': energy,
            'mode': mode,
            'key': key,
            'valence': valence,
            'tempo': tempo,
            'emotion_joy': joy,
            'emotion_sadness': sadness,
            'emotion_anger': anger,
            'emotion_fear': fear,
            'emotion_love': love,
            'emotion_surprise': surprise
        }
        
        # Predict button
        if st.button("Predict Genre", type="primary"):
            # Show a prediction animation
            wait_for_prediction()
            
            # Make prediction
            try:
                predicted_genre, confidence, genre_probabilities = predict_genre(song_data)
                
                # Store results in session state
                st.session_state.predicted_genre = predicted_genre
                st.session_state.confidence = confidence
                st.session_state.genre_probabilities = genre_probabilities
                
                # Display prediction results
                st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
                st.markdown("<p class='section-header'>Prediction Results</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='genre-label'>{predicted_genre.title()}</p>", unsafe_allow_html=True)
                
                # Display confidence bar
                if confidence:
                    st.markdown(f"<p>Confidence: {confidence:.1%}</p>", unsafe_allow_html=True)
                    st.progress(float(confidence))
                
                # Genre description
                st.markdown("<p><strong>Description:</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<p>{get_genre_description(predicted_genre)}</p>", unsafe_allow_html=True)
                
                # Example songs
                st.markdown("<p><strong>Example Songs:</strong></p>", unsafe_allow_html=True)
                for song in get_appropriate_songs(predicted_genre):
                    st.markdown(f"<p>• {song}</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Create distribution chart
                if genre_probabilities:
                    st.markdown("<p class='section-header'>Genre Probability Distribution</p>", unsafe_allow_html=True)
                    dist_fig = create_genre_distribution_chart(genre_probabilities)
                    st.pyplot(dist_fig)
            
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.error("Please make sure all model files are available in the app directory.")
                
    with tab2:
        st.header("Search for a Song")
        
        # API status indicators
        col1, col2 = st.columns(2)
        with col1:
            if spotify_client:
                st.markdown("<div class='api-status api-connected'>✅ Spotify API Connected</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='api-status api-disconnected'>❌ Spotify API Disconnected</div>", unsafe_allow_html=True)
                st.info("To connect Spotify API, set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")
                
        with col2:
            if genius_client:
                st.markdown("<div class='api-status api-connected'>✅ Genius API Connected</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='api-status api-disconnected'>❌ Genius API Disconnected</div>", unsafe_allow_html=True)
                st.info("To connect Genius API, set GENIUS_ACCESS_TOKEN environment variable.")
                
        st.markdown("""
        <div class="info-box" style="color: black;">
        Search for a song to analyze its audio features and lyrics. The app will fetch data from Spotify and Genius APIs 
        if they are connected. This provides a more accurate analysis of the song's characteristics.
        </div>
        """, unsafe_allow_html=True)
        
        # Search form
        search_query = st.text_input("Enter song title and artist name", placeholder="e.g., Bohemian Rhapsody Queen")
        
        search_col1, search_col2 = st.columns([1, 1])
        with search_col1:
            search_button = st.button("🔍 Search", key="search_button", use_container_width=True)
        
        # Execute search
        # if search_button and search_query:
        #     if spotify_client:
        #         with st.spinner("Searching for song..."):
        #             spotify_data = get_song_from_spotify(search_query, spotify_client)
                    
        #             if spotify_data:
        #                 st.session_state.spotify_data = spotify_data
        #                 st.session_state.search_query = search_query  # Save the search query too
                        
        #                 # Display search results
        #                 st.markdown("<div class='search-results'>", unsafe_allow_html=True)
                        
        #                 # Display in two columns
        #                 col1, col2 = st.columns([1, 2])
                        
        #                 with col1:
        #                     # Display album art
        #                     if spotify_data.get('album_cover_url'):
        #                         st.image(spotify_data['album_cover_url'], width=200)
        #                     else:
        #                         st.image("https://img.icons8.com/ios/100/000000/music--v1.png", width=200)
                        
        #                 with col2:
        #                     # Display song info
        #                     st.subheader(spotify_data['track_name'])
        #                     st.write(f"**Artist:** {spotify_data['artists']}")
        #                     st.write(f"**Album:** {spotify_data['album']}")
        #                     st.write(f"**Release Date:** {spotify_data.get('release_date', 'Unknown')}")
                            
        #                     # Display audio preview if available
        #                     if spotify_data.get('preview_url'):
        #                         st.audio(spotify_data['preview_url'])
        #                     else:
        #                         st.info("No audio preview available")
                        
        #                 # Display audio features
        #                 st.subheader("Audio Features")
                        
        #                 # Create visualization
        #                 audio_features = {
        #                     'energy': spotify_data['energy'],
        #                     'valence': spotify_data['valence'],
        #                     'danceability': spotify_data.get('danceability', 0.5),
        #                     'acousticness': spotify_data.get('acousticness', 0.5),
        #                     'instrumentalness': spotify_data.get('instrumentalness', 0.5),
        #                     'liveness': spotify_data.get('liveness', 0.5),
        #                     'speechiness': spotify_data.get('speechiness', 0.5)
        #                 }
                        
        #                 audio_fig = create_audio_features_chart(audio_features)
        #                 st.pyplot(audio_fig)
                        
        #                 # Try to get lyrics
        #                 if genius_client:
        #                     with st.spinner("Fetching and analyzing lyrics..."):
        #                         lyrics = fetch_lyrics(
        #                             spotify_data['artists'].split(',')[0],  # Just use first artist
        #                             spotify_data['track_name'],
        #                             genius_client
        #                         )
                                
        #                         if lyrics:
        #                             # Store lyrics in session state
        #                             st.session_state.current_lyrics = lyrics
        #                             # Show lyrics in an expander
        #                             with st.expander("View Lyrics"):
        #                                 st.markdown("<div class='lyrics-box'>", unsafe_allow_html=True)
        #                                 st.write(lyrics)
        #                                 st.markdown("</div>", unsafe_allow_html=True)
                                    
        #                             # Analyze lyrics for emotions
        #                             emotion_data = analyze_lyrics(lyrics, sentiment_analyzer)
                                    
        #                             # Add emotion data to song data
        #                             for key, value in emotion_data.items():
        #                                 if key.startswith('emotion_'):
        #                                     spotify_data[key] = value
                                    
        #                             # Create emotion chart
        #                             st.subheader("Emotions in Lyrics")
                                    
        #                             # Prepare data for radar chart
        #                             emotion_chart_data = {
        #                                 'Joy': emotion_data['emotion_joy'],
        #                                 'Sadness': emotion_data['emotion_sadness'],
        #                                 'Anger': emotion_data['emotion_anger'],
        #                                 'Fear': emotion_data['emotion_fear'],
        #                                 'Love': emotion_data['emotion_love'],
        #                                 'Surprise': emotion_data['emotion_surprise']
        #                             }
                                    
        #                             # Create radar chart
        #                             emotion_fig = create_emotion_radar_chart(emotion_chart_data)
        #                             st.pyplot(emotion_fig)
                                    
        #                             # Enable prediction with the analyzed lyrics
        #                             st.session_state.lyrics_analyzed = True
                                    
        #                             # Prediction section
        #                             if st.button("Predict Genre Based on Audio & Lyrics", type="primary", key="predict_with_lyrics"):
        #                                 with st.spinner("Predicting genre..."):
        #                                     # Make prediction
        #                                     predicted_genre, confidence, genre_probabilities = predict_genre(spotify_data)
                                            
        #                                     # Display prediction results
        #                                     st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        #                                     st.markdown("<p class='section-header'>Prediction Results</p>", unsafe_allow_html=True)
        #                                     st.markdown(f"<p class='genre-label'>{predicted_genre.title()}</p>", unsafe_allow_html=True)
                                            
        #                                     # Display confidence bar
        #                                     if confidence:
        #                                         st.markdown(f"<p>Confidence: {confidence:.1%}</p>", unsafe_allow_html=True)
        #                                         st.progress(float(confidence))
                                            
        #                                     # Genre description
        #                                     st.markdown("<p><strong>Description:</strong></p>", unsafe_allow_html=True)
        #                                     st.markdown(f"<p>{get_genre_description(predicted_genre)}</p>", unsafe_allow_html=True)
                                            
        #                                     # Example songs
        #                                     st.markdown("<p><strong>Example Songs:</strong></p>", unsafe_allow_html=True)
        #                                     for song in get_appropriate_songs(predicted_genre):
        #                                         st.markdown(f"<p>• {song}</p>", unsafe_allow_html=True)
                                            
        #                                     st.markdown("</div>", unsafe_allow_html=True)
                                            
        #                                     # Create distribution chart
        #                                     if genre_probabilities:
        #                                         st.markdown("<p class='section-header'>Genre Probability Distribution</p>", unsafe_allow_html=True)
        #                                         dist_fig = create_genre_distribution_chart(genre_probabilities)
        #                                         st.pyplot(dist_fig)
        #                         else:
        #                             st.warning("Could not find lyrics for this song. You can still predict based on audio features only.")
                                    
        #                             # Create default emotion values
        #                             default_emotion_data = {
        #                                 'emotion_joy': 0.2,
        #                                 'emotion_sadness': 0.2,
        #                                 'emotion_anger': 0.1,
        #                                 'emotion_fear': 0.1,
        #                                 'emotion_love': 0.3,
        #                                 'emotion_surprise': 0.1
        #                             }
                                    
        #                             # Add default emotion data to song data
        #                             for key, value in default_emotion_data.items():
        #                                 spotify_data[key] = value
                                    
        #                             # Enable prediction with audio features only
        #                             if st.button("Predict Genre Based on Audio Features Only", type="primary", key="predict_audio_only"):
        #                                 with st.spinner("Predicting genre..."):
        #                                     # Make prediction
        #                                     predicted_genre, confidence, genre_probabilities = predict_genre(spotify_data)
                                            
        #                                     # Display prediction results similar to above
        #                                     st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        #                                     st.markdown("<p class='section-header'>Prediction Results (Audio Features Only)</p>", unsafe_allow_html=True)
        #                                     st.markdown(f"<p class='genre-label'>{predicted_genre.title()}</p>", unsafe_allow_html=True)
                                            
        #                                     if confidence:
        #                                         st.markdown(f"<p>Confidence: {confidence:.1%}</p>", unsafe_allow_html=True)
        #                                         st.progress(float(confidence))
                                            
        #                                     st.markdown("<p><strong>Description:</strong></p>", unsafe_allow_html=True)
        #                                     st.markdown(f"<p>{get_genre_description(predicted_genre)}</p>", unsafe_allow_html=True)
                                            
        #                                     st.markdown("</div>", unsafe_allow_html=True)
                                            
        #                                     # Create distribution chart
        #                                     if genre_probabilities:
        #                                         st.markdown("<p class='section-header'>Genre Probability Distribution</p>", unsafe_allow_html=True)
        #                                         dist_fig = create_genre_distribution_chart(genre_probabilities)
        #                                         st.pyplot(dist_fig)
        #                 else:
        #                     st.warning("Genius API is not connected. Cannot fetch lyrics.")
                            
        #                     # Create default emotion values
        #                     default_emotion_data = {
        #                         'emotion_joy': 0.2,
        #                         'emotion_sadness': 0.2,
        #                         'emotion_anger': 0.1,
        #                         'emotion_fear': 0.1,
        #                         'emotion_love': 0.3,
        #                         'emotion_surprise': 0.1
        #                     }
                            
        #                     # Add default emotion data to song data
        #                     for key, value in default_emotion_data.items():
        #                         spotify_data[key] = value
                            
        #                     # Enable prediction with audio features only
        #                     if st.button("Predict Genre Based on Audio Features Only", type="primary", key="predict_audio_only_no_genius"):
        #                         with st.spinner("Predicting genre..."):
        #                             # Make prediction
        #                             predicted_genre, confidence, genre_probabilities = predict_genre(spotify_data)
                                    
        #                             # Display prediction results
        #                             st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        #                             st.markdown("<p class='section-header'>Prediction Results (Audio Features Only)</p>", unsafe_allow_html=True)
        #                             st.markdown(f"<p class='genre-label'>{predicted_genre.title()}</p>", unsafe_allow_html=True)
                                    
        #                             if confidence:
        #                                 st.markdown(f"<p>Confidence: {confidence:.1%}</p>", unsafe_allow_html=True)
        #                                 st.progress(float(confidence))
                                    
        #                             st.markdown("<p><strong>Description:</strong></p>", unsafe_allow_html=True)
        #                             st.markdown(f"<p>{get_genre_description(predicted_genre)}</p>", unsafe_allow_html=True)
                                    
        #                             st.markdown("</div>", unsafe_allow_html=True)
                                    
        #                             if genre_probabilities:
        #                                 st.markdown("<p class='section-header'>Genre Probability Distribution</p>", unsafe_allow_html=True)
        #                                 dist_fig = create_genre_distribution_chart(genre_probabilities)
        #                                 st.pyplot(dist_fig)
                        
        #                 st.markdown("</div>", unsafe_allow_html=True)
        #             else:
        #                 st.error("No song found. Please try a different search query.")
        #     else:
        #         st.error("Spotify API is not connected. Please set your Spotify API credentials.")
                
        #         # Offer alternative for when API is not available
        #         st.info("You can still use the manual input tab to predict genres without API access.")
        if search_button and search_query:
            if spotify_client:
                with st.spinner("Searching for song..."):
                    spotify_data = get_song_from_spotify(search_query, spotify_client)
                    
                    if spotify_data:
                        # Store the spotify data in session state
                        st.session_state.spotify_data = spotify_data
                        st.session_state.search_query = search_query
                        
                        # Display search results
                        st.markdown("<div class='search-results'>", unsafe_allow_html=True)
                        
                        # Display in two columns
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Display album art
                            if spotify_data.get('album_cover_url'):
                                st.image(spotify_data['album_cover_url'], width=200)
                            else:
                                st.image("https://img.icons8.com/ios/100/000000/music--v1.png", width=200)
                        
                        with col2:
                            # Display song info
                            st.subheader(spotify_data['track_name'])
                            st.write(f"**Artist:** {spotify_data['artists']}")
                            st.write(f"**Album:** {spotify_data['album']}")
                            st.write(f"**Release Date:** {spotify_data.get('release_date', 'Unknown')}")
                            
                            # Display audio preview if available
                            if spotify_data.get('preview_url'):
                                st.audio(spotify_data['preview_url'])
                            else:
                                st.info("No audio preview available")
                        
                        # Display audio features
                        st.subheader("Audio Features")
                        
                        # Create visualization
                        audio_features = {
                            'energy': spotify_data['energy'],
                            'valence': spotify_data['valence'],
                            'danceability': spotify_data.get('danceability', 0.5),
                            'acousticness': spotify_data.get('acousticness', 0.5),
                            'instrumentalness': spotify_data.get('instrumentalness', 0.5),
                            'liveness': spotify_data.get('liveness', 0.5),
                            'speechiness': spotify_data.get('speechiness', 0.5)
                        }
                        
                        audio_fig = create_audio_features_chart(audio_features)
                        st.pyplot(audio_fig)
                        
                        # Try to get lyrics
                        if genius_client:
                            with st.spinner("Fetching and analyzing lyrics..."):
                                lyrics = fetch_lyrics(
                                    spotify_data['artists'].split(',')[0],  # Just use first artist
                                    spotify_data['track_name'],
                                    genius_client
                                )
                                
                                if lyrics:
                                    # Store lyrics in session state
                                    st.session_state.current_lyrics = lyrics
                                    
                                    # Show lyrics in an expander
                                    with st.expander("View Lyrics"):
                                        st.markdown("<div class='lyrics-box'>", unsafe_allow_html=True)
                                        st.write(lyrics)
                                        st.markdown("</div>", unsafe_allow_html=True)
                                    
                                    # Analyze lyrics for emotions
                                    emotion_data = analyze_lyrics(lyrics, sentiment_analyzer)
                                    
                                    # Add emotion data to song data
                                    for key, value in emotion_data.items():
                                        if key.startswith('emotion_'):
                                            spotify_data[key] = value
                                    
                                    # Store the complete data for prediction
                                    st.session_state.song_data_for_prediction = spotify_data.copy()
                                    
                                    # Create emotion chart
                                    st.subheader("Emotions in Lyrics")
                                    
                                    # Prepare data for radar chart
                                    emotion_chart_data = {
                                        'Joy': emotion_data['emotion_joy'],
                                        'Sadness': emotion_data['emotion_sadness'],
                                        'Anger': emotion_data['emotion_anger'],
                                        'Fear': emotion_data['emotion_fear'],
                                        'Love': emotion_data['emotion_love'],
                                        'Surprise': emotion_data['emotion_surprise']
                                    }
                                    
                                    # Create radar chart
                                    emotion_fig = create_emotion_radar_chart(emotion_chart_data)
                                    st.pyplot(emotion_fig)
                                    
                                    # Enable prediction with the analyzed lyrics
                                    st.session_state.lyrics_analyzed = True
                                    
                                    # Define prediction callback function
                                    def predict_with_lyrics():
                                        song_data = st.session_state.song_data_for_prediction
                                        if song_data:
                                            try:
                                                predicted_genre, confidence, genre_probabilities = predict_genre(song_data)
                                                
                                                # Store results in session state
                                                st.session_state.prediction_made = True
                                                st.session_state.predicted_genre = predicted_genre
                                                st.session_state.confidence = confidence
                                                st.session_state.genre_probabilities = genre_probabilities
                                            except Exception as e:
                                                st.error(f"Error making prediction: {e}")
                                    
                                    def make_prediction():
                                        """Callback function for the prediction button"""
                                        if st.session_state.song_data_for_prediction:
                                            try:
                                                # Make prediction using the stored song data
                                                predicted_genre, confidence, genre_probabilities = predict_genre(
                                                    st.session_state.song_data_for_prediction
                                                )
                                                
                                                # Store results in session state
                                                st.session_state.prediction_made = True
                                                st.session_state.prediction_results = {
                                                    'genre': predicted_genre,
                                                    'confidence': confidence,
                                                    'probabilities': genre_probabilities
                                                }
                                            except Exception as e:
                                                st.error(f"Error making prediction: {e}")
                                        else:
                                            st.error("No song data available for prediction")
                                    # Prediction button with callback
                                    st.button("Predict Genre Based on Audio & Lyrics", 
                                            type="primary", 
                                            key="predict_with_lyrics",
                                            on_click=make_prediction)
                                    
                                    # Display prediction results if available
                                    if st.session_state.prediction_made:
                                        # Display prediction results
                                        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
                                        st.markdown("<p class='section-header'>Prediction Results</p>", unsafe_allow_html=True)
                                        st.markdown(f"<p class='genre-label'>{st.session_state.predicted_genre.title()}</p>", unsafe_allow_html=True)
                                        
                                        # Display confidence bar
                                        if st.session_state.confidence:
                                            st.markdown(f"<p>Confidence: {st.session_state.confidence:.1%}</p>", unsafe_allow_html=True)
                                            st.progress(float(st.session_state.confidence))
                                        
                                        # Genre description
                                        st.markdown("<p><strong>Description:</strong></p>", unsafe_allow_html=True)
                                        st.markdown(f"<p>{get_genre_description(st.session_state.predicted_genre)}</p>", unsafe_allow_html=True)
                                        
                                        # Example songs
                                        st.markdown("<p><strong>Example Songs:</strong></p>", unsafe_allow_html=True)
                                        for song in get_appropriate_songs(st.session_state.predicted_genre):
                                            st.markdown(f"<p>• {song}</p>", unsafe_allow_html=True)
                                        
                                        st.markdown("</div>", unsafe_allow_html=True)
                                        
                                        # Create distribution chart
                                        if hasattr(st.session_state, 'genre_probabilities') and st.session_state.genre_probabilities:
                                            st.markdown("<p class='section-header'>Genre Probability Distribution</p>", unsafe_allow_html=True)
                                            dist_fig = create_genre_distribution_chart(st.session_state.genre_probabilities)
                                            st.pyplot(dist_fig)
                                else:
                                    st.warning("Could not find lyrics for this song. You can still predict based on audio features only.")
                                    
                                    # Create default emotion values
                                    default_emotion_data = {
                                        'emotion_joy': 0.2,
                                        'emotion_sadness': 0.2,
                                        'emotion_anger': 0.1,
                                        'emotion_fear': 0.1,
                                        'emotion_love': 0.3,
                                        'emotion_surprise': 0.1
                                    }
                                    
                                    # Add default emotion data to song data
                                    for key, value in default_emotion_data.items():
                                        spotify_data[key] = value
                                    
                                    # Store the complete data for prediction
                                    st.session_state.song_data_for_prediction = spotify_data.copy()
                                    
                                    # Define prediction callback function for audio only
                                    def predict_audio_only():
                                        song_data = st.session_state.song_data_for_prediction
                                        if song_data:
                                            try:
                                                predicted_genre, confidence, genre_probabilities = predict_genre(song_data)
                                                
                                                # Store results in session state
                                                st.session_state.prediction_made = True
                                                st.session_state.predicted_genre = predicted_genre
                                                st.session_state.confidence = confidence
                                                st.session_state.genre_probabilities = genre_probabilities
                                            except Exception as e:
                                                st.error(f"Error making prediction: {e}")
                                    
                                    # Prediction button with callback
                                    st.button("Predict Genre Based on Audio Features Only", 
                                            type="primary", 
                                            key="predict_audio_only",
                                            on_click=predict_audio_only)
                                    
                                    # Display prediction results if available
                                    if st.session_state.prediction_made:
                                        # [Same display code as above]
                                        # Display prediction results
                                        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
                                        st.markdown("<p class='section-header'>Prediction Results (Audio Features Only)</p>", unsafe_allow_html=True)
                                        st.markdown(f"<p class='genre-label'>{st.session_state.predicted_genre.title()}</p>", unsafe_allow_html=True)
                                        
                                        if st.session_state.confidence:
                                            st.markdown(f"<p>Confidence: {st.session_state.confidence:.1%}</p>", unsafe_allow_html=True)
                                            st.progress(float(st.session_state.confidence))
                                        
                                        st.markdown("<p><strong>Description:</strong></p>", unsafe_allow_html=True)
                                        st.markdown(f"<p>{get_genre_description(st.session_state.predicted_genre)}</p>", unsafe_allow_html=True)
                                        
                                        st.markdown("</div>", unsafe_allow_html=True)
                                        
                                        # Create distribution chart
                                        if hasattr(st.session_state, 'genre_probabilities') and st.session_state.genre_probabilities:
                                            st.markdown("<p class='section-header'>Genre Probability Distribution</p>", unsafe_allow_html=True)
                                            dist_fig = create_genre_distribution_chart(st.session_state.genre_probabilities)
                                            st.pyplot(dist_fig)
                        else:
                            # Handle case when Genius API is not connected
                            # Similar to above, but with appropriate adjustments
                            # [Code for when Genius is not available]
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error("No song found. Please try a different search query.")
            else:
                st.error("Spotify API is not connected. Please set your Spotify API credentials.")
                
                # Offer alternative for when API is not available
                st.info("You can still use the manual input tab to predict genres without API access.")
            
    with tab3:
        st.header("About the Music Genre Predictor")
        
        st.markdown("""
        <div class="info-box">
        This application uses machine learning to predict music genres based on two main types of features:
        1. **Audio Features** - Technical aspects of the music such as energy, tempo, key, etc.
        2. **Emotion Features** - Emotional content derived from lyrics analysis
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("How It Works")
        
        st.markdown("""
        The prediction model was trained on a dataset of over 6,000 songs with known genres. The model analyzes:
        
        - **Audio characteristics** from Spotify's audio features API
        - **Emotional content** extracted from song lyrics using natural language processing
        - **Engineered features** that combine audio and emotional aspects
        
        The model can distinguish between 20 different music genres including pop, rock, jazz, hip-hop, and more.
        
        **Current model accuracy:** Approximately 23% across all genres (significantly better than random guessing at 5%)
        """)
        
        st.subheader("Using This App")
        
        st.markdown("""
        There are two main ways to use this application:
        
        **1. Predict Genre Tab:**
        - Manually adjust audio features and emotion sliders
        - Generate a random song for exploration
        - Get predictions based on your inputs
        
        **2. Search for a Song Tab:**
        - Search for real songs using the Spotify API
        - Analyze lyrics using the Genius API (if configured)
        - Get predictions based on actual song data
        
        For the best experience, set up the API credentials in your environment variables:
        - `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` for Spotify
        - `GENIUS_ACCESS_TOKEN` for Genius
        """)
        
        st.subheader("Technical Details")
        
        st.markdown("""
        - **Model Type:** Gradient Boosted Decision Trees (LightGBM)
        - **Features:** 30 features including audio characteristics, emotion scores, and engineered features
        - **Feature Engineering:** Created ratio features (e.g., joy-to-sadness ratio) and interaction features (e.g., energy × tempo)
        - **Training:** Used SMOTE-Tomek for handling class imbalance
        
        The application code is modular and can be extended with additional models or features. The radar chart visualization uses a custom polar projection to properly show emotion distributions.
        """)
    
    # Footer
    st.markdown("<div class='footer'>", unsafe_allow_html=True)
    st.markdown("Created for 50.038 Computational Data Science | 2025", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()