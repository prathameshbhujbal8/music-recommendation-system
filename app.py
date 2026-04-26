import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="centered")

st.markdown("""
<style>
  .main-header {
    background: #191414;
    border-left: 5px solid #1DB954;
    padding: 18px 24px;
    border-radius: 10px;
    margin-bottom: 24px;
  }
  .main-header h1 { color: #1DB954; font-size: 1.8rem; margin: 0; }
  .main-header p  { color: #aaa; font-size: 0.85rem; margin: 4px 0 0; }

  .query-card {
    background: #eafaf1;
    border: 1.5px solid #1DB954;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 18px;
  }
  .query-card h4 { color: #145a32; margin: 0 0 4px; font-size: 1rem; }
  .query-card p  { color: #1e8449; margin: 0; font-size: 0.85rem; }

  .rec-card {
    background: #f8f9fa;
    border-left: 4px solid #1DB954;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
  }
  .rec-card h4 { margin: 0 0 3px; color: #191414; font-size: 0.95rem; }
  .rec-card p  { margin: 2px 0; color: #555; font-size: 0.83rem; }

  .score-badge {
    display: inline-block;
    background: #1DB954;
    color: white;
    padding: 1px 9px;
    border-radius: 10px;
    font-weight: bold;
    font-size: 0.8rem;
  }

  .insight-box {
    background: #fffde7;
    border-left: 4px solid #f9a825;
    border-radius: 6px;
    padding: 10px 14px;
    margin-top: 14px;
    font-size: 0.85rem;
    color: #5d4037;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
  <h1>🎵 Music Recommendation System</h1>
  <p>Content-Based Filtering using KNN &amp; Cosine Similarity on Spotify Audio Features</p>
</div>""", unsafe_allow_html=True)

# ── Load & Preprocess Data ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("SpotifyFeatures.csv")

    features = ['popularity', 'danceability', 'energy', 'acousticness',
                'instrumentalness', 'liveness', 'loudness',
                'speechiness', 'tempo', 'valence']

    music_data = df[features].dropna().sample(8000, random_state=42)
    df = df.loc[music_data.index].reset_index(drop=True)
    music_data = music_data.reset_index(drop=True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(music_data)

    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(scaled)

    return df, scaled, knn, features

df, scaled_features, knn, features = load_data()

# ── Recommendation Function ───────────────────────────────────────────────────
def recommend_song(song_name):
    song_name = song_name.strip().lower()
    df['track_lower'] = df['track_name'].str.lower()

    if song_name not in df['track_lower'].values:
        return None, []

    idx = df[df['track_lower'] == song_name].index[0]
    distances, indices = knn.kneighbors([scaled_features[idx]])

    recs = []
    for i, (j, d) in enumerate(zip(indices[0][1:], distances[0][1:])):
        recs.append({
            'rank':   i + 1,
            'song':   df.iloc[j]['track_name'],
            'artist': df.iloc[j]['artist_name'],
            'score':  round(1 - d, 3),
            'idx':    j
        })
    return idx, recs

# ── Sample Songs Expander ─────────────────────────────────────────────────────
with st.expander("📋 Browse sample songs to find a valid input (click to expand)"):
    sample = (
        df[['track_name', 'artist_name']]
        .sample(25, random_state=5)
        .reset_index(drop=True)
    )
    sample.columns = ['Song', 'Artist']
    st.dataframe(sample, width='stretch')

# ── Input Row ─────────────────────────────────────────────────────────────────
st.markdown("---")
col_in, col_btn = st.columns([4, 1])
with col_in:
    song_input = st.text_input(
        "🔍 Enter a song name exactly as shown above",
        placeholder="e.g. Cold"
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    btn = st.button("Get Recommendations", use_container_width=False)

# ── Results ───────────────────────────────────────────────────────────────────
if btn:
    if not song_input.strip():
        st.warning("⚠️ Please enter a song name first.")

    else:
        query_idx, results = recommend_song(song_input)

        if not results:
            st.error(
                f"❌ '{song_input}' was not found in the dataset. "
                "Check the spelling or pick a song from the sample list above."
            )
        else:
            # ── Highlighted query card ────────────────────────────────────────
            st.markdown(f"""
            <div class='query-card'>
              <h4>🎧 You searched for: {song_input.title()}</h4>
              <p>Here are the 5 most acoustically similar songs
                 from our dataset of 8,000 tracks.</p>
            </div>""", unsafe_allow_html=True)

            # ── Two-column result cards ───────────────────────────────────────
            col1, col2 = st.columns(2)
            for i, r in enumerate(results):
                target_col = col1 if i % 2 == 0 else col2
                search_url = (
                    f"https://www.google.com/search?q="
                    f"{r['song'].replace(' ', '+')}+"
                    f"{r['artist'].replace(' ', '+')}+song"
                )
                target_col.markdown(f"""
                <div class='rec-card'>
                  <h4>🎵 {r['rank']}. {r['song']}</h4>
                  <p>🎙️ <b>Artist:</b> {r['artist']}</p>
                  <p>📊 <b>Match score:</b>
                     <span class='score-badge'>{r['score']}</span></p>
                  <p><a href='{search_url}' target='_blank'>
                     🔗 Search online</a></p>
                </div>""", unsafe_allow_html=True)

            # ── Insight message ───────────────────────────────────────────────
            top_score = results[0]['score']
            if top_score >= 0.95:
                insight = (
                    f"Very strong match! <b>{results[0]['song']}</b> shares "
                    "nearly identical audio characteristics with your song."
                )
            elif top_score >= 0.90:
                insight = (
                    f"Strong match. All recommendations share a similar mood, "
                    f"tempo, and energy level with <b>{song_input.title()}</b>."
                )
            else:
                insight = (
                    "Moderate match. The songs are acoustically similar "
                    "but may differ slightly in style or genre."
                )
            st.markdown(
                f"<div class='insight-box'>💡 {insight}</div>",
                unsafe_allow_html=True
            )

            # ── Chart 1: Audio Feature Comparison ────────────────────────────
            st.markdown("---")
            st.subheader("📊 Audio Feature Comparison")
            st.caption(
                "Comparing your input song with the top recommendation "
                "across all 10 audio features (normalised values)."
            )

            query_vals = scaled_features[query_idx]
            rec_vals   = scaled_features[results[0]['idx']]

            chart_df = pd.DataFrame({
                f"Your song  ({song_input.title()})": query_vals,
                f"Top match  ({results[0]['song']})": rec_vals
            }, index=features)

            st.bar_chart(chart_df, width='stretch', height=320)

            st.markdown(
                "<small>📌 <b>How to read this:</b> Each bar is the normalised value "
                "of one audio feature after StandardScaler. "
                "Features with similar bar heights drove this recommendation. "
                "Negative values mean the feature is below the dataset average — "
                "that is completely normal.</small>",
                unsafe_allow_html=True
            )

            # ── Chart 2: Similarity Scores ────────────────────────────────────
            st.markdown("---")
            st.subheader("📈 Similarity Scores — All 5 Recommendations")
            st.caption(
                "How similar each recommended song is to your input "
                "(cosine similarity score, max = 1.0)."
            )

            scores_df = pd.DataFrame({
                'Song': [f"{r['rank']}. {r['song']}" for r in results],
                'Similarity Score': [r['score'] for r in results]
            }).set_index('Song')

            st.bar_chart(scores_df, width='stretch', height=250)

            st.markdown(
                "<small>📌 Scores above 0.93 indicate a very strong acoustic match. "
                "All 5 songs share a similar musical profile with your input.</small>",
                unsafe_allow_html=True
            )
