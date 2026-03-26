import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.title("🎵 Music Recommendation System (KNN Based)")

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("SpotifyFeatures.csv")

features = [
    'popularity','danceability','energy','acousticness',
    'instrumentalness','liveness','loudness',
    'speechiness','tempo','valence'
]

music_data = df[features].dropna().sample(8000, random_state=42)

df = df.loc[music_data.index].reset_index(drop=True)
music_data = music_data.reset_index(drop=True)

# -----------------------
# Normalize Features
# -----------------------
scaler = StandardScaler()
scaled_features = scaler.fit_transform(music_data)

# -----------------------
# Train KNN Model
# -----------------------
knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(scaled_features)

# -----------------------
# Recommendation Function
# -----------------------
def recommend_song(song_name):

    song_name = song_name.strip().lower()
    df['track_lower'] = df['track_name'].str.lower()

    if song_name not in df['track_lower'].values:
        return ["Song not found in dataset"]

    index = df[df['track_lower'] == song_name].index[0]

    distances, indices = knn.kneighbors([scaled_features[index]])

    recommendations = []

    for i in indices[0][1:]:
        recommendations.append(df.iloc[i]['track_name'])

    return recommendations
st.subheader("Sample Songs You Can Try")
st.write(df['track_name'].sample(20))



# -----------------------
# User Input
# -----------------------
song = st.text_input("Enter Song Name")

if st.button("Recommend Songs"):
    results = recommend_song(song)

    st.write("### Recommended Songs:")
    for r in results:
        st.write(r)
