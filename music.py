import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv("SpotifyFeatures.csv")

print("Dataset Loaded Successfully")

# -------------------------------
# Step 2: Select Features
# -------------------------------
features = [
    'popularity','danceability','energy','acousticness',
    'instrumentalness','liveness','loudness',
    'speechiness','tempo','valence'
]

# -------------------------------
# Step 3: Use 8000 Songs Sample
# -------------------------------
music_data = df[features].dropna().sample(8000, random_state=42)

# Sync original dataframe with sampled data
df = df.loc[music_data.index].reset_index(drop=True)
music_data = music_data.reset_index(drop=True)

print("Total Songs Used:", len(df))

# -------------------------------
# Step 4: Normalize Features
# -------------------------------
scaler = StandardScaler()
scaled_features = scaler.fit_transform(music_data)

# -------------------------------
# Step 5: Create Similarity Matrix
# -------------------------------
similarity_matrix = cosine_similarity(scaled_features)

# -------------------------------
# Step 6: Recommendation Function
# -------------------------------
def recommend_song(song_name, num_recommendations=5):

    song_name = song_name.strip().lower()

    # Create lowercase comparison column
    df['track_lower'] = df['track_name'].str.lower()

    if song_name not in df['track_lower'].values:
        return "Song not found in dataset"

    index = df[df['track_lower'] == song_name].index[0]

    similarity_scores = list(enumerate(similarity_matrix[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommendations = []

    for i in similarity_scores[1:num_recommendations+1]:
        recommendations.append(df.iloc[i[0]]['track_name'])

    return recommendations


# -------------------------------
# Step 7: Show Sample Songs
# -------------------------------
print("\nSample Songs You Can Try:")
print(df['track_name'].head(20))

# -------------------------------
# Step 8: User Input
# -------------------------------
song = input("\nEnter song name from above list: ").strip()


print("\nRecommended Songs:")
print(recommend_song(song))
