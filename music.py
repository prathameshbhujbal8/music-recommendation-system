import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ── Step 1: Load Dataset ──────────────────────────────────────────────────────
df = pd.read_csv("SpotifyFeatures.csv")
print("Dataset loaded successfully.")
print(f"Total rows in dataset: {len(df)}")

# ── Step 2: Select Features ───────────────────────────────────────────────────
features = ['popularity', 'danceability', 'energy', 'acousticness',
            'instrumentalness', 'liveness', 'loudness',
            'speechiness', 'tempo', 'valence']

# ── Step 3: Sample 8000 Songs ─────────────────────────────────────────────────
music_data = df[features].dropna().sample(8000, random_state=42)
df = df.loc[music_data.index].reset_index(drop=True)
music_data = music_data.reset_index(drop=True)
print(f"Songs used for recommendation: {len(df)}")

# ── Step 4: Normalise Features ────────────────────────────────────────────────
scaler = StandardScaler()
scaled_features = scaler.fit_transform(music_data)
print("Features normalised using StandardScaler.")

# ── Step 5: Build Cosine Similarity Matrix ────────────────────────────────────
print("Building cosine similarity matrix... (this may take a moment)")
similarity_matrix = cosine_similarity(scaled_features)
print(f"Matrix shape: {similarity_matrix.shape}")

# ── Step 6: Recommendation Function ──────────────────────────────────────────
def recommend_song(song_name, num_recommendations=5):
    song_name = song_name.strip().lower()
    df['track_lower'] = df['track_name'].str.lower()

    if song_name not in df['track_lower'].values:
        return None

    index = df[df['track_lower'] == song_name].index[0]

    scores = sorted(
        list(enumerate(similarity_matrix[index])),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for i in scores[1:num_recommendations + 1]:
        results.append({
            'song':   df.iloc[i[0]]['track_name'],
            'artist': df.iloc[i[0]]['artist_name'],
            'score':  round(i[1], 3)
        })
    return results

# ── Step 7: Show Sample Songs ─────────────────────────────────────────────────
print("\n" + "="*50)
print("SAMPLE SONGS YOU CAN TRY:")
print("="*50)
for i, name in enumerate(df['track_name'].head(20), 1):
    print(f"  {i:2}. {name}")

# ── Step 8: User Input & Output ───────────────────────────────────────────────
print("\n" + "="*50)
song = input("Enter song name from the list above: ").strip()

results = recommend_song(song)

print("\n" + "="*50)
if results is None:
    print(f"Song '{song}' not found in the dataset.")
    print("Make sure the name is spelled exactly as shown in the list.")
else:
    print(f"TOP 5 RECOMMENDATIONS FOR: '{song.upper()}'")
    print("="*50)
    for r in results:
        print(f"  Song   : {r['song']}")
        print(f"  Artist : {r['artist']}")
        print(f"  Score  : {r['score']}")
        print("-"*40)
