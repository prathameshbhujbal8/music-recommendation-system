# music-recommendation-system
A machine learning-based music recommendation system that suggests similar songs based on audio features using content-based filtering and cosine similarity. The system is deployed using Streamlit for an interactive user experience.

🚀 Features
🎧 Content-based music recommendation
📊 Uses audio features like energy, tempo, danceability, etc.
⚖️ Feature weighting for better personalization
🔍 Cosine similarity for finding similar songs
🧹 Data preprocessing and normalization (StandardScaler)
🎯 Top-K recommendation (Top 5 songs)
🖥️ Interactive UI using Streamlit
🔡 Case-insensitive song search
📈 Displays similarity score and song details
🧠 Technologies Used
Python
Pandas
Scikit-learn
Streamlit
NumPy
📂 Dataset
Dataset used: SpotifyFeatures.csv
Contains audio features such as:
popularity
danceability
energy
acousticness
instrumentalness
liveness
loudness
speechiness
tempo
valence

👉 For performance optimization, a sample of 8000 songs is used.

⚙️ How It Works
Data Loading
Load Spotify dataset using Pandas.
Feature Selection
Select important audio features.
Data Preprocessing
Remove missing values
Normalize features using StandardScaler
Feature Weighting
Assign importance to features for better personalization.
Similarity Calculation
Use cosine similarity to compute similarity between songs.
Recommendation
Find input song
Compute similarity scores
Return top 5 similar songs
🖥️ Running the Project
Step 1: Install Dependencies
pip install pandas scikit-learn streamlit numpy
Step 2: Run Streamlit App
streamlit run app.py
Step 3: Open in Browser
http://localhost:8501
📸 Output
Enter or select a song
Click "Recommend Songs"
View top 5 similar songs with details
📊 Example

Input:
Believer

Output:

Song A
Song B
Song C
Song D
Song E

(Based on similarity scores)

📈 Future Enhancements
Hybrid recommendation (content + collaborative filtering)
Deep learning models (CNN, RNN)
Context-aware recommendations (mood, location)
Real-time user preference learning
Deployment on cloud (AWS / Streamlit Cloud)
🎯 Applications
Music streaming platforms (Spotify, Apple Music)
Personalized recommendation systems
AI-based entertainment systems
👨‍💻 Author

Prathamesh Bhujbal
B.Tech CSE Student

⭐ Key Learning Outcomes
Machine Learning fundamentals
Feature engineering and normalization
Similarity-based recommendation systems
Streamlit deployment
Real-world project development

🌐 Deploy this project online

Just say: “next step”
