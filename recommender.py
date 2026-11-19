"""
recommender.py
--------------
Step 2: Build an Advanced Movie Recommendation System
        Uses Content-Based Filtering with Multi-Attribute TF-IDF Vectorization
        (Genre + Director + Narrative Tropes + Overview + Cast)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), "movies.csv")
movies = pd.read_csv(csv_path)

print("=" * 60)
print("🎬  CINEMATCH AI RECOMMENDATION ENGINE (EXPANDED 70+ DB)  🎬")
print("=" * 60)

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING (MULTI-ATTRIBUTE SOUP)
# ─────────────────────────────────────────────
def create_content_soup(row):
    parts = [
        str(row.get("genre", "")),
        str(row.get("genre", "")),
        str(row.get("language", "")),
        str(row.get("director", "")),
        str(row.get("tropes", "")),
        str(row.get("tropes", "")),
        str(row.get("stars", "")),
        str(row.get("overview", ""))
    ]
    return " ".join(parts)

movies["tags"] = movies.apply(create_content_soup, axis=1)

print("\n📌 Sample content soup created:")
for _, r in movies[["title", "genre", "director"]].head(5).iterrows():
    print(f"  • {r['title']} | {r['genre']} | Dir: {r['director']}")

# ─────────────────────────────────────────────
# 3. TF-IDF VECTORIZATION
# ─────────────────────────────────────────────
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["tags"])

print(f"\n🔢 TF-IDF Matrix Shape: {tfidf_matrix.shape}")
print(f"   ({tfidf_matrix.shape[0]} movies × {tfidf_matrix.shape[1]} unique vocabulary features)")

# ─────────────────────────────────────────────
# 4. COSINE SIMILARITY
# ─────────────────────────────────────────────
similarity_matrix = cosine_similarity(tfidf_matrix)
print(f"📐 Similarity Matrix: {similarity_matrix.shape[0]}×{similarity_matrix.shape[1]} symmetric distance table")

# ─────────────────────────────────────────────
# 5. RECOMMENDATION FUNCTION
# ─────────────────────────────────────────────
def recommend_movies(movie_title, top_n=5, dealbreaker=None):
    """
    Given a movie title, returns the top N similar movies.
    Supports negative constraint dealbreaker filtering.
    """
    movie_title = movie_title.strip()
    matches = movies[movies["title"].str.lower() == movie_title.lower()]
    
    if matches.empty:
        print(f"\n❌ Movie '{movie_title}' not found!")
        return []
    
    movie_idx = matches.index[0]
    similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    results = []
    for idx, score in similarity_scores[1:]:
        cand = movies.iloc[idx]
        if dealbreaker:
            dbs = str(cand.get("dealbreakers", "")).lower()
            if dealbreaker.lower() in dbs:
                continue
        results.append({
            "title": cand["title"],
            "genre": cand["genre"],
            "director": cand.get("director", "N/A"),
            "rating": cand["rating"],
            "similarity": round(score * 100, 1),
            "ott_india": cand.get("ott_india", "N/A")
        })
        if len(results) >= top_n:
            break
            
    print(f"\n🎯 Top recommendations for '{movie_title}':")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['title']} ({r['rating']}★) — {r['similarity']}% Match | {r['genre']} | Stream: {r['ott_india']}")
    return results

if __name__ == "__main__":
    recommend_movies("Inception", top_n=5)
    recommend_movies("Tumbbad", top_n=5)
    recommend_movies("3 Idiots", top_n=5)
