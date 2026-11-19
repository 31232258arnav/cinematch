"""
collaborative_filter.py
-----------------------
Step 3: Collaborative Filtering — "Users Like You Also Liked..."
        Recommends films based on user rating correlations.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

base_dir = os.path.dirname(__file__)
movies_path = os.path.join(base_dir, "movies.csv")
ratings_path = os.path.join(base_dir, "user_ratings.csv")

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# Merge to get movie titles in ratings
ratings = ratings.merge(movies[["movie_id", "title"]], on="movie_id")

print("=" * 60)
print("👥  COLLABORATIVE FILTERING RECOMMENDER (USER-ITEM PIVOT)  👥")
print("=" * 60)

# ─────────────────────────────────────────────
# 2. CREATE PIVOT TABLE
# ─────────────────────────────────────────────
user_movie_matrix = ratings.pivot_table(
    index="user",
    columns="title",
    values="user_rating"
).fillna(0)

print(f"\n📊 Pivot Matrix: {user_movie_matrix.shape[0]} Users × {user_movie_matrix.shape[1]} Titles")

# ─────────────────────────────────────────────
# 3. COMPUTE USER SIMILARITY
# ─────────────────────────────────────────────
user_sim = cosine_similarity(user_movie_matrix)
user_sim_df = pd.DataFrame(
    user_sim,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

# ─────────────────────────────────────────────
# 4. RECOMMENDATION FUNCTION
# ─────────────────────────────────────────────
def recommend_for_user(target_user, top_n=5):
    if target_user not in user_sim_df.index:
        print(f"❌ User '{target_user}' not found!")
        return []

    # Get most similar users
    similar_users = user_sim_df[target_user].sort_values(ascending=False).drop(target_user)
    top_user = similar_users.index[0]
    similarity_val = round(similar_users.iloc[0] * 100, 1)

    print(f"\n👤 Target: '{target_user}'")
    print(f"🤝 Kindred User: '{top_user}' (Taste Correlation: {similarity_val}%)")

    # Films watched by top_user but unrated by target_user
    target_watched = set(ratings[ratings["user"] == target_user]["title"])
    top_user_ratings = ratings[(ratings["user"] == top_user) & (ratings["user_rating"] >= 7.5)]

    candidates = top_user_ratings[~top_user_ratings["title"].isin(target_watched)]
    candidates = candidates.sort_values(by="user_rating", ascending=False).head(top_n)

    print(f"\n⭐ Collaborative Recommendations for {target_user}:")
    for i, (_, row) in enumerate(candidates.iterrows(), 1):
        print(f"  {i}. {row['title']} (Rated {row['user_rating']}★ by {top_user})")
    return candidates["title"].tolist()

if __name__ == "__main__":
    recommend_for_user("User_1", top_n=5)
    recommend_for_user("User_3", top_n=5)
