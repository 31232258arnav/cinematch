"""
data_generator.py
-----------------
Generates user interaction ratings matrix for collaborative filtering
without overwriting the enriched movies.csv catalog.
"""

import os
import pandas as pd
import numpy as np
import random

base_dir = os.path.dirname(__file__)
movies_path = os.path.join(base_dir, "movies.csv")
ratings_path = os.path.join(base_dir, "user_ratings.csv")

movies_df = pd.read_csv(movies_path)
movie_ids = movies_df["movie_id"].tolist()

np.random.seed(42)
random.seed(42)

# Generate 25 distinct user profiles
users = [f"User_{i}" for i in range(1, 26)]
ratings_rows = []

for user in users:
    # Each user rates 20 to 40 films
    k_rated = random.randint(20, min(40, len(movie_ids)))
    rated_mids = random.sample(movie_ids, k=k_rated)
    for mid in rated_mids:
        base_rating = float(movies_df[movies_df["movie_id"] == mid]["rating"].iloc[0])
        # Add normal noise
        ur = round(min(10.0, max(2.0, base_rating + np.random.normal(0, 0.9))), 1)
        ratings_rows.append({"user": user, "movie_id": mid, "user_rating": ur})

ratings_df = pd.DataFrame(ratings_rows)
ratings_df.to_csv(ratings_path, index=False)
print(f"[OK] Generated {len(ratings_df)} user ratings across {len(users)} users for {len(movies_df)} movies.")
