"""
visualize_similarity.py
-----------------------
Step 4: Visualize Movie Similarities using Heatmaps and PCA Clusters
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "movies.csv")
movies = pd.read_csv(csv_path)

def create_tags(r):
    return f"{r.get('genre','')} {r.get('genre','')} {r.get('language','')} {r.get('director','')} {r.get('tropes','')}"

movies["tags"] = movies.apply(create_tags, axis=1)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_m = tfidf.fit_transform(movies["tags"])
sim_mat = cosine_similarity(tfidf_m)

# Short titles for readability
short_titles = [t[:14] for t in movies["title"]]

# ─────────────────────────────────────────────
# FIGURE 1: Similarity Heatmap
# ─────────────────────────────────────────────
plt.figure(figsize=(18, 14))
sns.heatmap(
    sim_mat,
    xticklabels=short_titles,
    yticklabels=short_titles,
    cmap="YlOrRd",
    linewidths=0.2,
    annot=False
)
plt.title("Movie Similarity Heatmap (Brighter = More Similar)", fontsize=14)
plt.xticks(rotation=90, fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
out_heatmap = os.path.join(base_dir, "similarity_heatmap.png")
plt.savefig(out_heatmap, dpi=150)
plt.close()
print(f"Saved: {out_heatmap}")

# ─────────────────────────────────────────────
# FIGURE 2: 2D PCA Clusters
# ─────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(tfidf_m.toarray())

movies["pca_x"] = pca_coords[:, 0]
movies["pca_y"] = pca_coords[:, 1]
movies["primary_genre"] = movies["genre"].apply(lambda g: str(g).split(",")[0].strip())

plt.figure(figsize=(14, 9))
sns.scatterplot(
    data=movies,
    x="pca_x",
    y="pca_y",
    hue="primary_genre",
    s=120,
    alpha=0.85
)

# Label first 20 movies
for i in range(min(25, len(movies))):
    plt.annotate(
        movies.iloc[i]["title"][:14],
        (movies.iloc[i]["pca_x"] + 0.01, movies.iloc[i]["pca_y"] + 0.01),
        fontsize=7,
        alpha=0.8
    )

plt.title("Movie Clusters in 2D Feature Space (PCA Decomposition)", fontsize=13)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.25)
plt.tight_layout()
out_pca = os.path.join(base_dir, "pca_clusters.png")
plt.savefig(out_pca, dpi=150)
plt.close()
print(f"Saved: {out_pca}")

# ─────────────────────────────────────────────
# FIGURE 3: Top 5 Similar Movies for a Pick
# ─────────────────────────────────────────────
TARGET = "Inception"
if TARGET in movies["title"].values:
    idx = movies[movies["title"] == TARGET].index[0]
    scores = list(enumerate(sim_mat[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    similar_movies = [movies.iloc[i]["title"] for i, _ in scores]
    similar_scores = [round(s, 3) for _, s in scores]

    plt.figure(figsize=(8, 4.5))
    bars = plt.barh(similar_movies, similar_scores, color="steelblue", edgecolor="black")
    plt.xlabel("Cosine Similarity Score")
    plt.title(f"Movies Most Similar to '{TARGET}'")
    plt.xlim(0, 1.0)
    for bar, score in zip(bars, similar_scores):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 str(score), va="center", fontsize=9)
    plt.tight_layout()
    out_top = os.path.join(base_dir, "top_similar.png")
    plt.savefig(out_top, dpi=150)
    plt.close()
    print(f"Saved: {out_top}")
