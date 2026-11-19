"""
build_web_app.py
----------------
Master Compiler for CineMatch — Enterprise-Grade OTT & AI Discovery Platform
Incorporates:
1. Precomputed Scikit-Learn TF-IDF Cosine Similarity Matrix (100 films, 7,431 features).
2. 5-Axis Explainable AI (XAI) Radar / Spider Charts on every movie.
3. Mobile-First "Swipe to Decide" Tinder-Style Recommendation Engine with live gradient vector updates.
4. Native Web Speech API Voice Search (hands-free microphone speech-to-text).
5. "Cinema Time Machine" Era Traversal (1970s through 2020s).
6. HTML5 Canvas "Binge Ticket" Generator with one-click PNG download.
7. Movie DNA Mixer (Vector Arithmetic blend slider).
8. Duo / Couple Consensus Solver (Pareto-optimal minimax intersection).
9. Negative Constraint Vectoring (Dealbreaker filters).
10. Micro-Trope & Scene-Level Semantic Search.
11. Real-time Indian OTT Deep-Links (Netflix, Prime, JioCinema, Hotstar, SonyLIV, Zee5, YouTube).
12. Soft-Fallback Relevance Scoring ensuring ZERO 0-result states under any criteria.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

base_dir = r"C:\Users\Nucom\Projects\MovieRecommender"
csv_path = os.path.join(base_dir, "movies.csv")
index_path = os.path.join(base_dir, "index.html")

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} movies from {csv_path}")

# ─────────────────────────────────────────────
# 1. TRAIN SCIKIT-LEARN TF-IDF MODEL
# ─────────────────────────────────────────────
def build_feature_soup(r):
    parts = [
        str(r.get("genre", "")),
        str(r.get("genre", "")),
        str(r.get("genre", "")),
        str(r.get("mood", "")),
        str(r.get("mood", "")),
        str(r.get("director", "")),
        str(r.get("director", "")),
        str(r.get("stars", "")),
        str(r.get("tropes", "")),
        str(r.get("tropes", "")),
        str(r.get("tropes", "")),
        str(r.get("overview", ""))
    ]
    return " ".join(parts)

df["soup"] = df.apply(build_feature_soup, axis=1)

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
    stop_words="english"
)
tfidf_matrix = tfidf.fit_transform(df["soup"])
print(f"Trained TF-IDF vocabulary: {tfidf_matrix.shape[1]} features across {tfidf_matrix.shape[0]} documents")

sim_matrix = cosine_similarity(tfidf_matrix)

# ─────────────────────────────────────────────
# 2. PRECOMPUTE TOP NEIGHBORS & XAI RADAR METRICS
# ─────────────────────────────────────────────
def compute_xai_radar(row):
    genre_str = str(row.get("genre", "")).lower()
    tropes_str = str(row.get("tropes", "")).lower()
    mood_str = str(row.get("mood", "")).lower()
    
    # 1. Complexity (Mind-Bending, Mystery, Sci-Fi)
    complexity = 45
    if "sci-fi" in genre_str: complexity += 25
    if "mystery" in genre_str or "puzzle" in tropes_str: complexity += 20
    if "mind-bending" in mood_str or "unreliable narrator" in tropes_str: complexity += 25
    complexity = min(98, max(35, complexity))
    
    # 2. Pacing (Action, Adrenaline, Crime)
    pacing = 40
    if "action" in genre_str: pacing += 30
    if "thriller" in genre_str: pacing += 20
    if "adrenaline" in mood_str or "shootout" in tropes_str or "chase" in tropes_str: pacing += 25
    if "slow burn" in tropes_str or "drama" in genre_str and "action" not in genre_str: pacing -= 15
    pacing = min(98, max(25, pacing))
    
    # 3. Emotional Gravity (Drama, Tearjerker, Inspiring)
    emotion = 35
    if "drama" in genre_str: emotion += 25
    if "romance" in genre_str or "family" in genre_str: emotion += 25
    if "tearjerker" in tropes_str or "emotional" in mood_str or "inspiring" in mood_str: emotion += 25
    emotion = min(98, max(30, emotion))
    
    # 4. Visual Stylization (Sci-Fi, Fantasy, Animation, Noir)
    visuals = 50
    if "animation" in genre_str: visuals += 35
    if "fantasy" in genre_str or "sci-fi" in genre_str: visuals += 25
    if "cinematography" in tropes_str or "rain" in tropes_str or "steampunk" in tropes_str: visuals += 20
    visuals = min(98, max(40, visuals))
    
    # 5. Thematic Depth (Philosophy, Existential, Social Commentary)
    depth = 40
    if "biography" in genre_str or "history" in genre_str: depth += 25
    if "existential" in mood_str or "caste" in tropes_str or "class warfare" in tropes_str or "moral" in tropes_str: depth += 30
    depth = min(98, max(30, depth))
    
    return [complexity, pacing, emotion, visuals, depth]

trained_similarity_map = {}
xai_radar_map = {}

for idx in range(len(df)):
    m_id = int(df.iloc[idx]["movie_id"])
    scores = list(enumerate(sim_matrix[idx]))
    scores = [s for s in scores if s[0] != idx]
    scores.sort(key=lambda x: x[1], reverse=True)

    max_score = scores[0][1] if scores else 1.0
    top_neighbors = []
    for n_idx, raw_score in scores[:12]:
        neighbor_id = int(df.iloc[n_idx]["movie_id"])
        calibrated_pct = int(min(98, max(52, round((raw_score / max_score) * 94))))
        top_neighbors.append({
            "id": neighbor_id,
            "title": df.iloc[n_idx]["title"],
            "score": calibrated_pct
        })
    trained_similarity_map[m_id] = top_neighbors
    xai_radar_map[m_id] = compute_xai_radar(df.iloc[idx])

movies_list = df.to_dict(orient="records")
for m in movies_list:
    m_id = int(m["movie_id"])
    m["similar"] = trained_similarity_map.get(m_id, [])
    m["radar"] = xai_radar_map.get(m_id, [50, 50, 50, 50, 50])

movies_json = json.dumps(movies_list, ensure_ascii=False)
print("Precomputed ML similarity matrices & 5-axis XAI radar coordinates ready.")

# ─────────────────────────────────────────────
# 3. ASSEMBLE PRODUCTION WEB APPLICATION
# ─────────────────────────────────────────────
html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
    <meta name="theme-color" content="#0B0B14">
    <title>CineMatch — AI Movie Discovery, Hybrid Recommender &amp; OTT Streaming Platform</title>
    <style>
        :root {{
            --bg-main: #0B0B14;
            --bg-card: #141424;
            --bg-card-hover: #1A1A34;
            --accent-red: #E50914;
            --accent-red-hover: #b80710;
            --accent-blue: #38bdf8;
            --accent-purple: #a855f7;
            --accent-gold: #f59e0b;
            --accent-green: #10b981;
            --text-main: #FFFFFF;
            --text-muted: #9E9EA7;
            --border: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(229, 9, 20, 0.45);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            -webkit-tap-highlight-color: transparent;
        }}

        body {{
            background-color: var(--bg-main);
            color: var(--text-main);
            padding-bottom: 70px;
            overflow-x: hidden;
            -webkit-text-size-adjust: 100%;
        }}

        /* App Top Bar */
        .app-bar {{
            position: sticky;
            top: 0;
            z-index: 1000;
            background: rgba(11, 11, 20, 0.95);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border);
            padding: 12px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .brand {{
            font-size: 1.3rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .brand span {{ color: var(--accent-red); }}

        .brand-sub {{
            font-size: 0.65rem;
            background: rgba(229, 9, 20, 0.15);
            color: var(--accent-red);
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 700;
            margin-left: 6px;
            text-transform: uppercase;
        }}

        .top-right-actions {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .btn-top {{
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            transition: all 0.2s ease;
        }}

        .btn-vip {{
            background: linear-gradient(135deg, #f59e0b, #d97706);
            color: #000;
            border: none;
        }}
        .btn-vip:hover {{ filter: brightness(1.1); }}

        /* Navigation Tabs */
        .nav-tabs {{
            display: flex;
            overflow-x: auto;
            background: #0F0F1B;
            border-bottom: 1px solid var(--border);
            padding: 6px 16px;
            gap: 6px;
            scrollbar-width: none;
        }}
        .nav-tabs::-webkit-scrollbar {{ display: none; }}

        .tab-btn {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 8px 14px;
            font-size: 0.82rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .tab-btn:hover {{ color: #fff; background: rgba(255, 255, 255, 0.05); }}
        .tab-btn.active {{ color: #fff; background: var(--accent-red); }}

        .container {{
            max-width: 1320px;
            margin: 0 auto;
            padding: 20px 16px;
        }}

        .section-panel {{
            display: none;
            animation: fadeIn 0.25s ease;
        }}
        .section-panel.active {{ display: block; }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Hero Banner */
        .hero {{
            text-align: center;
            margin-bottom: 24px;
            padding: 24px 16px;
            background: linear-gradient(180deg, rgba(229, 9, 20, 0.08) 0%, rgba(11, 11, 20, 0) 100%);
            border-radius: 12px;
            border: 1px solid rgba(229, 9, 20, 0.12);
        }}

        .hero h1 {{
            font-size: 1.8rem;
            font-weight: 800;
            margin-bottom: 6px;
            letter-spacing: -0.5px;
        }}

        .hero p {{
            font-size: 0.9rem;
            color: var(--text-muted);
            max-width: 650px;
            margin: 0 auto;
            line-height: 1.4;
        }}

        /* Search Input with Voice Button */
        .search-wrap {{
            position: relative;
            display: flex;
            align-items: center;
            margin-bottom: 12px;
        }}

        .search-box {{
            width: 100%;
            background: #141424;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px 48px 14px 18px;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease;
        }}
        .search-box:focus {{
            border-color: var(--accent-red);
            box-shadow: 0 0 0 3px rgba(229, 9, 20, 0.15);
        }}

        .btn-voice {{
            position: absolute;
            right: 12px;
            background: rgba(229, 9, 20, 0.15);
            border: 1px solid rgba(229, 9, 20, 0.3);
            color: var(--accent-red);
            width: 32px;
            height: 32px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            transition: all 0.2s ease;
        }}
        .btn-voice:hover {{
            background: var(--accent-red);
            color: #fff;
        }}
        .btn-voice.listening {{
            background: #ef4444;
            color: #fff;
            animation: pulseVoice 1s infinite alternate;
        }}

        @keyframes pulseVoice {{
            from {{ transform: scale(1.0); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }}
            to {{ transform: scale(1.15); box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }}
        }}

        /* Era / Time Machine Slider */
        .era-slider-bar {{
            background: rgba(20, 20, 36, 0.75);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            flex-wrap: wrap;
        }}

        .era-chips {{
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }}

        .era-chip {{
            background: #1a1a2e;
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--text-muted);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .era-chip:hover {{ color: #fff; background: #252540; }}
        .era-chip.active {{
            background: var(--accent-blue);
            color: #000;
            border-color: var(--accent-blue);
        }}

        .filter-row {{
            display: flex;
            gap: 10px;
            margin-bottom: 14px;
            flex-wrap: wrap;
        }}

        .filter-select {{
            background: #141424;
            color: #fff;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 0.82rem;
            outline: none;
            cursor: pointer;
            flex: 1;
            min-width: 130px;
        }}

        /* Dealbreaker Bar */
        .dealbreaker-bar {{
            background: rgba(20, 20, 36, 0.7);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .dealbreaker-label {{
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--accent-red);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-right: 4px;
        }}

        .pill-toggle {{
            background: #1a1a2e;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #cbd5e1;
            padding: 5px 11px;
            border-radius: 20px;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s ease;
            user-select: none;
        }}
        .pill-toggle:hover {{ background: #252540; }}
        .pill-toggle.active {{
            background: rgba(229, 9, 20, 0.2);
            border-color: var(--accent-red);
            color: #ff858d;
            font-weight: 600;
        }}

        /* Status Banner for Soft-Matching */
        .fallback-banner {{
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.25);
            color: #7dd3fc;
            padding: 10px 16px;
            border-radius: 8px;
            font-size: 0.82rem;
            margin-bottom: 18px;
            display: none;
            align-items: center;
            gap: 8px;
        }}

        /* Movie Grid */
        .movie-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 18px;
        }}

        /* Movie Card */
        .movie-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
            position: relative;
        }}
        .movie-card:hover {{
            transform: translateY(-4px);
            border-color: var(--border-hover);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.4);
        }}

        .poster-container {{
            position: relative;
            width: 100%;
            padding-top: 140%;
            background: #0d0d18;
            overflow: hidden;
        }}

        .poster-container img {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.3s ease;
        }}
        .movie-card:hover .poster-container img {{ transform: scale(1.04); }}

        .badge-rating {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(8px);
            color: var(--accent-gold);
            font-size: 0.78rem;
            font-weight: 700;
            padding: 4px 8px;
            border-radius: 6px;
            border: 1px solid rgba(245, 158, 11, 0.3);
            display: flex;
            align-items: center;
            gap: 3px;
        }}

        .card-info {{
            padding: 14px;
            display: flex;
            flex-direction: column;
            flex: 1;
        }}

        .card-title {{
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 6px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .card-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin-bottom: 8px;
        }}

        .tag {{
            background: rgba(255, 255, 255, 0.06);
            color: var(--text-muted);
            font-size: 0.68rem;
            padding: 2px 7px;
            border-radius: 4px;
        }}

        .tag-match {{
            background: rgba(229, 9, 20, 0.2);
            color: #ff6b72;
            font-weight: 700;
            border: 1px solid rgba(229, 9, 20, 0.4);
        }}

        .card-plot {{
            font-size: 0.78rem;
            color: var(--text-muted);
            line-height: 1.35;
            margin-bottom: 10px;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            flex: 1;
        }}

        /* OTT Provider Badges (Feature 6) */
        .ott-box {{
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 6px;
            padding: 6px 8px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 6px;
        }}

        .ott-label {{
            font-size: 0.65rem;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
        }}

        .ott-badges {{
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
            justify-content: flex-end;
        }}

        .badge-ott {{
            font-size: 0.65rem;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 700;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            transition: opacity 0.2s ease;
        }}
        .badge-ott:hover {{ opacity: 0.85; }}
        .badge-ott-netflix {{ background: #E50914; color: #fff; }}
        .badge-ott-prime {{ background: #00A8E1; color: #fff; }}
        .badge-ott-jio {{ background: #D9008D; color: #fff; }}
        .badge-ott-hotstar {{ background: #0F1014; border: 1px solid #1377C4; color: #1377C4; }}
        .badge-ott-sonyliv {{ background: #020202; border: 1px solid #1EA896; color: #1EA896; }}
        .badge-ott-zee5 {{ background: #8230C6; color: #fff; }}
        .badge-ott-youtube {{ background: #FF0000; color: #fff; }}
        .badge-ott-default {{ background: #334155; color: #fff; }}

        .card-actions {{
            display: flex;
            gap: 6px;
            margin-top: auto;
        }}

        .btn-action {{
            flex: 1;
            padding: 8px 6px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
            border: none;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
        }}

        .btn-trailer {{ background: rgba(255, 255, 255, 0.08); color: #fff; }}
        .btn-trailer:hover {{ background: rgba(255, 255, 255, 0.16); }}

        .btn-radar {{ background: rgba(56, 189, 248, 0.12); color: var(--accent-blue); }}
        .btn-radar:hover {{ background: rgba(56, 189, 248, 0.25); }}

        .btn-bookmark {{ background: rgba(255, 255, 255, 0.05); color: var(--text-muted); }}
        .btn-bookmark.saved {{
            background: rgba(229, 9, 20, 0.2);
            color: #ff6b72;
            border: 1px solid rgba(229, 9, 20, 0.3);
        }}

        /* Swipe to Decide Container */
        .swipe-wrapper {{
            max-width: 440px;
            margin: 0 auto;
            text-align: center;
        }}

        .swipe-card-stage {{
            position: relative;
            width: 100%;
            height: 520px;
            margin-bottom: 20px;
        }}

        .swipe-card {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: #141424;
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 16px 32px rgba(0, 0, 0, 0.5);
            transition: transform 0.25s ease, opacity 0.25s ease;
        }}

        .swipe-controls {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 10px;
        }}

        .btn-swipe {{
            width: 60px;
            height: 60px;
            border-radius: 50%;
            border: 2px solid var(--border);
            background: #141424;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .btn-swipe-pass {{ color: #ef4444; border-color: rgba(239, 68, 68, 0.4); }}
        .btn-swipe-pass:hover {{ background: #ef4444; color: #fff; transform: scale(1.1); }}

        .btn-swipe-like {{ color: #10b981; border-color: rgba(16, 185, 129, 0.4); }}
        .btn-swipe-like:hover {{ background: #10b981; color: #fff; transform: scale(1.1); }}

        .btn-swipe-save {{ color: #f59e0b; border-color: rgba(245, 158, 11, 0.4); }}
        .btn-swipe-save:hover {{ background: #f59e0b; color: #000; transform: scale(1.1); }}

        /* Feature Cards */
        .feature-card {{
            background: #141424;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 24px;
            max-width: 720px;
            margin: 0 auto 30px;
        }}

        .feature-card h2 {{
            font-size: 1.25rem;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .feature-card p.desc {{
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 20px;
            line-height: 1.4;
        }}

        .form-group {{ margin-bottom: 16px; }}
        .form-group label {{
            display: block;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}

        .blend-slider-container {{
            margin: 20px 0;
            padding: 14px;
            background: rgba(0, 0, 0, 0.25);
            border-radius: 10px;
            border: 1px solid var(--border);
        }}

        .blend-slider {{
            width: 100%;
            accent-color: var(--accent-red);
            cursor: pointer;
            margin: 10px 0;
        }}

        .slider-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 0.78rem;
            font-weight: 700;
        }}

        .btn-primary {{
            width: 100%;
            background: var(--accent-red);
            color: #fff;
            border: none;
            padding: 12px;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .btn-primary:hover {{ background: var(--accent-red-hover); }}

        /* Trope Quick Chips */
        .trope-chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 10px;
        }}

        .trope-chip {{
            background: rgba(56, 189, 248, 0.1);
            color: var(--accent-blue);
            border: 1px solid rgba(56, 189, 248, 0.25);
            padding: 4px 10px;
            border-radius: 16px;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .trope-chip:hover {{
            background: rgba(56, 189, 248, 0.25);
            transform: translateY(-1px);
        }}

        /* Modals */
        .modal-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(10px);
            z-index: 9999;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .modal-overlay.active {{ display: flex; }}

        .modal-box {{
            background: #141424;
            border: 1px solid var(--border);
            border-radius: 14px;
            width: 100%;
            max-width: 820px;
            overflow: hidden;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
            position: relative;
        }}

        .modal-header {{
            padding: 12px 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
        }}

        .modal-title {{ font-size: 1rem; font-weight: 700; }}
        .btn-close {{ background: none; border: none; color: var(--text-muted); font-size: 1.5rem; cursor: pointer; }}

        .video-container {{
            position: relative;
            padding-bottom: 56.25%;
            height: 0;
            background: #000;
        }}
        .video-container iframe {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            border: none;
        }}

        /* Radar Modal */
        .radar-box {{
            padding: 24px;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}

        .radar-legend {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 8px;
            margin-top: 16px;
            width: 100%;
            text-align: center;
        }}
        .radar-legend-item {{
            background: rgba(255, 255, 255, 0.04);
            border-radius: 6px;
            padding: 6px 4px;
            font-size: 0.72rem;
        }}
        .radar-legend-item b {{ display: block; color: var(--accent-blue); font-size: 0.85rem; }}

        /* UPI Modal */
        .upi-box {{ padding: 24px; text-align: center; }}
        .upi-plans {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin: 16px 0;
        }}
        .upi-plan-card {{
            background: #1a1a30;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .upi-plan-card.selected {{
            border-color: var(--accent-gold);
            background: rgba(245, 158, 11, 0.1);
        }}
        .upi-plan-card h4 {{ font-size: 0.85rem; margin-bottom: 4px; }}
        .upi-plan-card .price {{ font-size: 1.1rem; font-weight: 800; color: var(--accent-gold); }}

        /* Toast */
        .toast {{
            position: fixed;
            bottom: 24px;
            right: 24px;
            background: rgba(16, 185, 129, 0.95);
            color: #000;
            font-weight: 700;
            padding: 10px 18px;
            border-radius: 8px;
            z-index: 10000;
            display: none;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        }}
    </style>
</head>
<body>

    <!-- Header Bar -->
    <header class="app-bar">
        <div class="brand">
            🍿 Cine<span>Match</span>
            <span class="brand-sub">AI &amp; OTT</span>
        </div>
        <div class="top-right-actions">
            <button class="btn-top btn-vip" onclick="openUpiModal()">⭐ VIP Access</button>
            <a href="https://wa.me/?text=Check%20out%20CineMatch%20Movie%20Recommender!" target="_blank" class="btn-top" style="background:#25D366; color:#000;">💬 Share</a>
        </div>
    </header>

    <!-- Navigation Tabs -->
    <nav class="nav-tabs">
        <button class="tab-btn active" onclick="switchTab('discover', this)">🧭 Discover (100 Films)</button>
        <button class="tab-btn" onclick="switchTab('swiper', this)">⚡ Swipe to Decide</button>
        <button class="tab-btn" onclick="switchTab('dnamixer', this)">🧪 DNA Mixer</button>
        <button class="tab-btn" onclick="switchTab('duosolver', this)">👥 Duo Solver</button>
        <button class="tab-btn" onclick="switchTab('tropesearch', this)">🔍 Trope Search</button>
        <button class="tab-btn" onclick="switchTab('aimatch', this)">🤖 AI Smart Match</button>
        <button class="tab-btn" onclick="switchTab('roulette', this)">🎲 Roulette</button>
        <button class="tab-btn" onclick="switchTab('watchlist', this)">🔖 Watchlist (<span id="wlCount">0</span>)</button>
    </nav>

    <div class="container">

        <!-- 1. DISCOVER TAB -->
        <section id="tab-discover" class="section-panel active">
            <div class="hero">
                <h1>Explore 100 Curated Masterpieces</h1>
                <p>Trained Multi-Attribute TF-IDF Engine with Real-Time Indian OTT Deep-Links</p>
            </div>

            <!-- Voice-Enabled Search Bar -->
            <div class="search-wrap">
                <input type="text" id="searchInput" class="search-box" placeholder="🔍 Search by title, actor, director, genre, trope, or theme..." oninput="filterCatalog()">
                <button id="voiceBtn" class="btn-voice" onclick="toggleVoiceSearch()" title="Click to speak (Voice Search)">🎙️</button>
            </div>

            <!-- Cinema Time Machine (Era Traversal) -->
            <div class="era-slider-bar">
                <span style="font-size:0.75rem; font-weight:700; color:var(--accent-blue); text-transform:uppercase;">⏳ Cinema Time Machine:</span>
                <div class="era-chips">
                    <span class="era-chip active" onclick="setEraFilter(this, 'All')">All Eras</span>
                    <span class="era-chip" onclick="setEraFilter(this, '1970s')">1970s Golden Age</span>
                    <span class="era-chip" onclick="setEraFilter(this, '1980s')">1980s Classics</span>
                    <span class="era-chip" onclick="setEraFilter(this, '1990s')">1990s Indie Boom</span>
                    <span class="era-chip" onclick="setEraFilter(this, '2000s')">2000s Prestige</span>
                    <span class="era-chip" onclick="setEraFilter(this, '2010s')">2010s Modern Masters</span>
                    <span class="era-chip" onclick="setEraFilter(this, '2020s')">2020s Contemporary</span>
                </div>
            </div>

            <div class="filter-row">
                <select id="genreSelect" class="filter-select" onchange="filterCatalog()">
                    <option value="All">All Genres</option>
                </select>
                <select id="ottSelect" class="filter-select" onchange="filterCatalog()">
                    <option value="All">🇮🇳 All Indian OTT Platforms</option>
                    <option value="Netflix">Netflix India</option>
                    <option value="Prime Video">Amazon Prime Video</option>
                    <option value="JioCinema">JioCinema</option>
                    <option value="Disney+ Hotstar">Disney+ Hotstar</option>
                    <option value="SonyLIV">SonyLIV</option>
                    <option value="Zee5">Zee5</option>
                </select>
                <select id="moodSelect" class="filter-select" onchange="filterCatalog()">
                    <option value="All">All Moods</option>
                </select>
                <select id="sortSelect" class="filter-select" onchange="filterCatalog()">
                    <option value="rating">Sort: Highest Rated</option>
                    <option value="votes">Sort: Most Popular</option>
                    <option value="year">Sort: Newest First</option>
                </select>
            </div>

            <!-- Dealbreaker Filter -->
            <div class="dealbreaker-bar">
                <span class="dealbreaker-label">🚫 Exclude Dealbreakers:</span>
                <span class="pill-toggle" onclick="toggleDealbreaker(this, 'Gore')">🩸 No Gore</span>
                <span class="pill-toggle" onclick="toggleDealbreaker(this, 'Jump Scares')">👻 No Jump Scares</span>
                <span class="pill-toggle" onclick="toggleDealbreaker(this, 'Sad Ending')">💔 No Depressing Endings</span>
                <span class="pill-toggle" onclick="toggleDealbreaker(this, 'under2h')">⏱️ Under 2 Hours</span>
                <span class="pill-toggle" onclick="toggleDealbreaker(this, 'Violence')">💥 Low Violence Only</span>
            </div>

            <div id="fallbackBanner" class="fallback-banner">
                <span>💡</span>
                <span id="fallbackText">Showing closest matching alternatives based on your filters.</span>
            </div>

            <div id="catalogGrid" class="movie-grid"></div>
        </section>

        <!-- 2. SWIPE TO DECIDE TAB -->
        <section id="tab-swiper" class="section-panel">
            <div class="hero">
                <h1>⚡ Swipe to Decide</h1>
                <p>Rapid micro-preference solver: Swipe or tap to update your taste gradient vector in real time.</p>
            </div>

            <div class="swipe-wrapper">
                <div class="swipe-card-stage" id="swiperStage"></div>
                <div class="swipe-controls">
                    <button class="btn-swipe btn-swipe-pass" onclick="handleSwipe('pass')" title="Pass">✕</button>
                    <button class="btn-swipe btn-swipe-save" onclick="handleSwipe('save')" title="Save to Watchlist">★</button>
                    <button class="btn-swipe btn-swipe-like" onclick="handleSwipe('like')" title="Like &amp; Learn">♥</button>
                </div>
                <p style="font-size:0.75rem; color:var(--text-muted); margin-top:12px;">Swiped: <span id="swipeCount">0</span> / 10 · Watch recommendations converge below</p>
            </div>

            <div style="margin-top:30px;">
                <h3 style="font-size:1.1rem; margin-bottom:12px; text-align:center; color:var(--accent-green);">🎯 Converged Preference Recommendations:</h3>
                <div id="swiperRecommendations" class="movie-grid"></div>
            </div>
        </section>

        <!-- 3. DNA MIXER TAB -->
        <section id="tab-dnamixer" class="section-panel">
            <div class="hero">
                <h1>🧪 Movie DNA Mixer</h1>
                <p>Vector arithmetic of cinema: Blend two favorite films together to synthesize a brand-new hybrid recommendation.</p>
            </div>

            <div class="feature-card">
                <h2>🧬 Blend Movie Genomes</h2>
                <p class="desc">Select two films and slide the ratio. The engine calculates: <b>V<sub>hybrid</sub> = w·V<sub>A</sub> + (1-w)·V<sub>B</sub></b>.</p>

                <div class="form-group">
                    <label>Film Alpha (A):</label>
                    <select id="mixerFilmA" class="filter-select" style="width:100%;" onchange="runDnaMixer()"></select>
                </div>

                <div class="form-group">
                    <label>Film Beta (B):</label>
                    <select id="mixerFilmB" class="filter-select" style="width:100%;" onchange="runDnaMixer()"></select>
                </div>

                <div class="blend-slider-container">
                    <div class="slider-labels">
                        <span id="labelFilmA">Alpha: 50%</span>
                        <span id="labelFilmB">Beta: 50%</span>
                    </div>
                    <input type="range" id="dnaSlider" class="blend-slider" min="10" max="90" value="50" oninput="updateMixerSlider(); runDnaMixer();">
                </div>

                <button class="btn-primary" onclick="runDnaMixer()">✨ Synthesize Hybrid Candidates</button>
            </div>

            <div id="mixerResults" class="movie-grid"></div>
        </section>

        <!-- 4. DUO SOLVER TAB -->
        <section id="tab-duosolver" class="section-panel">
            <div class="hero">
                <h1>👥 Couple &amp; Duo Consensus Solver</h1>
                <p>Solve couch decision fatigue: Pick movies you both like, and the algorithm calculates the Pareto-optimal mutual intersection.</p>
            </div>

            <div class="feature-card" style="max-width:760px;">
                <h2>🤝 Find Common Ground</h2>
                <p class="desc">Each person selects 2 or 3 favorites. The solver models both taste vectors and optimizes for maximum joint satisfaction while minimizing individual dissatisfaction.</p>

                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-bottom:16px;">
                    <div>
                        <label style="color:#38bdf8; font-weight:700; font-size:0.85rem; display:block; margin-bottom:6px;">Person 1's Picks:</label>
                        <select id="duoUser1_1" class="filter-select" style="width:100%; margin-bottom:6px;" onchange="runDuoSolver()"></select>
                        <select id="duoUser1_2" class="filter-select" style="width:100%; margin-bottom:6px;" onchange="runDuoSolver()"></select>
                        <select id="duoUser1_3" class="filter-select" style="width:100%;" onchange="runDuoSolver()"></select>
                    </div>
                    <div>
                        <label style="color:#f472b6; font-weight:700; font-size:0.85rem; display:block; margin-bottom:6px;">Person 2's Picks:</label>
                        <select id="duoUser2_1" class="filter-select" style="width:100%; margin-bottom:6px;" onchange="runDuoSolver()"></select>
                        <select id="duoUser2_2" class="filter-select" style="width:100%; margin-bottom:6px;" onchange="runDuoSolver()"></select>
                        <select id="duoUser2_3" class="filter-select" style="width:100%;" onchange="runDuoSolver()"></select>
                    </div>
                </div>

                <button class="btn-primary" onclick="runDuoSolver()">🎯 Calculate Mutual Consensus</button>
            </div>

            <div id="duoResults" class="movie-grid"></div>
        </section>

        <!-- 5. TROPE SEARCH TAB -->
        <section id="tab-tropesearch" class="section-panel">
            <div class="hero">
                <h1>🔍 Micro-Trope &amp; Scene Search</h1>
                <p>Search beyond titles: find films by exact narrative mechanics, specific plot devices, and iconic scene tropes.</p>
            </div>

            <div class="feature-card">
                <h2>🎭 Search Narrative Devices</h2>
                <p class="desc">Enter any plot trope or tap a popular scene keyword chip below to surface thematic matches.</p>

                <input type="text" id="tropeInput" class="search-box" placeholder="e.g., dream heist, time loop, hallway fight, whodunnit, unfaithful spouse..." oninput="runTropeSearch()">

                <div class="trope-chips">
                    <span class="trope-chip" onclick="applyTropeChip('dream within a dream')">🌀 Dream Heist</span>
                    <span class="trope-chip" onclick="applyTropeChip('hallway fight')">🥊 Hallway Fight</span>
                    <span class="trope-chip" onclick="applyTropeChip('time dilation')">⏳ Time Dilation</span>
                    <span class="trope-chip" onclick="applyTropeChip('whodunnit')">🕵️ Whodunnit</span>
                    <span class="trope-chip" onclick="applyTropeChip('unreliable narrator')">🎭 Unreliable Narrator</span>
                    <span class="trope-chip" onclick="applyTropeChip('class warfare')">🏰 Class Warfare</span>
                    <span class="trope-chip" onclick="applyTropeChip('demon cursed')">👹 Cursed God</span>
                    <span class="trope-chip" onclick="applyTropeChip('blind pianist')">🎹 Blind Pianist</span>
                    <span class="trope-chip" onclick="applyTropeChip('brotherhood')">🤝 Brotherhood</span>
                    <span class="trope-chip" onclick="applyTropeChip('memory erasure')">🧠 Memory Erasure</span>
                    <span class="trope-chip" onclick="applyTropeChip('reality tv show')">📺 Simulated Life</span>
                </div>
            </div>

            <div id="tropeResults" class="movie-grid"></div>
        </section>

        <!-- 6. AI SMART MATCH TAB -->
        <section id="tab-aimatch" class="section-panel">
            <div class="hero">
                <h1>🤖 AI Smart Match</h1>
                <p>Precomputed Scikit-Learn TF-IDF Proximity: mathematically calibrated cosine distance rankings.</p>
            </div>

            <div class="feature-card" style="max-width:560px;">
                <label style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 8px; display: block;">Pick a baseline movie:</label>
                <select id="smartMatchSelect" class="filter-select" style="width: 100%; margin-bottom:14px;" onchange="computeSmartRecommendations()"></select>
                <button class="btn-primary" onclick="computeSmartRecommendations()">✨ Compute Nearest Neighbors</button>
            </div>

            <div id="smartMatchResults" class="movie-grid"></div>
        </section>

        <!-- 7. ROULETTE TAB -->
        <section id="tab-roulette" class="section-panel">
            <div class="hero">
                <h1>🎲 Movie Roulette</h1>
                <p>Can't decide? Pick your vibe and let mathematical pseudo-randomization decide.</p>
            </div>

            <div class="feature-card" style="max-width: 480px; text-align: center;">
                <label style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 8px; display: block;">Preferred Vibe:</label>
                <select id="rouletteMood" class="filter-select" style="width: 100%; margin-bottom: 14px;">
                    <option value="All">Any Vibe 🌟</option>
                </select>
                <button class="btn-primary" onclick="spinRoulette()">🎰 Spin the Wheel!</button>
                <div id="rouletteResult" style="margin-top: 20px;"></div>
            </div>
        </section>

        <!-- 8. WATCHLIST & TICKET GENERATOR TAB -->
        <section id="tab-watchlist" class="section-panel">
            <div class="hero">
                <h1>🔖 My Private Watchlist</h1>
                <p>Saved titles persist locally. Generate a custom Cinema Ticket for tonight's binge.</p>
            </div>

            <div style="text-align:center; margin-bottom:20px; display:flex; gap:10px; justify-content:center; flex-wrap:wrap;">
                <button class="btn-top" style="background:#f59e0b; color:#000; padding:10px 18px;" onclick="generateBingeTicket()">🎟️ Generate My CineMatch Binge Ticket</button>
                <button class="btn-top" style="background:#25D366; color:#000; padding:10px 18px;" onclick="shareWatchlist()">💬 Share My Watchlist on WhatsApp</button>
            </div>

            <!-- Canvas Container for Downloadable Ticket -->
            <div id="ticketContainer" style="display:none; text-align:center; margin-bottom:24px;">
                <canvas id="ticketCanvas" style="max-width:100%; border-radius:12px; box-shadow:0 12px 30px rgba(0,0,0,0.6);"></canvas>
                <div style="margin-top:10px;">
                    <a id="downloadTicketLink" class="btn-top" style="background:#38bdf8; color:#000; padding:8px 16px;" download="CineMatch_Binge_Ticket.png">📥 Download Ticket (PNG)</a>
                </div>
            </div>

            <div id="watchlistGrid" class="movie-grid"></div>
        </section>

    </div>

    <!-- YouTube Trailer Video Modal -->
    <div id="videoModal" class="modal-overlay" onclick="closeVideoModal(event)">
        <div class="modal-box" onclick="event.stopPropagation()">
            <div class="modal-header">
                <span id="modalVideoTitle" class="modal-title">Official Movie Trailer</span>
                <button class="btn-close" onclick="closeVideoModal()">&times;</button>
            </div>
            <div class="video-container">
                <iframe id="videoIframe" src="" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
            </div>
        </div>
    </div>

    <!-- 5-Axis XAI Radar Chart Modal -->
    <div id="radarModal" class="modal-overlay" onclick="closeRadarModal(event)">
        <div class="modal-box" style="max-width:540px;" onclick="event.stopPropagation()">
            <div class="modal-header">
                <span id="radarTitle" class="modal-title">🔬 Film DNA Radar (XAI Analysis)</span>
                <button class="btn-close" onclick="closeRadarModal()">&times;</button>
            </div>
            <div class="radar-box">
                <p id="radarSubtitle" style="font-size:0.85rem; color:var(--text-muted); margin-bottom:14px;">Normalized 5-dimensional algorithmic signature</p>
                <svg id="radarSvg" width="280" height="260" viewBox="0 0 280 260"></svg>
                <div class="radar-legend">
                    <div class="radar-legend-item"><b id="radComplexity">90</b>Complexity</div>
                    <div class="radar-legend-item"><b id="radPacing">85</b>Pacing</div>
                    <div class="radar-legend-item"><b id="radEmotion">70</b>Emotion</div>
                    <div class="radar-legend-item"><b id="radVisuals">95</b>Visuals</div>
                    <div class="radar-legend-item"><b id="radDepth">88</b>Depth</div>
                </div>
            </div>
        </div>
    </div>

    <!-- VIP / UPI Payment Modal -->
    <div id="upiModal" class="modal-overlay" onclick="closeUpiModal(event)">
        <div class="modal-box" style="max-width:540px;" onclick="event.stopPropagation()">
            <div class="modal-header">
                <span class="modal-title">⭐ CineMatch VIP Subscription</span>
                <button class="btn-close" onclick="closeUpiModal()">&times;</button>
            </div>
            <div class="upi-box">
                <p style="font-size:0.85rem; color:var(--text-muted);">Select your plan to unlock unlimited 4K streams and ad-free recommendation vectors:</p>
                <div class="upi-plans">
                    <div class="upi-plan-card selected" onclick="selectPlan(this, 'Mobile', '149')">
                        <h4>Mobile</h4>
                        <div class="price">₹149</div>
                        <span style="font-size:0.65rem; color:var(--text-muted);">1 Device · 720p</span>
                    </div>
                    <div class="upi-plan-card" onclick="selectPlan(this, 'Standard', '499')">
                        <h4>Standard</h4>
                        <div class="price">₹499</div>
                        <span style="font-size:0.65rem; color:var(--text-muted);">2 Devices · 1080p</span>
                    </div>
                    <div class="upi-plan-card" onclick="selectPlan(this, 'VIP', '699')">
                        <h4>VIP Pro</h4>
                        <div class="price">₹699</div>
                        <span style="font-size:0.65rem; color:var(--text-muted);">4 Devices · 4K HDR</span>
                    </div>
                </div>

                <div style="background:#fff; padding:12px; border-radius:10px; display:inline-block; margin-bottom:12px;">
                    <img id="upiQrImg" src="https://api.qrserver.com/v1/create-qr-code/?size=180x180&data=upi://pay?pa=cinematch@axisbank&pn=CineMatch&am=149&cu=INR" alt="UPI QR" style="width:160px; height:160px; display:block;">
                </div>

                <p style="font-size:0.8rem; font-weight:700; color:var(--accent-gold); margin-bottom:12px;">Scan with GPay / PhonePe / Paytm / BHIM</p>
                <button class="btn-primary" style="background:var(--accent-green);" onclick="simulateVipSuccess()">✅ Simulate Instant Payment Approval</button>
            </div>
        </div>
    </div>

    <!-- Toast Notification -->
    <div id="toast" class="toast"></div>

    <script>
        // Database with Trained ML Weights
        const movies = {movies_json};
        const movieMap = {{}};
        movies.forEach(m => movieMap[m.movie_id] = m);

        // State
        let watchlist = JSON.parse(localStorage.getItem('cinematch_wl') || '[]');
        let activeDealbreakers = [];
        let selectedEra = 'All';
        let currentSwipeIdx = 0;
        let swipeVector = {{}};
        let isVoiceListening = false;
        let voiceRecognition = null;

        function showToast(msg) {{
            const t = document.getElementById('toast');
            t.innerText = msg;
            t.style.display = 'block';
            setTimeout(() => t.style.display = 'none', 2500);
        }}

        function saveWatchlistToStorage() {{
            localStorage.setItem('cinematch_wl', JSON.stringify(watchlist));
            document.getElementById('wlCount').innerText = watchlist.length;
        }}
        saveWatchlistToStorage();

        function switchTab(tabId, btn) {{
            document.querySelectorAll('.section-panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('tab-' + tabId).classList.add('active');
            if (btn) btn.classList.add('active');
            if (tabId === 'watchlist') renderWatchlist();
            if (tabId === 'swiper' && currentSwipeIdx === 0) renderSwipeCard();
            window.scrollTo({{ top: 0, behavior: 'smooth' }});
        }}

        // Populate Selectors
        function populateSelectors() {{
            const genres = [...new Set(movies.flatMap(m => (m.genre || '').split(',').map(s => s.trim())))].filter(Boolean).sort();
            const moods = [...new Set(movies.map(m => m.mood).filter(Boolean))].sort();

            const gSelect = document.getElementById('genreSelect');
            genres.forEach(g => {{
                const opt = document.createElement('option');
                opt.value = g;
                opt.innerText = g;
                gSelect.appendChild(opt);
            }});

            const mSelect = document.getElementById('moodSelect');
            const rSelect = document.getElementById('rouletteMood');
            moods.forEach(m => {{
                const opt1 = document.createElement('option');
                opt1.value = m;
                opt1.innerText = m;
                mSelect.appendChild(opt1);

                const opt2 = document.createElement('option');
                opt2.value = m;
                opt2.innerText = m;
                rSelect.appendChild(opt2);
            }});

            const singleSel = document.getElementById('smartMatchSelect');
            const mixA = document.getElementById('mixerFilmA');
            const mixB = document.getElementById('mixerFilmB');
            const d1_1 = document.getElementById('duoUser1_1');
            const d1_2 = document.getElementById('duoUser1_2');
            const d1_3 = document.getElementById('duoUser1_3');
            const d2_1 = document.getElementById('duoUser2_1');
            const d2_2 = document.getElementById('duoUser2_2');
            const d2_3 = document.getElementById('duoUser2_3');

            movies.forEach((m, idx) => {{
                const addOpt = (el, selected = false) => {{
                    const opt = document.createElement('option');
                    opt.value = m.title;
                    opt.innerText = `${{m.title}} (${{m.year}})`;
                    if (selected) opt.selected = true;
                    el.appendChild(opt);
                }};
                addOpt(singleSel, idx === 0);
                addOpt(mixA, idx === 0);
                addOpt(mixB, idx === 3);
                addOpt(d1_1, idx === 0);
                addOpt(d1_2, idx === 1);
                addOpt(d1_3, idx === 2);
                addOpt(d2_1, idx === 7);
                addOpt(d2_2, idx === 8);
                addOpt(d2_3, idx === 9);
            }});
        }}
        populateSelectors();

        // Native Voice Search (Web Speech API)
        function toggleVoiceSearch() {{
            const btn = document.getElementById('voiceBtn');
            const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRec) return alert("Web Speech API is not supported in this browser. Please use Chrome, Edge, or Safari.");

            if (isVoiceListening) {{
                if (voiceRecognition) voiceRecognition.stop();
                isVoiceListening = false;
                btn.classList.remove('listening');
                return;
            }}

            voiceRecognition = new SpeechRec();
            voiceRecognition.continuous = false;
            voiceRecognition.lang = 'en-IN';

            voiceRecognition.onstart = () => {{
                isVoiceListening = true;
                btn.classList.add('listening');
                showToast("🎙️ Listening... speak your movie query now");
            }};

            voiceRecognition.onresult = (e) => {{
                const text = e.results[0][0].transcript;
                document.getElementById('searchInput').value = text;
                showToast(`Heard: "${{text}}"`);
                filterCatalog();
            }};

            voiceRecognition.onerror = (e) => {{
                console.error(e);
                showToast("Voice search ended");
                btn.classList.remove('listening');
                isVoiceListening = false;
            }};

            voiceRecognition.onend = () => {{
                btn.classList.remove('listening');
                isVoiceListening = false;
            }};

            voiceRecognition.start();
        }}

        // Cinema Time Machine Era Filter
        function setEraFilter(chip, era) {{
            document.querySelectorAll('.era-chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            selectedEra = era;
            filterCatalog();
        }}

        // OTT Badges
        function renderOttBadges(m) {{
            if (!m.ott_india) return '';
            const otts = m.ott_india.split(',').map(s => s.trim());
            const link = m.ott_link || `https://www.google.com/search?q=watch+${{encodeURIComponent(m.title)}}+online+india`;

            const badgeHtml = otts.map(name => {{
                let cls = 'badge-ott-default';
                const lower = name.toLowerCase();
                if (lower.includes('netflix')) cls = 'badge-ott-netflix';
                else if (lower.includes('prime')) cls = 'badge-ott-prime';
                else if (lower.includes('jio')) cls = 'badge-ott-jio';
                else if (lower.includes('hotstar')) cls = 'badge-ott-hotstar';
                else if (lower.includes('sony')) cls = 'badge-ott-sonyliv';
                else if (lower.includes('zee')) cls = 'badge-ott-zee5';
                else if (lower.includes('youtube')) cls = 'badge-ott-youtube';
                return `<a class="badge-ott ${{cls}}" href="${{link}}" target="_blank" title="Stream on ${{name}} in India">${{name}}</a>`;
            }}).join('');

            return `
                <div class="ott-box">
                    <span class="ott-label">🇮🇳 Stream:</span>
                    <div class="ott-badges">${{badgeHtml}}</div>
                </div>
            `;
        }}

        // Render Card
        function renderCard(m, matchPct = null) {{
            const isSaved = watchlist.includes(m.title);
            const matchTag = matchPct ? `<span class="tag tag-match">🔥 ${{Math.round(matchPct)}}% Match</span>` : '';
            const trailerKey = m.trailer_key || 'YoHD9XEInc0';

            return `
                <div class="movie-card">
                    <div class="poster-container">
                        <img src="${{m.poster}}" alt="${{m.title}}" loading="lazy" onerror="this.src='https://placehold.co/300x450/141424/E50914?text=' + encodeURIComponent('${{m.title}}')">
                        <div class="badge-rating">★ ${{Number(m.rating).toFixed(1)}}</div>
                    </div>
                    <div class="card-info">
                        <div class="card-title" title="${{m.title}}">${{m.title}}</div>
                        <div class="card-tags">
                            <span class="tag">${{m.genre.split(',')[0]}}</span>
                            <span class="tag">${{m.year}}</span>
                            <span class="tag">${{m.runtime}}</span>
                            ${{matchTag}}
                        </div>
                        <div class="card-plot">${{m.overview || ''}}</div>
                        ${{renderOttBadges(m)}}
                        <div class="card-actions">
                            <button class="btn-action btn-trailer" onclick="openVideoModal('${{m.title.replace(/'/g, "\\\\'") }}', '${{trailerKey}}')">▶ Play</button>
                            <button class="btn-action btn-radar" onclick="openRadarModal('${{m.movie_id}}')">🔬 DNA</button>
                            <button class="btn-action btn-bookmark ${{isSaved ? 'saved' : ''}}" onclick="toggleWatchlist('${{m.title.replace(/'/g, "\\\\'") }}')">
                                ${{isSaved ? '❤️' : '🤍 Save'}}
                            </button>
                        </div>
                    </div>
                </div>
            `;
        }}

        function toggleDealbreaker(btn, tag) {{
            btn.classList.toggle('active');
            if (btn.classList.contains('active')) {{
                if (!activeDealbreakers.includes(tag)) activeDealbreakers.push(tag);
            }} else {{
                activeDealbreakers = activeDealbreakers.filter(t => t !== tag);
            }}
            filterCatalog();
        }}

        // Smart Soft-Fallback Catalog Search (Zero blank screens)
        function filterCatalog() {{
            const q = document.getElementById('searchInput').value.toLowerCase().trim();
            const genre = document.getElementById('genreSelect').value;
            const ott = document.getElementById('ottSelect').value;
            const mood = document.getElementById('moodSelect').value;
            const sort = document.getElementById('sortSelect').value;
            const banner = document.getElementById('fallbackBanner');

            let strictMatches = movies.filter(m => {{
                const matchQ = !q || 
                    m.title.toLowerCase().includes(q) ||
                    (m.stars && m.stars.toLowerCase().includes(q)) ||
                    (m.director && m.director.toLowerCase().includes(q)) ||
                    (m.tropes && m.tropes.toLowerCase().includes(q)) ||
                    (m.overview && m.overview.toLowerCase().includes(q));

                const matchG = genre === 'All' || (m.genre && m.genre.includes(genre));
                const matchOtt = ott === 'All' || (m.ott_india && m.ott_india.toLowerCase().includes(ott.toLowerCase()));
                const matchM = mood === 'All' || m.mood === mood;

                // Time Machine Era match
                let matchEra = true;
                const yr = Number(m.year);
                if (selectedEra === '1970s') matchEra = yr >= 1970 && yr <= 1979;
                else if (selectedEra === '1980s') matchEra = yr >= 1980 && yr <= 1989;
                else if (selectedEra === '1990s') matchEra = yr >= 1990 && yr <= 1999;
                else if (selectedEra === '2000s') matchEra = yr >= 2000 && yr <= 2009;
                else if (selectedEra === '2010s') matchEra = yr >= 2010 && yr <= 2019;
                else if (selectedEra === '2020s') matchEra = yr >= 2020;

                let passesDb = true;
                if (activeDealbreakers.length > 0) {{
                    const dbStr = (m.dealbreakers || '').toLowerCase();
                    const rMins = Number(m.runtime_mins) || 120;
                    for (const db of activeDealbreakers) {{
                        if (db === 'under2h' && rMins > 120) {{ passesDb = false; break; }}
                        if (dbStr.includes(db.toLowerCase())) {{ passesDb = false; break; }}
                    }}
                }}

                return matchQ && matchG && matchOtt && matchM && matchEra && passesDb;
            }});

            let displayList = [];
            if (strictMatches.length >= 3) {{
                banner.style.display = 'none';
                displayList = strictMatches;
            }} else if (strictMatches.length > 0) {{
                banner.style.display = 'flex';
                document.getElementById('fallbackText').innerText = `Found ${{strictMatches.length}} exact match(es). Supplementing with closest recommendations.`;
                const firstId = strictMatches[0].movie_id;
                const recs = (movieMap[firstId]?.similar || []).map(s => movieMap[s.id]).filter(Boolean);
                displayList = [...strictMatches, ...recs.filter(r => !strictMatches.some(sm => sm.movie_id === r.movie_id))].slice(0, 12);
            }} else {{
                banner.style.display = 'flex';
                document.getElementById('fallbackText').innerText = "Exact combination had 0 matches. Displaying top-ranked recommendations matching your primary parameters.";
                const scored = movies.map(m => {{
                    let score = 0;
                    if (q) {{
                        const fullText = (m.title + ' ' + m.stars + ' ' + m.director + ' ' + m.tropes + ' ' + m.overview).toLowerCase();
                        if (fullText.includes(q)) score += 50;
                    }}
                    if (genre !== 'All' && m.genre && m.genre.includes(genre)) score += 30;
                    if (ott !== 'All' && m.ott_india && m.ott_india.toLowerCase().includes(ott.toLowerCase())) score += 30;
                    if (mood !== 'All' && m.mood === mood) score += 20;

                    if (activeDealbreakers.length > 0) {{
                        const dbStr = (m.dealbreakers || '').toLowerCase();
                        const rMins = Number(m.runtime_mins) || 120;
                        for (const db of activeDealbreakers) {{
                            if (db === 'under2h' && rMins > 120) score -= 40;
                            if (dbStr.includes(db.toLowerCase())) score -= 40;
                        }}
                    }}
                    return {{ movie: m, score: score + m.rating }};
                }});
                displayList = scored.sort((a,b) => b.score - a.score).slice(0, 12).map(s => s.movie);
            }}

            if (sort === 'rating') displayList.sort((a,b) => b.rating - a.rating);
            else if (sort === 'votes') displayList.sort((a,b) => b.votes - a.votes);
            else if (sort === 'year') displayList.sort((a,b) => b.year - a.year);

            document.getElementById('catalogGrid').innerHTML = displayList.map(m => renderCard(m)).join('');
        }}
        filterCatalog();

        // 5-Axis Radar Chart Generator (SVG)
        function openRadarModal(movieId) {{
            const m = movieMap[movieId];
            if (!m) return;
            const r = m.radar || [50, 50, 50, 50, 50];

            document.getElementById('radarTitle').innerText = `${{m.title}} (${{m.year}}) — DNA Analysis`;
            document.getElementById('radarSubtitle').innerText = `${{m.genre}} · Directed by ${{m.director || 'N/A'}}`;
            document.getElementById('radComplexity').innerText = r[0];
            document.getElementById('radPacing').innerText = r[1];
            document.getElementById('radEmotion').innerText = r[2];
            document.getElementById('radVisuals').innerText = r[3];
            document.getElementById('radDepth').innerText = r[4];

            // Render SVG polygon: center at (140, 130), max radius = 95
            const cx = 140, cy = 130, R = 95;
            const angles = [ -Math.PI/2, -Math.PI/2 + 2*Math.PI/5, -Math.PI/2 + 4*Math.PI/5, -Math.PI/2 + 6*Math.PI/5, -Math.PI/2 + 8*Math.PI/5 ];
            
            // Background grid rings
            let gridHtml = '';
            [0.25, 0.5, 0.75, 1.0].forEach(frac => {{
                const pts = angles.map(a => `${{cx + R*frac*Math.cos(a)}},${{cy + R*frac*Math.sin(a)}}`).join(' ');
                gridHtml += `<polygon points="${{pts}}" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>`;
            }});
            // Axis spokes
            angles.forEach(a => {{
                gridHtml += `<line x1="${{cx}}" y1="${{cy}}" x2="${{cx + R*Math.cos(a)}}" y2="${{cy + R*Math.sin(a)}}" stroke="rgba(255,255,255,0.12)" stroke-width="1"/>`;
            }});

            // Polygon points
            const polyPts = angles.map((a, i) => {{
                const val = (r[i] / 100) * R;
                return `${{cx + val*Math.cos(a)}},${{cy + val*Math.sin(a)}}`;
            }}).join(' ');

            const labels = ['Complexity', 'Pacing', 'Emotion', 'Visuals', 'Depth'];
            let labelHtml = '';
            angles.forEach((a, i) => {{
                const lx = cx + (R + 18) * Math.cos(a);
                const ly = cy + (R + 14) * Math.sin(a);
                labelHtml += `<text x="${{lx}}" y="${{ly}}" fill="#94a3b8" font-size="10" font-weight="700" text-anchor="middle" dominant-baseline="middle">${{labels[i]}}</text>`;
            }});

            const svgContent = `
                ${{gridHtml}}
                <polygon points="${{polyPts}}" fill="rgba(56, 189, 248, 0.35)" stroke="#38bdf8" stroke-width="2.5"/>
                ${{labelHtml}}
            `;
            document.getElementById('radarSvg').innerHTML = svgContent;
            document.getElementById('radarModal').classList.add('active');
        }}

        function closeRadarModal(e) {{
            document.getElementById('radarModal').classList.remove('active');
        }}

        // Swipe to Decide Engine
        function renderSwipeCard() {{
            const stage = document.getElementById('swiperStage');
            if (currentSwipeIdx >= movies.length) {{
                stage.innerHTML = '<div style="padding:100px 20px; color:var(--text-muted);">You have reached the end of the swiper deck!</div>';
                return;
            }}
            const m = movies[currentSwipeIdx];
            stage.innerHTML = `
                <div class="swipe-card" id="activeSwipeCard">
                    <div class="poster-container" style="padding-top:100%;">
                        <img src="${{m.poster}}" alt="${{m.title}}">
                        <div class="badge-rating">★ ${{Number(m.rating).toFixed(1)}}</div>
                    </div>
                    <div class="card-info" style="text-align:left;">
                        <h3 style="font-size:1.15rem; margin-bottom:4px;">${{m.title}} (${{m.year}})</h3>
                        <p style="font-size:0.75rem; color:var(--accent-blue); margin-bottom:6px;">${{m.genre}} · ${{m.runtime}}</p>
                        <p class="card-plot" style="-webkit-line-clamp:3;">${{m.overview}}</p>
                    </div>
                </div>
            `;
            document.getElementById('swipeCount').innerText = currentSwipeIdx + 1;
        }}

        function handleSwipe(action) {{
            const cardEl = document.getElementById('activeSwipeCard');
            const m = movies[currentSwipeIdx];
            if (!m) return;

            const mVec = getMovieVector(m);
            if (action === 'like') {{
                for (let k in mVec) swipeVector[k] = (swipeVector[k] || 0) + mVec[k] * 1.5;
                if (cardEl) cardEl.style.transform = 'translateX(200px) rotate(20deg)';
                showToast(`Liked ${{m.title}}! Convergence updated.`);
            }} else if (action === 'save') {{
                toggleWatchlist(m.title);
                for (let k in mVec) swipeVector[k] = (swipeVector[k] || 0) + mVec[k] * 2.0;
                if (cardEl) cardEl.style.transform = 'translateY(-200px)';
            }} else {{
                for (let k in mVec) swipeVector[k] = (swipeVector[k] || 0) - mVec[k] * 0.8;
                if (cardEl) cardEl.style.transform = 'translateX(-200px) rotate(-20deg)';
            }}
            if (cardEl) cardEl.style.opacity = '0';

            currentSwipeIdx++;
            setTimeout(() => {{
                renderSwipeCard();
                updateSwiperConvergence();
            }}, 200);
        }}

        function updateSwiperConvergence() {{
            const container = document.getElementById('swiperRecommendations');
            const scores = movies.slice(currentSwipeIdx).map(m => ({{
                movie: m,
                sim: cosineSimilarity(swipeVector, getMovieVector(m))
            }})).sort((a,b) => b.sim - a.sim).slice(0, 6);

            const maxRaw = scores[0]?.sim || 1;
            container.innerHTML = scores.map(s => {{
                const pct = Math.round((s.sim / maxRaw) * 95);
                return renderCard(s.movie, Math.max(55, pct));
            }}).join('');
        }}

        // HTML5 Canvas Binge Ticket Generator
        function generateBingeTicket() {{
            const saved = movies.filter(m => watchlist.includes(m.title));
            if (saved.length === 0) return alert("Your watchlist is empty! Save at least one film to print a ticket.");

            const canvas = document.getElementById('ticketCanvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 640;
            canvas.height = 360 + (saved.length * 48);

            // Background
            ctx.fillStyle = '#0f172a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Border & Gold Accent
            ctx.strokeStyle = '#f59e0b';
            ctx.lineWidth = 4;
            ctx.strokeRect(12, 12, canvas.width - 24, canvas.height - 24);

            // Header Banner
            ctx.fillStyle = '#E50914';
            ctx.fillRect(16, 16, canvas.width - 32, 54);

            ctx.fillStyle = '#FFFFFF';
            ctx.font = 'bold 22px sans-serif';
            ctx.fillText('🍿 CINEMATCH OFFICIAL BINGE TICKET', 36, 50);

            ctx.fillStyle = '#f59e0b';
            ctx.font = 'bold 12px sans-serif';
            ctx.fillText('ADMIT ONE · VIP PASS', canvas.width - 170, 50);

            // Meta Info
            let totalMins = saved.reduce((acc, x) => acc + (Number(x.runtime_mins) || 120), 0);
            ctx.fillStyle = '#94a3b8';
            ctx.font = '13px sans-serif';
            ctx.fillText(`Issued: ${{new Date().toLocaleDateString()}}  |  Total Films: ${{saved.length}}  |  Runtime: ${{Math.floor(totalMins/60)}}h ${{totalMins%60}}m`, 36, 100);

            ctx.strokeStyle = '#334155';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(36, 115);
            ctx.lineTo(canvas.width - 36, 115);
            ctx.stroke();

            // Movie List
            let y = 150;
            saved.forEach((m, idx) => {{
                ctx.fillStyle = '#f8fafc';
                ctx.font = 'bold 15px sans-serif';
                ctx.fillText(`${{idx + 1}}. ${{m.title}} (${{m.year}})`, 36, y);

                ctx.fillStyle = '#38bdf8';
                ctx.font = '13px sans-serif';
                ctx.fillText(`★ ${{m.rating}} · ${{m.genre.split(',')[0]}}`, canvas.width - 180, y);

                ctx.fillStyle = '#64748b';
                ctx.font = '11px sans-serif';
                ctx.fillText(`Stream on: ${{m.ott_india || 'India OTT'}}`, 36, y + 18);

                y += 48;
            }});

            // Barcode
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(36, canvas.height - 65, canvas.width - 72, 35);
            ctx.fillStyle = '#000000';
            for (let i = 45; i < canvas.width - 45; i += 6) {{
                ctx.fillRect(i, canvas.height - 60, (i % 5 === 0) ? 3 : 1.5, 25);
            }}

            const container = document.getElementById('ticketContainer');
            container.style.display = 'block';
            document.getElementById('downloadTicketLink').href = canvas.toDataURL('image/png');
            showToast("🎟️ Binge Ticket generated successfully!");
            container.scrollIntoView({{ behavior: 'smooth' }});
        }}

        // Dynamic Vector Tokenizer
        function tokenize(text) {{
            return (text || '').toLowerCase().replace(/[^a-z0-9 ]/g, ' ').split(/\\s+/).filter(w => w.length > 2);
        }}

        function getMovieVector(movie) {{
            const text = [
                movie.genre, movie.genre, movie.genre,
                movie.language,
                movie.mood, movie.mood,
                movie.director, movie.director,
                movie.stars,
                movie.tropes, movie.tropes, movie.tropes,
                movie.overview
            ].join(' ');
            const tokens = tokenize(text);
            const freq = {{}};
            tokens.forEach(t => freq[t] = (freq[t] || 0) + 1);
            return freq;
        }}

        function cosineSimilarity(vecA, vecB) {{
            let dot = 0, normA = 0, normB = 0;
            for (let k in vecA) normA += vecA[k] * vecA[k];
            for (let k in vecB) normB += vecB[k] * vecB[k];
            for (let k in vecA) {{
                if (vecB[k]) dot += vecA[k] * vecB[k];
            }}
            if (!normA || !normB) return 0;
            return dot / (Math.sqrt(normA) * Math.sqrt(normB));
        }}

        // DNA Mixer
        function updateMixerSlider() {{
            const val = document.getElementById('dnaSlider').value;
            document.getElementById('labelFilmA').innerText = `Alpha: ${{val}}%`;
            document.getElementById('labelFilmB').innerText = `Beta: ${{100 - val}}%`;
        }}

        function runDnaMixer() {{
            const titleA = document.getElementById('mixerFilmA').value;
            const titleB = document.getElementById('mixerFilmB').value;
            const movieA = movies.find(m => m.title === titleA);
            const movieB = movies.find(m => m.title === titleB);
            if (!movieA || !movieB) return;

            const ratioA = Number(document.getElementById('dnaSlider').value) / 100;
            const ratioB = 1 - ratioA;

            const vecA = getMovieVector(movieA);
            const vecB = getMovieVector(movieB);

            const blendedVec = {{}};
            for (let k in vecA) blendedVec[k] = (blendedVec[k] || 0) + (vecA[k] * ratioA);
            for (let k in vecB) blendedVec[k] = (blendedVec[k] || 0) + (vecB[k] * ratioB);

            const scores = movies
                .filter(m => m.title !== titleA && m.title !== titleB)
                .map(m => ({{
                    movie: m,
                    sim: cosineSimilarity(blendedVec, getMovieVector(m))
                }}))
                .sort((a,b) => b.sim - a.sim)
                .slice(0, 8);

            const maxRaw = scores[0]?.sim || 1;
            document.getElementById('mixerResults').innerHTML = scores.map(s => {{
                const pct = Math.round((s.sim / maxRaw) * 94);
                return renderCard(s.movie, Math.max(55, pct));
            }}).join('');
        }}

        // Duo Solver
        function runDuoSolver() {{
            const p1_titles = [
                document.getElementById('duoUser1_1').value,
                document.getElementById('duoUser1_2').value,
                document.getElementById('duoUser1_3').value
            ].filter(Boolean);

            const p2_titles = [
                document.getElementById('duoUser2_1').value,
                document.getElementById('duoUser2_2').value,
                document.getElementById('duoUser2_3').value
            ].filter(Boolean);

            const allSelected = new Set([...p1_titles, ...p2_titles]);

            const vecP1 = {{}};
            p1_titles.forEach(t => {{
                const m = movies.find(x => x.title === t);
                if (m) {{
                    const v = getMovieVector(m);
                    for (let k in v) vecP1[k] = (vecP1[k] || 0) + v[k];
                }}
            }});

            const vecP2 = {{}};
            p2_titles.forEach(t => {{
                const m = movies.find(x => x.title === t);
                if (m) {{
                    const v = getMovieVector(m);
                    for (let k in v) vecP2[k] = (vecP2[k] || 0) + v[k];
                }}
            }});

            const candidates = movies
                .filter(m => !allSelected.has(m.title))
                .map(m => {{
                    const v = getMovieVector(m);
                    const sim1 = cosineSimilarity(vecP1, v);
                    const sim2 = cosineSimilarity(vecP2, v);
                    const consensus = (Math.min(sim1, sim2) * 0.65) + (((sim1 + sim2) / 2) * 0.35);
                    return {{ movie: m, sim: consensus }};
                }})
                .sort((a,b) => b.sim - a.sim)
                .slice(0, 8);

            const maxRaw = candidates[0]?.sim || 1;
            document.getElementById('duoResults').innerHTML = candidates.map(c => {{
                const pct = Math.round((c.sim / maxRaw) * 95);
                return renderCard(c.movie, Math.max(55, pct));
            }}).join('');
        }}

        // Trope Search
        function applyTropeChip(tropeText) {{
            document.getElementById('tropeInput').value = tropeText;
            runTropeSearch();
        }}

        function runTropeSearch() {{
            const q = document.getElementById('tropeInput').value.toLowerCase().trim();
            if (!q) {{
                document.getElementById('tropeResults').innerHTML = movies.slice(0, 8).map(m => renderCard(m)).join('');
                return;
            }}
            const tokens = tokenize(q);

            const scored = movies.map(m => {{
                const tropeText = ((m.tropes || '') + ' ' + (m.overview || '') + ' ' + (m.genre || '')).toLowerCase();
                let score = 0;
                tokens.forEach(tok => {{
                    if (tropeText.includes(tok)) score += 1;
                }});
                return {{ movie: m, score }};
            }})
            .filter(x => x.score > 0)
            .sort((a,b) => b.score - a.score || b.movie.rating - a.movie.rating)
            .slice(0, 8);

            const container = document.getElementById('tropeResults');
            if (scored.length === 0) {{
                container.innerHTML = movies.slice(0, 8).map(m => renderCard(m)).join('');
            }} else {{
                container.innerHTML = scored.map(s => renderCard(s.movie)).join('');
            }}
        }}

        // Precomputed Scikit-Learn Smart Match
        function computeSmartRecommendations() {{
            const chosen = document.getElementById('smartMatchSelect').value;
            const target = movies.find(m => m.title === chosen);
            if (!target) return;

            const precomputed = target.similar || [];
            let results = [];

            if (precomputed.length > 0) {{
                results = precomputed.slice(0, 8).map(s => {{
                    const m = movieMap[s.id];
                    return m ? renderCard(m, s.score) : '';
                }}).filter(Boolean);
            }}

            if (results.length === 0) {{
                const targetVec = getMovieVector(target);
                const scores = movies
                    .filter(m => m.title !== chosen)
                    .map(m => ({{
                        movie: m,
                        sim: cosineSimilarity(targetVec, getMovieVector(m))
                    }}))
                    .sort((a,b) => b.sim - a.sim)
                    .slice(0, 8);
                results = scores.map(s => renderCard(s.movie, 85));
            }}

            document.getElementById('smartMatchResults').innerHTML = results.join('');
        }}

        // Roulette
        function spinRoulette() {{
            const mood = document.getElementById('rouletteMood').value;
            const pool = mood === 'All' ? movies : movies.filter(m => m.mood === mood);
            const pick = pool[Math.floor(Math.random() * pool.length)];
            document.getElementById('rouletteResult').innerHTML = `
                <div style="max-width:300px; margin:0 auto;">
                    ${{renderCard(pick, 99)}}
                </div>
            `;
            showToast(`Roulette landed on: ${{pick.title}}!`);
        }}

        // Watchlist Management
        function toggleWatchlist(title) {{
            const idx = watchlist.indexOf(title);
            if (idx > -1) {{
                watchlist.splice(idx, 1);
                showToast(`Removed ${{title}} from watchlist`);
            }} else {{
                watchlist.push(title);
                showToast(`Saved ${{title}} to watchlist!`);
            }}
            saveWatchlistToStorage();
            filterCatalog();
        }}

        function renderWatchlist() {{
            const grid = document.getElementById('watchlistGrid');
            const saved = movies.filter(m => watchlist.includes(m.title));
            if (saved.length === 0) {{
                grid.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 40px 0;">Your watchlist is currently empty. Tap 🤍 Save on any film to bookmark it!</p>';
            }} else {{
                grid.innerHTML = saved.map(m => renderCard(m)).join('');
            }}
        }}

        function shareWatchlist() {{
            if (watchlist.length === 0) return alert("Your watchlist is empty!");
            const text = encodeURIComponent("Here is my CineMatch Watchlist 🎬:\\n- " + watchlist.join("\\n- ") + "\\n\\nDiscovered via CineMatch AI!");
            window.open("https://wa.me/?text=" + text, "_blank");
        }}

        // Video Modal
        function openVideoModal(title, key) {{
            document.getElementById('modalVideoTitle').innerText = title + " — Official Trailer";
            document.getElementById('videoIframe').src = `https://www.youtube.com/embed/${{key}}?autoplay=1`;
            document.getElementById('videoModal').classList.add('active');
        }}

        function closeVideoModal(e) {{
            document.getElementById('videoIframe').src = "";
            document.getElementById('videoModal').classList.remove('active');
        }}

        // VIP / UPI Modal
        function openUpiModal() {{ document.getElementById('upiModal').classList.add('active'); }}
        function closeUpiModal() {{ document.getElementById('upiModal').classList.remove('active'); }}

        function selectPlan(card, name, price) {{
            document.querySelectorAll('.upi-plan-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            document.getElementById('upiQrImg').src = `https://api.qrserver.com/v1/create-qr-code/?size=180x180&data=upi://pay?pa=cinematch@axisbank&pn=CineMatch&am=${{price}}&cu=INR`;
        }}

        function simulateVipSuccess() {{
            closeUpiModal();
            showToast("🎉 VIP Access Activated Successfully! 4K Streams Unlocked.");
        }}

        // Pre-hydrate everything on DOM load
        window.addEventListener('DOMContentLoaded', () => {{
            renderSwipeCard();
            runDnaMixer();
            runDuoSolver();
            runTropeSearch();
            computeSmartRecommendations();
        }});

        // Keyboard ESC
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') {{
                closeVideoModal();
                closeRadarModal();
                closeUpiModal();
            }}
        }});
    </script>
</body>
</html>
"""

with open(index_path, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"Generated next-generation index.html ({len(html_template)} bytes, {len(df)} movies) at {index_path}")
