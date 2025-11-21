import pandas as pd
import json
import os
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Trọng số theo thời tiết & cá nhân hoá (đơn giản)
OUTDOOR = {"park","garden","viewpoint","attraction"}
FOOD = {"restaurant","cafe","fast_food","bar","pub"}

def _load_featured_pois(city: str) -> List[Dict]:
    """Load featured POIs with guaranteed images"""
    featured_path = "data/featured_pois.json"
    if os.path.exists(featured_path):
        try:
            with open(featured_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Normalize city name
                city_key = city.strip()
                for key in data:
                    if key.lower() in city.lower() or city.lower() in key.lower():
                        return data[key]
        except Exception as e:
            print(f"⚠️ Error loading featured POIs: {e}")
    return []

def _cosine_rank(df: pd.DataFrame, query: str) -> pd.Series:
    text = (df["name"].fillna("") + " " + df["category"].fillna("") + " " + df["description"].fillna(""))
    vec = TfidfVectorizer()
    M = vec.fit_transform(text)
    q = vec.transform([query or ""])
    return pd.Series(cosine_similarity(q, M).flatten(), index=df.index)

def _weather_penalty(category: str, weather_desc: str) -> float:
    """trả về [0..1], 1 = tốt; mưa làm giảm điểm các outdoor."""
    if not weather_desc: return 1.0
    w = weather_desc.lower()
    if any(k in w for k in ["mưa","storm","rain"]):
        if category in OUTDOOR: return 0.6
    return 1.0

def recommend_pois(
    city: str,
    poi_df: pd.DataFrame,
    user_query: str,
    taste_tags: List[str],
    activity_tags: List[str],
    budget_per_day: int,
    walk_tolerance_km: float,
    weather_desc: str = ""
) -> List[Dict]:
    # Prioritize featured POIs with guaranteed images
    featured = _load_featured_pois(city)
    featured_results = []
    if featured:
        for poi in featured[:8]:  # Take top 8 featured
            poi["final"] = 0.95  # High score for featured
            if "avg_cost" not in poi:
                poi["avg_cost"] = budget_per_day // 3
            featured_results.append(poi)
    
    df = poi_df.copy()
    # lọc city (cache theo city rồi, nhưng vẫn giữ phòng hờ)
    df = df[df["city"].str.contains(city.split(",")[0], case=False, na=False)]

    # Exclude featured POIs from regular recommendations
    if featured_results:
        featured_names = [p["name"] for p in featured_results]
        df = df[~df["name"].isin(featured_names)]

    # lọc theo sở thích "activity"
    if activity_tags:
        df = df[df["category"].isin(activity_tags + list(FOOD) + list(OUTDOOR)) | df["category"].str.contains("|".join(activity_tags), case=False, na=False)]

    # cosine cho truy vấn + taste & activity
    query = " ".join([user_query] + taste_tags + activity_tags + [city])
    df["sim"] = _cosine_rank(df, query)

    # gần ngân sách (điểm 1 nếu gần budget, giảm dần khi lệch)
    if "avg_cost" in df.columns:
        diff = (df["avg_cost"] - budget_per_day/3).abs()  # chi phí/điểm tham quan ~ 1/3 ngân sách ngày
        df["budget_score"] = 1 - diff / max(diff.max(), 1)
    else:
        df["budget_score"] = 0.5

    # thời tiết
    df["weather_score"] = df["category"].apply(lambda c: _weather_penalty(str(c), weather_desc))

    # tổng điểm
    df["final"] = 0.55*df["sim"] + 0.2*df["budget_score"] + 0.25*df["weather_score"]

    # ưu tiên food nếu taste có food
    if any(t in ["Vietnamese","Japanese","Italian","Cafe","Seafood","Vegetarian"] for t in taste_tags):
        df.loc[df["category"].isin(FOOD), "final"] += 0.05

    df = df.sort_values("final", ascending=False)
    cols = ["name","category","city","avg_cost","description","lat","lon","final"]
    regular_results = df[cols].head(12 - len(featured_results)).to_dict(orient="records")
    
    # Combine featured + regular
    return featured_results + regular_results
