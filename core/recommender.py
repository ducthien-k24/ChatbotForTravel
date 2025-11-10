import pandas as pd
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Trọng số theo thời tiết & cá nhân hoá (đơn giản)
OUTDOOR = {"park","garden","viewpoint","attraction"}
FOOD = {"restaurant","cafe","fast_food","bar","pub"}

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
    df = poi_df.copy()
    # lọc city (cache theo city rồi, nhưng vẫn giữ phòng hờ)
    df = df[df["city"].str.contains(city.split(",")[0], case=False, na=False)]

    # lọc theo sở thích “activity”
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
    return df[cols].head(12).to_dict(orient="records")
