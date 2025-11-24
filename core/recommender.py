import pandas as pd
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unidecode

OUTDOOR = {"park", "garden", "viewpoint", "attraction"}
FOOD = {"restaurant", "cafe", "fast_food", "bar", "pub", "food"}


def load_category_data(city: str, category: str, base_dir="data/") -> pd.DataFrame:
    """Tải dữ liệu offline tương ứng với category người dùng chọn."""
    category = category.lower()
    mapping = {
        "food": "pois_hcm_food.csv",
        "cafe": "pois_hcm_cafe.csv",
        "entertainment": "pois_hcm_entertainment.csv",
        "shopping": "pois_hcm_shopping.csv",
        "attraction": "pois_hcm_attraction.csv",
    }
    file_name = mapping.get(category)
    if not file_name:
        raise ValueError(f"Không có dữ liệu cho category: {category}")

    path = f"{base_dir}/{file_name}"
    df = pd.read_csv(path)
    df["city"] = city
    df["source_file"] = file_name
    return df


def _cosine_rank(df: pd.DataFrame, query: str) -> pd.Series:
    text = (
        df["name"].fillna("") + " " +
        df.get("tag", pd.Series([""] * len(df))) + " " +
        df.get("description", pd.Series([""] * len(df)))
    )

    if text.str.strip().eq("").all():
        return pd.Series([0.0] * len(df), index=df.index)

    vec = TfidfVectorizer(stop_words=None)
    try:
        M = vec.fit_transform(text)
        if M.shape[1] == 0:
            return pd.Series([0.0] * len(df), index=df.index)
        q = vec.transform([query or ""])
        sims = cosine_similarity(q, M).flatten()
    except Exception:
        sims = [0.0] * len(df)
    return pd.Series(sims, index=df.index)


def _weather_penalty(tag: str, weather_desc: str) -> float:
    if not weather_desc:
        return 1.0
    if any(k in weather_desc.lower() for k in ["mưa", "storm", "rain"]) and tag in OUTDOOR:
        return 0.6
    return 1.0


def recommend_pois(
    city: str,
    poi_df: pd.DataFrame = None,
    category: str = "food",
    user_query: str = "",
    taste_tags: List[str] = [],
    activity_tags: List[str] = [],
    budget_per_day: int = 500000,
    walk_tolerance_km: float = 5.0,
    weather_desc: str = ""
) -> List[Dict]:
    """Gợi ý địa điểm dựa trên loại file CSV tương ứng"""
    df = load_category_data(city, category)

    # Normalize tên thành phố
    df["city_norm"] = df["city"].apply(lambda x: unidecode.unidecode(str(x).lower()))
    if "ho chi minh" not in df["city_norm"].iloc[0]:
        df = df[df["city_norm"].str.contains("ho chi minh")]

    # Cosine similarity cho truy vấn
    query = " ".join([user_query] + taste_tags + activity_tags + [city])
    df["sim"] = _cosine_rank(df, query)

    # Ngân sách
    if "avg_cost" in df.columns:
        diff = (df["avg_cost"] - budget_per_day/3).abs()
        df["budget_score"] = 1 - diff / max(diff.max(), 1)
    else:
        df["budget_score"] = 0.5


    # Thời tiết
    df["weather_score"] = df["tag"].apply(lambda c: _weather_penalty(str(c), weather_desc))
    df["final"] = 0.55 * df["sim"] + 0.2 * df["budget_score"] + 0.25 * df["weather_score"]

    if any(t in ["Vietnamese", "Japanese", "Italian", "Cafe", "Seafood", "Vegetarian"] for t in taste_tags):
        df.loc[df["tag"].isin(FOOD), "final"] += 0.05

    df = df.sort_values("final", ascending=False)
    cols = [c for c in ["name", "tag", "city", "avg_cost", "description", "lat", "lon",
                        "image_url1", "image_url2", "address", "rating", "reviews", "final"] if c in df.columns]
    return df[cols].head(12).to_dict(orient="records")
