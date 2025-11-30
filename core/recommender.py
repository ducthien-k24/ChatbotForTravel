# core/recommender.py
from __future__ import annotations
from typing import List, Dict, Optional
from pathlib import Path
import re
import random
import unidecode
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OUTDOOR_HINTS = {"park", "garden", "viewpoint", "beach", "outdoor", "lake"}
_CANON_SET = {"food", "cafe", "entertainment", "attraction", "shopping", "unknown"}


# --------------------------
# Normalizers & small utils
# --------------------------
def _canonicalize_category(cat: str) -> str:
    c = (cat or "").strip().lower()
    if c in _CANON_SET:
        return c
    if any(k in c for k in ["restaurant", "eatery", "food"]): return "food"
    if "cafe" in c or "coffee" in c: return "cafe"
    if any(k in c for k in ["entertainment", "theater", "cinema", "amusement", "game", "arcade"]): return "entertainment"
    if any(k in c for k in ["attraction", "museum", "landmark", "park", "sightseeing", "temple", "church", "beach"]): return "attraction"
    if any(k in c for k in ["shopping", "mall", "market", "boutique", "store"]): return "shopping"
    return "unknown"

def _parse_coord(val):
    """Làm sạch toạ độ kiểu '10.791.858.651.304.300' -> 10.7918586513043."""
    if val is None: return None
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip()
    if not s: return None
    out, used_decimal, i = [], False, 0
    if s[0] in "+-": out.append(s[0]); i = 1
    for ch in s[i:]:
        if ch.isdigit(): out.append(ch)
        elif ch in ".,": 
            if not used_decimal: out.append("."); used_decimal = True
    if not out or "".join(out) in {"+", "-"}: return None
    try:
        return float("".join(out))
    except Exception:
        m = re.search(r"[-+]?\d+(?:[.,]\d+)?", s)
        return float(m.group(0).replace(",", ".")) if m else None

def _split_tags(s: str) -> List[str]:
    s = (s or "").lower()
    for ch in [";", "|"]:
        s = s.replace(ch, ",")
    return [t.strip() for t in s.split(",") if t.strip()]

def _city_norm(s: str) -> str:
    return unidecode.unidecode((s or "").lower())

def _soft_tag_filter(df: pd.DataFrame, tag_filter: Optional[List[str]]) -> pd.DataFrame:
    """Khớp mềm: bất kỳ tag_filter xuất hiện trong tags_list hoặc substring của 'tag'."""
    if df is None or df.empty or not tag_filter:
        return df
    wants = [t.lower().strip() for t in tag_filter if isinstance(t, str) and t.strip()]
    if not wants:
        return df
    lst_mask = df["tags_list"].apply(lambda ts: any(w in ts for w in wants))
    subs_mask = df["tag"].str.contains("|".join(map(re.escape, wants)), case=False, na=False)
    out = df[lst_mask | subs_mask]
    return out if not out.empty else df  # nếu lọc ra rỗng → trả lại df để tránh crash

def _tfidf_cosine(texts: pd.Series, query: str) -> pd.Series:
    if texts.str.strip().eq("").all():
        return pd.Series([0.0] * len(texts), index=texts.index)
    vec = TfidfVectorizer(stop_words=None)
    try:
        M = vec.fit_transform(texts)
        if M.shape[1] == 0:
            return pd.Series([0.0] * len(texts), index=texts.index)
        q = vec.transform([query or ""])
        sims = cosine_similarity(q, M).flatten()
        return pd.Series(sims, index=texts.index)
    except Exception:
        return pd.Series([0.0] * len(texts), index=texts.index)

def _weather_penalty_row(row: pd.Series, weather_desc: str) -> float:
    if not weather_desc:
        return 1.0
    w = weather_desc.lower()
    if not any(k in w for k in ["mưa", "rain", "storm"]):
        return 1.0
    cat = str(row.get("category", "")).lower()
    tags = set(row.get("tags_list", []))
    if cat == "attraction" or (tags & OUTDOOR_HINTS):
        return 0.6
    return 1.0


# --------------------------
# Data loading
# --------------------------
def load_category_data(city: str, category: str, base_dir: str = "data") -> pd.DataFrame:
    """Tải dữ liệu offline tương ứng với category người dùng chọn."""
    mapping = {
        "food": "pois_hcm_food.csv",
        "cafe": "pois_hcm_cafe.csv",
        "entertainment": "pois_hcm_entertainment.csv",
        "shopping": "pois_hcm_shopping.csv",
        "attraction": "pois_hcm_attraction.csv",
    }
    cat = (category or "unknown").lower()
    fn = mapping.get(cat)
    p = Path(base_dir) / fn if fn else None
    if not p or not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p)
    # gắn nhãn & metadata
    df["category"] = cat
    df["city"] = city
    if "tag" not in df.columns:
        df["tag"] = ""
    df["tag"] = df["tag"].fillna("").astype(str)
    df["tags_list"] = df["tag"].apply(_split_tags)

    # clean cost
    if "avg_cost" in df.columns:
        df["avg_cost"] = (
            df["avg_cost"].astype(str).str.replace("[^0-9.,]", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        df["avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce")

    # name
    if "name" not in df.columns:
        df["name"] = df.get("place_id", df.index).astype(str)
    else:
        df["name"] = df["name"].astype(str)
        empty = df["name"].str.strip().eq("")
        if "place_id" in df.columns:
            df.loc[empty, "name"] = df["place_id"].astype(str)
            empty = df["name"].str.strip().eq("")
        if "address" in df.columns:
            df.loc[empty, "name"] = df["address"].fillna("").astype(str)

    # lat/lon
    if "lat" in df.columns: df["lat"] = df["lat"].apply(_parse_coord)
    if "lon" in df.columns: df["lon"] = df["lon"].apply(_parse_coord)

    # city_norm
    df["city_norm"] = _city_norm(city)  # vì ta đã gán df["city"]=city
    # rating_num (nếu có)
    if "rating" in df.columns:
        def _to_float(x):
            try: return float(x)
            except Exception: return None
        df["rating_num"] = df["rating"].apply(_to_float)
    else:
        df["rating_num"] = None

    return df


# --------------------------
# Public API
# --------------------------
def recommend_pois(
    city: str,
    poi_df: Optional[pd.DataFrame] = None,
    category: str = "food",
    user_query: str = "",
    taste_tags: Optional[List[str]] = None,
    activity_tags: Optional[List[str]] = None,
    budget_per_day: int = 500_000,
    walk_tolerance_km: float = 5.0,   # hiện chưa dùng, để mở rộng
    weather_desc: str = "",
    tag_filter: Optional[List[str]] = None,
    top_k: int = 30,
) -> List[Dict]:
    """
    Gợi ý địa điểm cho 1 category:
      - An toàn khi rỗng (không bao giờ .iloc[0]).
      - Lọc tag mềm.
      - Phạt thời tiết nhẹ cho attraction/outdoor khi mưa.
      - Xếp hạng: TF-IDF theo (name + tag + description) + ngân sách + thời tiết + rating/ảnh.
    """
    # 1) nguồn dữ liệu
    if poi_df is not None:
        df = poi_df.copy()
        # ép category nếu gọi từ ngoài truyền tổng hợp
        df["category"] = df.get("category", "unknown").astype(str).str.lower().map(_canonicalize_category)
        df = df[df["category"].eq(category.lower())]
        if "tag" not in df.columns:
            df["tag"] = ""
        df["tags_list"] = df["tag"].fillna("").astype(str).apply(_split_tags)
        if "lat" in df.columns: df["lat"] = df["lat"].apply(_parse_coord)
        if "lon" in df.columns: df["lon"] = df["lon"].apply(_parse_coord)
        df["city"] = city
        df["city_norm"] = _city_norm(city)
    else:
        df = load_category_data(city, category)

    if df is None or df.empty:
        return []

    # 2) lọc theo tag (mềm)
    df = _soft_tag_filter(df, tag_filter)

    # 3) TF-IDF theo truy vấn
    taste_tags = taste_tags or []
    activity_tags = activity_tags or []
    query = " ".join([user_query] + taste_tags + activity_tags + [city])
    texts = df["name"].fillna("") + " " + df["tag"].fillna("") + " " + df.get("description", "")
    df["sim"] = _tfidf_cosine(texts, query)

    # 4) Ngân sách
    if "avg_cost" in df.columns and df["avg_cost"].notna().any():
        diff = (df["avg_cost"] - budget_per_day / 3).abs()
        df["budget_score"] = 1 - diff / max(diff.max(), 1)
    else:
        df["budget_score"] = 0.5

    # 5) Thời tiết
    df["weather_score"] = df.apply(lambda r: _weather_penalty_row(r, weather_desc), axis=1)

    # 6) Điểm tổng
    # thêm tín hiệu rating & có ảnh
    if "image_url1" in df.columns:
        has_img = df["image_url1"].astype(str).str.startswith(("http://", "https://")).astype(int)
    else:
        has_img = 0

    rating = df.get("rating_num", pd.Series([0] * len(df), index=df.index)).fillna(0)

    df["final"] = (
        0.55 * df["sim"]
        + 0.20 * df["budget_score"]
        + 0.15 * df["weather_score"]
        + 0.10 * rating
        + 0.05 * has_img
    )

    # boost nhỏ nếu category là food/cafe và có overlap với taste_tags
    if category.lower() in {"food", "cafe"} and taste_tags:
        wants = set(t.lower() for t in taste_tags)
        df["has_taste"] = df["tags_list"].apply(lambda ts: any(t in wants for t in ts))
        df.loc[df["has_taste"], "final"] += 0.05

    # 7) sort + lấy top_k
    df = df.sample(frac=1.0, random_state=random.randint(1, 9999))  # shuffle nhẹ
    df = df.sort_values("final", ascending=False)

    cols = [c for c in [
        "name", "category", "tag", "city", "avg_cost", "description",
        "lat", "lon", "image_url1", "image_url2", "address", "rating", "reviews", "final"
    ] if c in df.columns]

    return df[cols].head(min(top_k, 50)).to_dict(orient="records")
