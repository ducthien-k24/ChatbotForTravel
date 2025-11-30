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
from core.datasource import load_category_df

def load_category_data(city: str, category: str, base_dir: str = "data") -> pd.DataFrame:  # adapter wrapper
    """Delegate to unified datasource (API or CSV)."""
    df = load_category_df(city, category)
    if df is None or df.empty:
        return pd.DataFrame()
    # Recompute tags_list & rating_num for scoring consistency
    if "tag" not in df.columns:
        df["tag"] = ""
    df["tag"] = df["tag"].fillna("").astype(str)
    df["tags_list"] = df["tag"].apply(_split_tags)
    if "rating" in df.columns:
        def _to_float(x):
            try: return float(x)
            except Exception: return None
        df["rating_num"] = df["rating"].apply(_to_float)
    else:
        df["rating_num"] = None
    # city_norm
    df["city"] = city
    df["city_norm"] = _city_norm(city)
    # sanitize lat/lon
    if "lat" in df.columns: df["lat"] = df["lat"].apply(_parse_coord)
    if "lon" in df.columns: df["lon"] = df["lon"].apply(_parse_coord)
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
      - An toàn khi rỗng (không .iloc[0]).
      - Lọc tag mềm.
      - Phạt thời tiết nhẹ cho attraction/outdoor khi mưa.
      - Xếp hạng: TF-IDF(name+tag+description) + ngân sách + thời tiết + rating/ảnh.
    """
    # 1) Nguồn dữ liệu
    if poi_df is not None:
        df = poi_df.copy()
        # ép & canonicalize category
        df["category"] = df.get("category", "unknown").astype(str).str.lower().map(_canonicalize_category)
        df = df[df["category"].eq(category.lower())]
        # đảm bảo cột tag/tags_list tồn tại
        if "tag" not in df.columns:
            df["tag"] = ""
        df["tag"] = df["tag"].fillna("").astype(str)
        df["tags_list"] = df["tag"].apply(_split_tags)
        # tọa độ: ép về float an toàn
        if "lat" in df.columns: df["lat"] = df["lat"].apply(_parse_coord)
        if "lon" in df.columns: df["lon"] = df["lon"].apply(_parse_coord)
        # city
        df["city"] = city
        df["city_norm"] = _city_norm(city)
    else:
        df = load_category_data(city, category)

    if df is None or df.empty:
        return []

    # 2) Lọc (mềm) theo tag được chọn ở UI (không loại cứng toàn bộ dataset)
    df = _soft_tag_filter(df, tag_filter)

    if df.empty:
        return []

    # 3) TF-IDF theo truy vấn (name + tag + description)
    taste_tags = taste_tags or []
    activity_tags = activity_tags or []
    query = " ".join([user_query] + taste_tags + activity_tags + [city])

    desc_series = df["description"] if "description" in df.columns else pd.Series([""] * len(df), index=df.index)
    texts = df["name"].fillna("").astype(str) + " " + df["tag"].fillna("").astype(str) + " " + desc_series.fillna("").astype(str)
    df["sim"] = _tfidf_cosine(texts, query)

    # 4) Ngân sách (ÉP VỀ SỐ trước khi tính)
    if "avg_cost" in df.columns:
        df["avg_cost_num"] = pd.to_numeric(df["avg_cost"], errors="coerce")
        if df["avg_cost_num"].notna().any():
            target = budget_per_day / 3.0
            diff = (df["avg_cost_num"] - target).abs()
            df["budget_score"] = 1 - diff / max(diff.max(), 1.0)
        else:
            df["budget_score"] = 0.5
    else:
        df["budget_score"] = 0.5

    # 5) Thời tiết (phạt nhẹ outdoor khi mưa)
    df["weather_score"] = df.apply(lambda r: _weather_penalty_row(r, weather_desc), axis=1)

    # 6) Các tín hiệu bổ sung: rating, có ảnh
    if "rating" in df.columns:
        df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    else:
        df["rating_num"] = 0.0

    if "image_url1" in df.columns:
        df["has_img"] = df["image_url1"].astype(str).str.startswith(("http://", "https://")).astype(int)
    else:
        df["has_img"] = 0

    df["final"] = (
        0.55 * df["sim"]
        + 0.20 * df["budget_score"]
        + 0.15 * df["weather_score"]
        + 0.10 * df["rating_num"]
        + 0.05 * df["has_img"]
    )

    # 7) Boost nhẹ cho taste tag khi category là food/cafe
    if category.lower() in {"food", "cafe"} and (taste_tags or []):
        wants = set(t.lower() for t in taste_tags or [])
        df["has_taste"] = df["tags_list"].apply(lambda ts: any(t in wants for t in ts))
        df.loc[df["has_taste"], "final"] += 0.05

    # 8) sort + lấy top_k (shuffle nhẹ để tránh đồng hạng)
    df = df.sample(frac=1.0, random_state=random.randint(1, 9999)).sort_values("final", ascending=False)

    cols = [c for c in [
        "name", "category", "tag", "city", "avg_cost", "description",
        "lat", "lon", "image_url1", "image_url2", "address", "rating", "reviews", "final"
    ] if c in df.columns]

    return df[cols].head(min(top_k, 50)).to_dict(orient="records")
