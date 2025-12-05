# core/datasource.py
"""
Adapter dữ liệu cho TravelGPT+
- Cho phép lấy POI từ CSV (offline) hoặc từ API (JSON server).
- Đầu ra luôn là pandas.DataFrame với schema chuẩn mà app đang dùng.
Sử dụng qua ENV:
  DATA_PROVIDER = "csv" | "api"         (mặc định: "csv")
  API_BASE_URL  = "https://.../api"     (bỏ dấu / ở cuối)
  API_TOKEN     = "..."                  (tuỳ chọn)
"""

from typing import Dict, List, Optional
from pathlib import Path
import os
import time  # (giữ nếu muốn logging/throttle sau này)

import pandas as pd
import requests
import streamlit as st

# ====== Config qua ENV ======
API_BASE_URL_DEFAULT = "http://localhost:3000/api"
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "api").lower()   # "csv" | "api"
API_BASE_URL  = os.getenv("API_BASE_URL", API_BASE_URL_DEFAULT).rstrip("/")   # vd: https://asking-.../api
API_TOKEN     = os.getenv("API_TOKEN", "")                  # nếu server yêu cầu

# CSV filenames (giống dự án hiện tại)
CSV_MAP = {
    "food": "pois_hcm_food.csv",
    "cafe": "pois_hcm_cafe.csv",
    "entertainment": "pois_hcm_entertainment.csv",
    "shopping": "pois_hcm_shopping.csv",
    "attraction": "pois_hcm_attraction.csv",
}

# Schema chuẩn app đang dùng
CANON_COLUMNS = [
    "name","category","tag","description",
    "lat","lon",
    "avg_cost","rating","reviews",
    "address","image_url1","image_url2",
    "place_id","city",
]

# ------------ helpers ------------
def _num(x, fn=float):
    try:
        return fn(x)
    except Exception:
        return None


def _headers():
    h = {"Accept": "application/json"}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Đảm bảo đủ cột & kiểu dữ liệu nhẹ nhàng."""
    if df is None or getattr(df, "empty", True):
        df = pd.DataFrame(columns=CANON_COLUMNS)
    for c in CANON_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df["tag"] = df["tag"].fillna("").astype(str)
    return df[CANON_COLUMNS]

# ------------ CSV backend ------------
def _load_category_csv(category: str, base_dir="data") -> pd.DataFrame:
    fname = CSV_MAP.get(category)
    if not fname:
        return pd.DataFrame(columns=CANON_COLUMNS)
    p = Path(base_dir) / fname
    if not p.exists():
        return pd.DataFrame(columns=CANON_COLUMNS)
    df = pd.read_csv(p)
    if "category" not in df.columns:
        df["category"] = category
    if "tag" not in df.columns:
        df["tag"] = ""
    return _ensure_cols(df)

# ------------ API backend ------------
def _api_get(path: str, params: Optional[Dict]=None, timeout: int=10):
    if not API_BASE_URL:
        raise RuntimeError("API_BASE_URL is empty (set ENV API_BASE_URL)")
    url = f"{API_BASE_URL}/{path.lstrip('/')}"
    r = requests.get(url, params=params, headers=_headers(), timeout=timeout)
    r.raise_for_status()
    return r.json()


def _fetch_images(location_id: int) -> List[str]:
    """GET /locations/{id}/images → lấy tối đa 2 ảnh (nếu API có)."""
    try:
        data = _api_get(f"locations/{location_id}/images")
        items = data if isinstance(data, list) else data.get("data") or data.get("items") or []
        # ưu tiên theo sort_order nếu có
        items = sorted(items, key=lambda x: (x.get("sort_order") is None, x.get("sort_order", 9999)))
        urls = [i.get("url") for i in items if i.get("url")]
        return urls[:2]
    except Exception:
        return []


def _normalize_location(raw: Dict, category_hint: Optional[str], city: str) -> Dict:
    """
    Map JSON server -> schema chuẩn của app.
    - category: ưu tiên raw['category'], nếu rỗng dùng category_hint (nếu có).
    - tag: dùng 'type' (vd: active, fun, nature, mystery) để app lọc mềm.
    """
    imgs = _fetch_images(int(raw.get("id"))) if raw.get("id") is not None else []
    img1 = (imgs[0] if imgs else None) or raw.get("image_url") or ""
    img2 = (imgs[1] if len(imgs) > 1 else None) or ""

    category = (raw.get("category") or category_hint or "unknown").strip().lower()

    return {
        "name":       raw.get("name") or "",
        "category":   category,
        "tag":        (raw.get("type") or ""),
        "description":raw.get("description") or "",
        "lat":        _num(raw.get("latitude"), float),
        "lon":        _num(raw.get("longitude"), float),
        "avg_cost":   _num(raw.get("price"), float),
        "rating":     _num(raw.get("rating"), float),
        "reviews":    _num(raw.get("review_count"), int),
        "address":    raw.get("address") or "",
        "image_url1": img1,
        "image_url2": img2,
        "place_id":   raw.get("id"),
        "city":       city,
    }


def _load_category_api(city: str, category: str) -> pd.DataFrame:
    """Nếu server filter được category thì dùng params={"category": category}."""
    try:
        # Nếu server hỗ trợ: data = _api_get("locations", params={"category": category})
        data = _api_get("locations")
        items = data.get("data") if isinstance(data, dict) else data
        if not isinstance(items, list):
            items = []
    except Exception:
        items = []

    if category:
        items = [it for it in items if (it.get("category") or "").strip().lower() == category]

    norm = [_normalize_location(it, category, city) for it in items]
    return _ensure_cols(pd.DataFrame(norm))

# ------------ Public API ------------
def load_category_df(city: str, category: str, fallback_to_csv=True) -> pd.DataFrame:
    """Trả DataFrame đúng schema cho 1 category; ưu tiên API nếu DATA_PROVIDER=api."""
    category = (category or "").lower()
    if DATA_PROVIDER == "api":
        df = _load_category_api(city, category)
        if not df.empty:
            return df
        if not fallback_to_csv:
            return pd.DataFrame(columns=CANON_COLUMNS)

    df = _load_category_csv(category)
    if not df.empty:
        df["city"] = city
    return df

@st.cache_data
def load_all_categories(city: str, categories: List[str]) -> pd.DataFrame:
    """Gom nhiều category vào một DataFrame, loại trùng cơ bản."""
    frames: List[pd.DataFrame] = []

    for c in categories:
        dfc = load_category_df(city, c)
        if dfc is None or dfc.empty:
            continue
        # bỏ các dòng toàn-NA để tránh 'all-NA entries' khi concat
        dfc = dfc.dropna(how="all")
        # đảm bảo đúng schema/cột
        dfc = _ensure_cols(dfc)
        frames.append(dfc)

    if not frames:
        return pd.DataFrame(columns=CANON_COLUMNS)

    # concat an toàn (không còn frame rỗng/all-NA)
    df = pd.concat(frames, ignore_index=True, sort=False)

    # loại trùng nhẹ theo tên + tọa độ
    df.drop_duplicates(subset=["name", "lat", "lon"], inplace=True, keep="first")
    return df

