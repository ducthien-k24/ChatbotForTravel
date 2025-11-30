import json
from pathlib import Path
from typing import Optional
import pandas as pd
import requests

from .config import (
    API_LOCATIONS_URL,
    API_LOCATION_IMAGES_URL,
    API_TIMEOUT_SEC,
    CACHE_DIR,
)

SAFE_COLUMNS = [
    "id", "name", "category", "type", "price", "description",
    "latitude", "longtitude", "address", "opening_hours", "closing_hours",
    "image_url", "rating", "review_count", "city"
]


def _cache_path(city: str, category: Optional[str] = None) -> Path:
    key = city.strip().lower().replace(" ", "_")
    if category:
        cat = category.strip().lower().replace(" ", "_")
        return CACHE_DIR / f"{key}_{cat}.json"
    return CACHE_DIR / f"{key}_all.json"


def _normalize_locations(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    # Ensure expected columns exist
    for col in SAFE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Map to app schema
    df_out = pd.DataFrame({
        "name": df["name"],
        "category": df.get("category", None),
        "tag": df.get("type", None),
        "description": df.get("description", None),
        "lat": df.get("latitude", None),
        "lon": df.get("longtitude", None),
        "avg_cost": df.get("price", None),
        "rating": df.get("rating", None),
        "reviews": df.get("review_count", None),
        "address": df.get("address", None),
        "opening_hours": df.get("opening_hours", None),
        "image_url1": df.get("image_url", None),
        "image_url2": None,
        "city": df.get("city", None),
        "place_id": df.get("id", None),
    })
    # Drop rows without name
    df_out["name"] = df_out["name"].fillna("").astype(str)
    df_out = df_out[df_out["name"].str.strip() != ""]
    # De-duplicate by name
    df_out = df_out.drop_duplicates(subset=["name"])\
        .reset_index(drop=True)
    return df_out


def fetch_pois_city_api(city: str, use_cache: bool = True) -> pd.DataFrame:
    cache_file = _cache_path(city)
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                rows = json.load(f)
            return _normalize_locations(rows)
        except Exception:
            pass

    # API query by city via q parameter
    params = {"q": city, "limit": 1000}
    resp = requests.get(API_LOCATIONS_URL, params=params, timeout=API_TIMEOUT_SEC)
    resp.raise_for_status()
    data = resp.json()
    rows = data["data"] if isinstance(data, dict) and "data" in data else data

    # cache raw JSON rows
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
    except Exception:
        pass

    return _normalize_locations(rows)


def fetch_pois_category_api(city: str, category: str, use_cache: bool = True) -> pd.DataFrame:
    cache_file = _cache_path(city, category)
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                rows = json.load(f)
            return _normalize_locations(rows)
        except Exception:
            pass

    params = {"category": category, "q": city, "limit": 1000}
    resp = requests.get(API_LOCATIONS_URL, params=params, timeout=API_TIMEOUT_SEC)
    resp.raise_for_status()
    data = resp.json()
    rows = data["data"] if isinstance(data, dict) and "data" in data else data

    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
    except Exception:
        pass

    return _normalize_locations(rows)
