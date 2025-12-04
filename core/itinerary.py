import enum
from typing import Dict, List
import random
import math
import re
import pandas as pd

from .route_optimizer import pairwise_distance_matrix, mst_order, total_distance
from .geo_graph import road_graph_for_city, shortest_distance_km
from .weather import get_daily_forecast
from core.llm_composer import generate_day_summary
from core.ai_planner import analyze_user_preferences


# ============================================================
# Canonicalization & utilities
# ============================================================

_CANON_SET = {"food", "cafe", "entertainment", "attraction", "shopping", "unknown"}


def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\s/_\-\|]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", "-", s).strip("-")


def _grid(val: float, size: float = 0.001) -> float:
    try:
        return round(float(val) / size) * size
    except Exception:
        return 0.0


def _canonicalize_category(cat: str) -> str:
    c = (cat or "").strip().lower()
    if c in _CANON_SET:
        return c
    if any(k in c for k in ["restaurant", "eatery", "food"]):
        return "food"
    if "cafe" in c or "coffee" in c:
        return "cafe"
    if any(k in c for k in ["entertainment", "theater", "cinema", "amusement", "game", "arcade"]):
        return "entertainment"
    if any(k in c for k in ["attraction", "museum", "landmark", "park", "sightseeing", "temple", "church"]):
        return "attraction"
    if any(k in c for k in ["shopping", "mall", "market", "boutique", "store"]):
        return "shopping"
    return "unknown"


# ---------- sanitize lat/lon strings ----------
def _parse_coord(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip()
    if not s:
        return None

    out = []
    used_decimal = False
    i = 0
    if s[0] in "+-":
        out.append(s[0]); i = 1

    for ch in s[i:]:
        if ch.isdigit():
            out.append(ch)
        elif ch in ".,": 
            if not used_decimal:
                out.append(".")
                used_decimal = True

    if not out or "".join(out) in {"+", "-"}:
        return None

    try:
        return float("".join(out))
    except Exception:
        m = re.search(r"[-+]?\d+(?:[.,]\d+)?", s)
        if m:
            return float(m.group(0).replace(",", "."))
        return None


def _sanitize_latlon(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "lat" in df.columns:
        df["lat"] = df["lat"].apply(_parse_coord)
    if "lon" in df.columns:
        df["lon"] = df["lon"].apply(_parse_coord)
    return df


def _has_coords(row) -> bool:
    try:
        lat = row.get("lat")
        lon = row.get("lon")
        if lat is None or lon is None:
            return False
        float(lat); float(lon)
        return True
    except Exception:
        return False


def _df_valid_coords(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    mask = df.apply(_has_coords, axis=1)
    return df[mask]


def _normalize_tags_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    tag_col = None
    for c in ["tag", "tags", "keywords", "Labels", "labels"]:
        if c in df.columns:
            tag_col = c
            break

    df["tag"] = df[tag_col].fillna("").astype(str) if tag_col else ""

    def _split_tags(s: str):
        s = s.lower().replace(";", ",").replace("|", ",")
        parts = [t.strip() for t in s.split(",")]
        return [t for t in parts if t]

    df["tags_list"] = df["tag"].apply(_split_tags)
    return df


def _ensure_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "name" not in df.columns:
        if "place_id" in df.columns:
            df["name"] = df["place_id"].astype(str)
        elif "address" in df.columns:
            df["name"] = df["address"].fillna("").astype(str)
        else:
            df["name"] = df.index.map(lambda i: f"poi_{i}")
    else:
        df["name"] = df["name"].astype(str)
        df.loc[df["name"].str.strip().eq(""), "name"] = df.index.map(lambda i: f"poi_{i}")

    df["category"] = df.get("category", "unknown").astype(str).map(_canonicalize_category)

    def _build_key(row):
        nm = _slug(row["name"])
        latg = _grid(row.get("lat", 0) or 0.0)
        long = _grid(row.get("lon", 0) or 0.0)
        pid = str(row.get("place_id", "") or "")
        base = f"{nm}|{latg:.3f},{long:.3f}"
        return f"{base}|{pid}" if pid else base

    df["unique_key"] = df.apply(_build_key, axis=1)
    return df


def _filter_by_tags(df: pd.DataFrame, prefs: List[str]) -> pd.DataFrame:
    if not prefs:
        return df
    prefs = [p.lower().strip() for p in prefs]
    if "tags_list" not in df.columns:
        df = _normalize_tags_column(df)
    mask = df["tags_list"].apply(lambda ts: any(p in ts for p in prefs))
    return df[mask]


def _sample_no_repeat(df: pd.DataFrame, used_keys: set, n: int) -> List[Dict]:
    if df is None or df.empty or n <= 0:
        return []
    df = _ensure_key_columns(df)
    pool = df[~df["unique_key"].isin(used_keys)]
    if pool.empty:
        return []
    take = pool.sample(min(n, len(pool)), random_state=random.randint(1, 9999))
    used_keys.update(take["unique_key"].tolist())
    return take.to_dict("records")


# ============================================================
# AI–DRIVEN POI SELECTION
# ============================================================

def _select_diverse_pois(
    poi_df: pd.DataFrame,
    used_keys: set,
    max_per_day: int,
    food_target: int,
    cafe_target: int,
    att_target: int,
    ent_target: int,
    shop_target: int,
    min_mappable: int = 3,
):
    df = _ensure_key_columns(_normalize_tags_column(_sanitize_latlon(poi_df.copy())))

    # category splits
    df_food = df[df["category"] == "food"]
    df_cafe = df[df["category"] == "cafe"]
    df_att  = df[df["category"] == "attraction"]
    df_ent  = df[df["category"] == "entertainment"]
    df_shop = df[df["category"] == "shopping"]
    df_nonfood = df[~df["category"].isin({"food", "cafe"})]

    v_food = _df_valid_coords(df_food)
    v_cafe = _df_valid_coords(df_cafe)
    v_att  = _df_valid_coords(df_att)
    v_ent  = _df_valid_coords(df_ent)
    v_shop = _df_valid_coords(df_shop)
    v_nonf = _df_valid_coords(df_nonfood)
    v_all  = _df_valid_coords(df)

    chosen: List[Dict] = []

    def add_from(pref, fall, n):
        if n <= 0:
            return
        need = n
        got = _sample_no_repeat(pref, used_keys, need)
        chosen.extend(got)
        need -= len(got)
        if need > 0:
            chosen.extend(_sample_no_repeat(fall, used_keys, need))

    # apply AI targets
    add_from(v_food, df_food, food_target)
    add_from(v_cafe, df_cafe, cafe_target)
    add_from(v_att, df_att, att_target)
    add_from(v_ent, df_ent, ent_target)
    add_from(v_shop, df_shop, shop_target)

    # fill remaining
    remaining = max_per_day - len(chosen)
    if remaining > 0:
        add_from(v_nonf, df_nonfood, remaining)

    # ensure mappable
    def count_map(lst):
        return sum(1 for p in lst if _has_coords(p))

    if count_map(chosen) < min_mappable:
        add_from(v_all, df, min_mappable - count_map(chosen))

    return chosen[:max_per_day], used_keys


# ============================================================
# PUBLIC API
# ============================================================

def build_itinerary(params: Dict, poi_df, weather_now: Dict):

    city = params["city"]
    days = int(params.get("days", 2))
    max_per_day = int(params.get("max_poi_per_day", 6))

    # AI planner
    ai_plan = analyze_user_preferences(params)
    dist = ai_plan["distribution"]

    food_target = dist.get("food", 2)
    cafe_target = dist.get("cafe", 0)
    att_target = dist.get("attraction", 0)
    ent_target = dist.get("entertainment", 0)
    shop_target = dist.get("shopping", 0)

    daily_forecast = get_daily_forecast(city, days)

    # prepare dataframe
    base_df = poi_df.copy() if isinstance(poi_df, pd.DataFrame) else pd.DataFrame(poi_df)
    base_df = _sanitize_latlon(base_df)
    base_df = _normalize_tags_column(base_df)
    base_df["category"] = base_df.get("category", "unknown").astype(str).map(_canonicalize_category)
    base_df = _ensure_key_columns(base_df)

    # tag preferences
    food_tags = params.get("food_tags", [])
    ent_tags  = params.get("entertainment_tags", [])
    att_tags  = params.get("attraction_tags", [])

    df_food = _filter_by_tags(base_df[base_df["category"] == "food"], food_tags)
    df_ent  = _filter_by_tags(base_df[base_df["category"] == "entertainment"], ent_tags)
    df_att  = _filter_by_tags(base_df[base_df["category"] == "attraction"], att_tags)
    df_shop = base_df[base_df["category"] == "shopping"]

    df_all = pd.concat([df_food, df_ent, df_att, df_shop, base_df], ignore_index=True)
    df_all = _ensure_key_columns(df_all).drop_duplicates(subset="unique_key")

    used = set()
    days_pois = []

    # daily POI selection
    for _ in range(days):
        daily, used = _select_diverse_pois(
            poi_df=df_all,
            used_keys=used,
            max_per_day=max_per_day,
            food_target=food_target,
            cafe_target=cafe_target,
            att_target=att_target,
            ent_target=ent_target,
            shop_target=shop_target,
        )
        days_pois.append(daily)

    # route optimize
    out_days = []
    G = road_graph_for_city(city)

    for i, dpois in enumerate(days_pois):
        fw = daily_forecast[i] if i < len(daily_forecast) else None
        weather_line = f"{fw['summary']} — {fw['advice']}" if fw else ""

        if len(dpois) < 2:
            out_days.append({
                "title": f"Khám phá nhẹ - Ngày {i+1}",
                "pois": dpois,
                "distance": 0.0,
                "weather": weather_line
            })
            continue

        dist_mat, coords, _ = pairwise_distance_matrix(city, dpois)
        order = mst_order(dist_mat)
        ordered = [dpois[k] for k in order]
        total_km = total_distance(dist_mat, order)

        for j in range(len(ordered) - 1):
            try:
                d_km = shortest_distance_km(
                    G,
                    (ordered[j]["lat"], ordered[j]["lon"]),
                    (ordered[j+1]["lat"], ordered[j+1]["lon"])
                )
                ordered[j]["next_distance_km"] = round(d_km, 2)
            except Exception:
                ordered[j]["next_distance_km"] = "?"

        out_days.append({
            "title": f"Khám phá ngày {i+1}",
            "pois": ordered,
            "distance": round(total_km, 2),
            "weather": weather_line
        })

    # summaries
    for i, day in enumerate(out_days, start=1):
        try:
            day["summary"] = generate_day_summary(i, {"order": day["pois"]})
        except:
            day["summary"] = f"Day {i}: summary error"

    return out_days
