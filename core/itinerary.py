from typing import Dict, List
import random
import math
import re
import pandas as pd

from .route_optimizer import pairwise_distance_matrix, mst_order, total_distance
from .geo_graph import road_graph_for_city, shortest_distance_km


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
    """Chuyển chuỗi toạ độ bẩn kiểu '10.791.858.651.304.300' -> 10.7918586513043."""
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
        elif ch in ".,":  # decimal mark
            if not used_decimal:
                out.append("."); used_decimal = True
        # skip others
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
        lat = row.get("lat"); lon = row.get("lon")
        if lat is None or lon is None:
            return False
        float(lat); float(lon)
        return True
    except Exception:
        return False

def _df_valid_coords(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    m = df.apply(_has_coords, axis=1)
    return df[m]
# ------------------------------------------------------------

def _normalize_tags_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        df = pd.DataFrame()
        df["tag"] = ""
        df["tags_list"] = [[]]
        return df
    df = df.copy()

    tag_col = None
    for c in ["tag", "tags", "keywords", "Labels", "labels"]:
        if c in df.columns:
            tag_col = c
            break

    if tag_col is None:
        df["tag"] = ""
    else:
        df["tag"] = df[tag_col].fillna("").astype(str)

    def _split_tags(s: str):
        s = s.lower()
        for ch in [";", "|"]:
            s = s.replace(ch, ",")
        parts = [t.strip() for t in s.split(",")]
        return [t for t in parts if t]

    df["tags_list"] = df["tag"].apply(_split_tags)
    return df

def _ensure_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Đảm bảo có name + unique_key để chống trùng (dùng lat/lon đã sanitize)."""
    df = df.copy()
    # name
    if "name" not in df.columns:
        if "place_id" in df.columns:
            df["name"] = df["place_id"].astype(str)
        elif "address" in df.columns:
            df["name"] = df["address"].fillna("").astype(str)
        else:
            df["name"] = df.index.map(lambda i: f"poi_{i}")
    else:
        df["name"] = df["name"].astype(str)
        if "place_id" in df.columns:
            df.loc[df["name"].str.strip().eq(""), "name"] = df["place_id"].astype(str)
        if "address" in df.columns:
            df.loc[df["name"].str.strip().eq(""), "name"] = df["address"].fillna("").astype(str)
        df.loc[df["name"].str.strip().eq(""), "name"] = df.index.map(lambda i: f"poi_{i}")

    # category
    if "category" not in df.columns:
        df["category"] = "unknown"
    df["category"] = df["category"].fillna("unknown").astype(str).map(_canonicalize_category)

    # unique_key: slug(name) + grid(lat/lon) + place_id (nếu có)
    def _build_key(row):
        nm = _slug(str(row.get("name", "")))
        pid = str(row.get("place_id", "") or "")
        latg = _grid(row.get("lat", 0) or 0.0)
        long = _grid(row.get("lon", 0) or 0.0)
        base = f"{nm}|{latg:.3f},{long:.3f}"
        return f"{base}|{pid}" if pid else base

    df["unique_key"] = df.apply(_build_key, axis=1)
    return df

def _filter_by_tags(df: pd.DataFrame, prefs: List[str]) -> pd.DataFrame:
    if df is None or df.empty or not prefs:
        return df
    prefs = [p.lower().strip() for p in prefs if isinstance(p, str) and p.strip()]
    if not prefs:
        return df
    if "tags_list" not in df.columns:
        df = _normalize_tags_column(df)
    mask = df["tags_list"].apply(lambda ts: any(p in ts for p in prefs))
    return df[mask]

def _df_cat_exact(df: pd.DataFrame, cats: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    need = {c.lower() for c in cats}
    return df[df["category"].isin(need)]

def _count_food_like(pois: List[Dict]) -> int:
    return sum(1 for p in pois if (p.get("category") or "").lower() in {"food", "cafe"})


# ============================================================
# Sampling without duplicates (by unique_key)
# ============================================================

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
# Core picking logic (per day)
# ============================================================

def _select_diverse_pois(
    poi_df: pd.DataFrame,
    used_keys: set,
    max_per_day: int = 10,
    include_shopping: bool = False,
    include_ent: bool = False,
    include_att: bool = False,
    min_food: int = 2,
    max_food_ratio_if_full: float = 0.40,  # khi chọn đủ bộ
    max_food_abs: int = 4,                  # trần tuyệt đối khi chọn đủ bộ
    basic_food_ratio_cap: float = 0.50,     # trần mềm khi KHÔNG bật att/ent
    min_mappable: int = 3,                  # tối thiểu POI có toạ độ / ngày
):
    """
    - Luôn >= min_food (food/cafe) — nếu thiếu sẽ thay thế để đủ.
    - Nếu bật attraction/entertainment → tổng (att + ent) >= 2 (min).
    - Shopping nếu bật → 1.
    - Ưu tiên POI có toạ độ; đảm bảo ít nhất `min_mappable` điểm có toạ độ mỗi ngày.
    - Không bao giờ lấy trùng (dựa trên unique_key).
    """
    if poi_df is None or poi_df.empty:
        return [], used_keys

    df = _ensure_key_columns(_normalize_tags_column(_sanitize_latlon(poi_df.copy())))

    # Nhóm chuẩn
    df_food = _df_cat_exact(df, ["food", "cafe"])
    df_ent  = _df_cat_exact(df, ["entertainment"])
    df_att  = _df_cat_exact(df, ["attraction"])
    df_shop = _df_cat_exact(df, ["shopping"])
    df_nonfood = df[~df["category"].isin({"food", "cafe"})]

    # Bản có toạ độ
    v_food = _df_valid_coords(df_food)
    v_ent  = _df_valid_coords(df_ent)
    v_att  = _df_valid_coords(df_att)
    v_shop = _df_valid_coords(df_shop)
    v_nonf = _df_valid_coords(df_nonfood)
    v_all  = _df_valid_coords(df)

    # Cap food
    full_mix_selected = include_shopping and (include_att or include_ent)
    if full_mix_selected:
        cap_ratio = math.floor(max_per_day * max_food_ratio_if_full)
        food_cap = max(min_food, min(cap_ratio, max_food_abs))
    else:
        food_cap = max(min_food, math.floor(max_per_day * basic_food_ratio_cap))

    chosen: List[Dict] = []

    def add_from(df_pref: pd.DataFrame, df_fallback: pd.DataFrame, n: int):
        """Ưu tiên df_pref (có toạ độ), thiếu mới lấy df_fallback."""
        if n <= 0:
            return
        need = n
        got = _sample_no_repeat(df_pref, used_keys, need)
        chosen.extend(got)
        need -= len(got)
        if need > 0:
            chosen.extend(_sample_no_repeat(df_fallback, used_keys, need))

    # 1) Min food (bắt buộc)
    add_from(v_food, df_food, min_food)

    # 2) Min att+ent nếu bật (cố gắng 1-1; nếu thiếu thì bù từ pool chung)
    if include_att or include_ent:
        added = sum(1 for p in chosen if p.get("category") in {"attraction", "entertainment"})
        if include_att and added < 2:
            add_from(v_att, df_att, 1); added = sum(1 for p in chosen if p.get("category") in {"attraction", "entertainment"})
        if include_ent and added < 2:
            add_from(v_ent, df_ent, 1); added = sum(1 for p in chosen if p.get("category") in {"attraction", "entertainment"})
        if added < 2:
            pool_v = pd.concat([v_att, v_ent]).drop_duplicates(subset="unique_key")
            pool_f = pd.concat([df_att, df_ent]).drop_duplicates(subset="unique_key")
            add_from(pool_v, pool_f, 2 - added)

    # 3) Shopping = 1 nếu bật
    if include_shopping:
        add_from(v_shop, df_shop, 1)

    # 4) Fill đủ số lượng (ưu tiên non-food, tránh tràn food)
    def can_add_more_food(current: List[Dict]) -> bool:
        return _count_food_like(current) < food_cap

    while len(chosen) < max_per_day:
        before = len(chosen)
        need = max_per_day - before

        if include_att or include_ent:
            pool_v = pd.concat([v_att, v_ent]).drop_duplicates(subset="unique_key")
            pool_f = pd.concat([df_att, df_ent]).drop_duplicates(subset="unique_key")
            add_from(pool_v, pool_f, need)
            need = max_per_day - len(chosen)

        if need > 0:
            add_from(v_nonf, df_nonfood, need)
            need = max_per_day - len(chosen)

        if need > 0 and can_add_more_food(chosen):
            allowed = food_cap - _count_food_like(chosen)
            if allowed > 0:
                add_from(v_food, df_food, min(need, allowed))

        if len(chosen) == before:
            break

    # 5) Bảo đảm min_food tuyệt đối (nếu vì lý do nào đó vẫn thiếu)
    food_now = _count_food_like(chosen)
    if food_now < min_food:
        need = min_food - food_now
        extra = _sample_no_repeat(df_food, used_keys, need)
        if extra:
            free = max_per_day - len(chosen)
            if free > 0:
                chosen.extend(extra[:free]); extra = extra[free:]
            if extra:
                repl_idx = [i for i, p in enumerate(chosen) if (p.get("category") or "").lower() not in {"food","cafe"}]
                for i, newp in zip(repl_idx, extra):
                    chosen[i] = newp

    # 6) Đảm bảo tối thiểu số POI có toạ độ để vẽ map (thay thế trước, sau đó impute nếu vẫn thiếu)
    def count_mappable(lst: List[Dict]) -> int:
        return sum(1 for p in lst if _has_coords(p))

    have_coords = count_mappable(chosen)
    if have_coords < min_mappable:
        needed = min_mappable - have_coords
        repl_candidates = [i for i, p in enumerate(chosen) if not _has_coords(p)]
        ext = _sample_no_repeat(v_all, used_keys, needed)
        for i, newp in zip(repl_candidates, ext):
            chosen[i] = newp
        have_coords = count_mappable(chosen)

    if have_coords < min_mappable:
        # Trung tâm TP.HCM (fallback) — có thể chỉnh theo city nếu muốn
        lat_c, lon_c = 10.776, 106.700
        coords = [(float(p["lat"]), float(p["lon"])) for p in chosen if _has_coords(p)]
        if coords:
            lat_c = sum(lat for lat, _ in coords) / len(coords)
            lon_c = sum(lon for _, lon in coords) / len(coords)
        for p in chosen:
            if not _has_coords(p):
                p["lat"] = lat_c + random.uniform(-0.004, 0.004)
                p["lon"] = lon_c + random.uniform(-0.004, 0.004)

    return chosen[:max_per_day], used_keys


# ============================================================
# Public API
# ============================================================

def build_itinerary(params: Dict, poi_df, weather_now: Dict):
    """
    Không dùng KMeans. Chia ngày bằng cách chọn tuần tự từ toàn bộ pool,
    theo các quy tắc:
      - ≥ 2 food/cafe mỗi ngày.
      - Nếu bật attraction/entertainment → tổng ≥ 2; shopping ≤ 1.
      - Cap food: khi đủ bộ → 40% & abs 4; khi không đủ bộ → cap mềm 50%.
      - Ưu tiên POI có toạ độ; luôn đủ để vẽ map; không trùng giữa các ngày.
    """
    city = params["city"]
    days = int(params.get("days", 2))
    weather_desc = (weather_now or {}).get("description", "")
    max_per_day = int(params.get("max_poi_per_day", 10))  # mặc định 10

    # Preferences
    food_tags = params.get("food_tags", [])
    do_shopping = bool(params.get("do_shopping", False))
    do_entertainment = bool(params.get("do_entertainment", False))
    do_attraction = bool(params.get("do_attraction", False))
    entertainment_tags = params.get("entertainment_tags", [])
    attraction_tags = params.get("attraction_tags", [])

    # 1) Chuẩn hoá data
    if isinstance(poi_df, pd.DataFrame):
        base_df = poi_df.copy()
    elif isinstance(poi_df, list):
        base_df = pd.DataFrame(poi_df)
    else:
        base_df = pd.DataFrame()

    if base_df.empty:
        return []

    base_df = _sanitize_latlon(base_df)
    base_df = _normalize_tags_column(base_df)
    base_df["category"] = base_df.get("category", "unknown")
    base_df["category"] = base_df["category"].fillna("unknown").astype(str).map(_canonicalize_category)
    base_df = _ensure_key_columns(base_df)

    # phạt thời tiết (nhẹ)
    if "final" not in base_df.columns:
        base_df["final"] = 1.0
    if weather_desc:
        w = weather_desc.lower()
        if any(k in w for k in ["mưa", "rain", "storm"]):
            outdoor_mask = base_df["category"].eq("attraction") | base_df["tag"].str.contains("outdoor|park|garden", na=False)
            base_df.loc[outdoor_mask, "final"] = base_df.loc[outdoor_mask, "final"] * 0.8

    # 2) Lọc theo tag trên nhóm chuẩn (để ưu tiên)
    df_food = _filter_by_tags(base_df[base_df["category"].isin({"food", "cafe"})], food_tags)
    df_ent  = _filter_by_tags(base_df[base_df["category"].isin({"entertainment"})], entertainment_tags)
    df_att  = _filter_by_tags(base_df[base_df["category"].isin({"attraction"})], attraction_tags)
    df_shop = base_df[base_df["category"].isin({"shopping"})]

    # Hợp lại có ưu tiên theo tag (sau đó loại trùng)
    df_all = pd.concat([df_food, df_ent, df_att, df_shop, base_df], ignore_index=True)
    df_all = _ensure_key_columns(df_all).drop_duplicates(subset="unique_key")

    # 3) Chia ngày tuần tự (không KMeans)
    used_keys: set = set()
    days_pois: List[List[Dict]] = []

    for _ in range(days):
        daily, used_keys = _select_diverse_pois(
            poi_df=df_all,
            used_keys=used_keys,
            max_per_day=max_per_day,
            include_shopping=do_shopping,
            include_ent=do_entertainment,
            include_att=do_attraction,
            min_food=2,
            max_food_ratio_if_full=0.40,
            max_food_abs=4,
            basic_food_ratio_cap=0.50,
            min_mappable=3,
        )
        # fallback nếu còn thiếu nặng (dataset quá nhỏ): rút ngẫu nhiên không trùng
        if len(daily) < max(2, int(max_per_day * 0.6)):
            need = max_per_day - len(daily)
            extra = _sample_no_repeat(df_all, used_keys, need)
            daily.extend(extra)
        days_pois.append(daily[:max_per_day])

    # 4) Tối ưu khoảng cách trong ngày
    out_days: List[Dict] = []
    G = road_graph_for_city(city)
    for day_idx, dpois in enumerate(days_pois):
        if len(dpois) < 2:
            out_days.append({
                "title": f"Khám phá nhẹ - Ngày {day_idx + 1}",
                "pois": dpois,
                "distance": 0.0,
                "weather": weather_desc
            })
            continue

        dist, coords, _ = pairwise_distance_matrix(city, dpois)
        order = mst_order(dist)
        ordered_pois = [dpois[i] for i in order]
        total_km = total_distance(dist, order)

        for i in range(len(ordered_pois) - 1):
            try:
                d_km = shortest_distance_km(
                    G,
                    (ordered_pois[i]["lat"], ordered_pois[i]["lon"]),
                    (ordered_pois[i + 1]["lat"], ordered_pois[i + 1]["lon"])
                )
                ordered_pois[i]["next_distance_km"] = round(d_km, 2)
            except Exception:
                ordered_pois[i]["next_distance_km"] = "?"

        out_days.append({
            "title": f"Khám phá ngày {day_idx + 1}",
            "pois": ordered_pois,
            "distance": round(total_km, 2),
            "weather": weather_desc
        })

    return out_days
