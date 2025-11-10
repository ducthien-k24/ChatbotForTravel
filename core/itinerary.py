from typing import Dict, List
from .route_optimizer import pairwise_distance_matrix, mst_order, greedy_path, total_distance
from .recommender import recommend_pois

def _penalize_by_weather(pois: List[Dict], weather_desc: str):
    if not weather_desc: return pois
    w = weather_desc.lower()
    if "mưa" in w or "rain" in w or "storm" in w:
        # đẩy outdoor xuống thấp (nếu itinerary chọn quá nhiều outdoor)
        for p in pois:
            if str(p["category"]) in {"park","garden","viewpoint","attraction"}:
                p["final"] *= 0.85
    return pois

def _select_pois_for_days(pois: List[Dict], days: int, max_per_day: int = 6):
    pois = sorted(pois, key=lambda x: x["final"], reverse=True)
    k = min(len(pois), days*max_per_day)
    chosen = pois[:k]
    # chia đều theo ngày
    per_day = []
    for d in range(days):
        per_day.append(chosen[d::days][:max_per_day])
    return per_day

def build_itinerary(params: Dict, poi_df, weather_now: Dict):
    """
    params: {city, budget_vnd, days, taste_tags, activity_tags, walk_tolerance_km, transport}
    return: dict {weather, days: [ {order:[pois], distance_km: float} ] }
    """
    city   = params["city"]
    days   = int(params.get("days", 2))
    budget = int(params.get("budget_vnd", 1_500_000))
    taste  = params.get("taste_tags", [])
    acts   = params.get("activity_tags", [])
    walk_km = float(params.get("walk_tolerance_km", 5.0))
    weather_desc = weather_now.get("description", "")

    # 1) lấy ứng viên theo cá nhân hoá
    base = recommend_pois(
        city=city, poi_df=poi_df, user_query="",
        taste_tags=taste, activity_tags=acts,
        budget_per_day=budget, walk_tolerance_km=walk_km,
        weather_desc=weather_desc
    )
    base = _penalize_by_weather(base, weather_desc)

    # 2) chia ngày (mỗi ngày ~5-6 điểm)
    days_pois = _select_pois_for_days(base, days, max_per_day=6)

    # 3) tối ưu thứ tự trong từng ngày bằng MST/Greedy trên mạng đường thực (Dijkstra)
    out_days = []
    for dpois in days_pois:
        if len(dpois) < 2:
            out_days.append({"order": dpois, "distance_km": 0.0})
            continue
        dist, coords, G = pairwise_distance_matrix(city, dpois)
        order = mst_order(dist)     # hoặc greedy_path(dist)
        km = total_distance(dist, order)
        ordered_pois = [dpois[i] for i in order]
        out_days.append({"order": ordered_pois, "distance_km": round(km, 2)})

    return {"weather": weather_desc, "days": out_days}
