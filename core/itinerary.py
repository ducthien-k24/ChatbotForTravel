from typing import Dict, List
from .route_optimizer import pairwise_distance_matrix, mst_order, greedy_path, total_distance
from .recommender import recommend_pois
from .geo_graph import road_graph_for_city, shortest_distance_km

def _penalize_by_weather(pois: List[Dict], weather_desc: str):
    if not weather_desc:
        return pois

    w = weather_desc.lower()
    if "mưa" in w or "rain" in w or "storm" in w:
        # giảm điểm các địa điểm ngoài trời
        for p in pois:
            cat = str(p.get("category", "")).lower()
            if cat in {"park", "garden", "viewpoint", "attraction"}:
                p["final"] = p.get("final", 1) * 0.85
    return pois



def _select_pois_for_days(pois: List[Dict], days: int, max_per_day: int = 6):
    pois = sorted(pois, key=lambda x: x["final"], reverse=True)
    k = min(len(pois), days * max_per_day)
    chosen = pois[:k]
    per_day = []
    for d in range(days):
        per_day.append(chosen[d::days][:max_per_day])
    return per_day


def build_itinerary(params: Dict, poi_df, weather_now: Dict):
    """
    Sinh lịch trình tối ưu hoá theo ngày.
    Trả về dạng dễ render bằng card.
    """
    city   = params["city"]
    days   = int(params.get("days", 2))
    budget = int(params.get("budget_vnd", 1_500_000))
    taste  = params.get("taste_tags", [])
    acts   = params.get("activity_tags", [])
    walk_km = float(params.get("walk_tolerance_km", 5.0))
    weather_desc = weather_now.get("description", "")

    # 1️⃣ Gợi ý địa điểm phù hợp
    base = recommend_pois(
        city=city, poi_df=poi_df, user_query="",
        taste_tags=taste, activity_tags=acts,
        budget_per_day=budget, walk_tolerance_km=walk_km,
        weather_desc=weather_desc
    )
    base = _penalize_by_weather(base, weather_desc)

    # 2️⃣ Chia địa điểm theo ngày (mỗi ngày ~5-6 điểm)
    days_pois = _select_pois_for_days(base, days, max_per_day=6)

    # 3️⃣ Tối ưu thứ tự + tính khoảng cách thực tế giữa các điểm
    out_days = []
    G = road_graph_for_city(city)
    for dpois in days_pois:
        if len(dpois) < 2:
            out_days.append({"title": "Tham quan nhẹ", "pois": dpois, "distance": 0.0})
            continue

        dist, coords, _ = pairwise_distance_matrix(city, dpois)
        order = mst_order(dist)
        ordered_pois = [dpois[i] for i in order]

        # tổng quãng đường
        total_km = total_distance(dist, order)

        # tính khoảng cách giữa các điểm liên tiếp để hiển thị
        for i in range(len(ordered_pois) - 1):
            try:
                d_km = shortest_distance_km(
                    G,
                    (ordered_pois[i]["lat"], ordered_pois[i]["lon"]),
                    (ordered_pois[i+1]["lat"], ordered_pois[i+1]["lon"])
                )
                ordered_pois[i]["next_distance_km"] = round(d_km, 2)
            except Exception as e:
                print("⚠️ Lỗi khi tính khoảng cách:", e)
                ordered_pois[i]["next_distance_km"] = "?"

        out_days.append({
            "title": "Khám phá ngày " + str(len(out_days)+1),
            "pois": ordered_pois,
            "distance": round(total_km, 2),
            "weather": weather_desc
        })

    return out_days
