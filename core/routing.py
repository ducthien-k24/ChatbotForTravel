# core/routing.py
import random

def build_routes(pois, budget, mood):
    """
    Ghép vài kịch bản tuyến đường: Chill / Ẩm thực / Check-in
    mỗi tuyến 3 điểm (hoặc ít hơn nếu dữ liệu ít).
    """
    routes = []
    if not pois:
        return routes

    styles = ["Chill", "Ẩm thực", "Check-in"]
    for style in styles:
        k = min(3, len(pois))
        selected = random.sample(pois, k) if k > 0 else []
        total_cost = sum(int(p.get("avg_cost", 100000) or 100000) for p in selected)
        routes.append({
            "style": style,
            "mood": mood,
            "places": selected,
            "total_cost": total_cost,
            "estimated_duration": random.randint(3, 6)  # giờ
        })
    return routes
