from core.weather import get_weather
from core.recommender import recommend_pois
from core.routing import build_routes

def generate_travel_plans(params):
    """
    Sinh kế hoạch du lịch offline dựa vào CSV Hồ Chí Minh
    """
    city = params.get("city", "Hồ Chí Minh")
    budget = params.get("budget_vnd", 300000)
    taste = params.get("taste_tags", [])
    activities = params.get("activity_tags", [])
    walk_tolerance_km = params.get("walk_tolerance_km", 5.0)

    weather = get_weather(city)
    pois = recommend_pois(
        city=city,
        poi_df=params.get("poi_df"),
        user_query=" ".join(taste + activities + [city]),
        taste_tags=taste,
        activity_tags=activities,
        budget_per_day=budget,
        walk_tolerance_km=walk_tolerance_km,
        weather_desc=weather.get("description", "")
    )
    routes = build_routes(pois, budget, "normal")
    return {"weather": weather, "routes": routes}
