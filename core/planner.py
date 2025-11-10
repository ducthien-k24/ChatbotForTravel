# core/planner.py
from core.weather import get_weather
from core.recommender import recommend_pois
from core.routing import build_routes

def generate_travel_plans(params):
    """
    params:
      - city, budget_vnd, mood, taste_tags, activity_tags, lat, lon
    """
    city = params.get("city", "Đà Lạt")
    mood = params.get("mood", "neutral")
    budget = params.get("budget_vnd", 300000)
    taste = params.get("taste_tags", [])
    act = params.get("activity_tags", [])
    user_lat = params.get("lat")
    user_lon = params.get("lon")

    weather = get_weather(city)
    pois = recommend_pois(
        city=city,
        taste_tags=taste,
        activity_tags=act,
        limit=15,
        user_lat=user_lat,
        user_lon=user_lon,
        budget=budget,
        user_query=" ".join(taste + act + [city])
    )
    routes = build_routes(pois, budget, mood)
    return {"weather": weather, "routes": routes}
