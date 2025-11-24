import os
import requests
from dotenv import load_dotenv

# --- Nạp biến môi trường từ .env ---
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("⚠️ Chưa có GOOGLE_API_KEY trong file .env")


# =====================================================
# 🔍 TÌM PLACE_ID (ưu tiên theo tọa độ)
# =====================================================
def search_place(name: str, city: str, lat: float = None, lon: float = None):
    """
    Tìm place_id của địa điểm trên Google Places.
    Ưu tiên Nearby Search (dựa trên lat/lon), fallback về Text Search nếu cần.
    """
    if lat and lon:
        query = f"{name} in {city}"
        url = (
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            f"?location={lat},{lon}&radius=500&keyword={query}&key={API_KEY}"
        )
    else:
        query = f"{name}, {city}, Vietnam"
        url = (
            f"https://maps.googleapis.com/maps/api/place/textsearch/json?"
            f"query={query}&key={API_KEY}"
        )

    r = requests.get(url)
    data = r.json()

    if "results" not in data or not data["results"]:
        print(f"❌ Không tìm thấy: {name}")
        return None

    place_id = data["results"][0]["place_id"]
    return place_id


# =====================================================
# 🏨 LẤY THÔNG TIN CHI TIẾT (rating, review, giờ mở cửa)
# =====================================================
def get_place_details(place_id: str):
    url = (
        "https://maps.googleapis.com/maps/api/place/details/json"
        f"?place_id={place_id}"
        "&fields=name,rating,user_ratings_total,opening_hours,photos"
        f"&key={API_KEY}"
    )
    r = requests.get(url)
    data = r.json()
    if "result" not in data:
        return {}
    return data["result"]


# =====================================================
# 🖼️ LẤY LINK ẢNH
# =====================================================
def get_photo_url(photo_ref: str, maxwidth: int = 800):
    return (
        "https://maps.googleapis.com/maps/api/place/photo"
        f"?maxwidth={maxwidth}&photo_reference={photo_ref}&key={API_KEY}"
    )
