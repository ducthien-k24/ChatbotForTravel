import os, requests, random
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_weather(city: str):
    if API_KEY:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city},VN&appid={API_KEY}&units=metric&lang=vi"
            r = requests.get(url, timeout=8)
            data = r.json()
            if data.get("cod") == 200:
                return {
                    "city": city,
                    "temp": round(float(data["main"]["temp"]), 1),
                    "humidity": int(data["main"]["humidity"]),
                    "description": data["weather"][0]["description"]
                }
        except Exception:
            pass
    # Fallback mô phỏng
    return {
        "city": city,
        "temp": random.randint(20, 33),
        "humidity": random.randint(55, 85),
        "description": random.choice(["nắng nhẹ","mưa rào","mây rải rác","âm u"])
    }
