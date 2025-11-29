# core/weather.py
from __future__ import annotations
import os
import requests
import random
import datetime as dt
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")


# -----------------------
# Helpers
# -----------------------
def _advice(desc_vi: str) -> str:
    """Đưa ra lời khuyên mang đồ theo mô tả thời tiết (tiếng Việt)."""
    d = (desc_vi or "").lower()
    if any(k in d for k in ["mưa", "rain", "storm", "dông", "giông"]):
        return "☔ Có mưa — nhớ mang áo mưa/ô, bọc chống nước cho đồ điện."
    if any(k in d for k in ["nắng", "clear"]):
        return "🧴 Nắng đẹp — bôi kem chống nắng, mang nón & nước uống."
    if any(k in d for k in ["mây", "cloud"]):
        return "⛅ Trời nhiều mây — thời tiết dễ chịu."
    return "ℹ️ Thời tiết ổn — mang nước và giày đi bộ thoải mái."

def _fmt_summary(temp: Optional[float], humid: Optional[int], desc: str) -> str:
    """Gộp mô tả, nhiệt độ, độ ẩm thành một dòng gọn."""
    bits = []
    if desc:
        bits.append(desc)
    if temp is not None:
        bits.append(f"{temp:.0f}°C")
    if humid is not None:
        bits.append(f"RH {humid}%")
    return " • ".join(bits)

def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


# -----------------------
# Current weather (header)
# -----------------------
def get_weather(city: str) -> Dict:
    """
    Thời tiết hiện tại (để hiển thị nhỏ ở header).
    Trả về: {'city','temp','humidity','description'}
    """
    if API_KEY:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city},VN&appid={API_KEY}&units=metric&lang=vi"
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            data = r.json()
            if data.get("cod") == 200:
                return {
                    "city": city,
                    "temp": _safe_float(data.get("main", {}).get("temp")),
                    "humidity": _safe_int(data.get("main", {}).get("humidity")),
                    "description": (data.get("weather", [{}])[0].get("description") or "").strip(),
                }
        except Exception:
            pass

    # Fallback mô phỏng (offline)
    return {
        "city": city,
        "temp": random.randint(20, 33),
        "humidity": random.randint(55, 85),
        "description": random.choice(["nắng nhẹ", "mưa rào", "mây rải rác", "âm u"]),
    }


# -----------------------
# Daily forecast (per day)
# -----------------------
def get_daily_forecast(city: str, days: int = 3) -> List[Dict]:
    """
    Dự báo từng ngày, độ dài = days (tối đa 7).
    Mỗi phần tử:
      {
        'date': 'YYYY-MM-DD',
        'temp': 31.5,             # nhiệt độ trung bình ngày (°C)
        'humidity': 70,           # RH (%)
        'description': 'mưa rào', # mô tả VI từ API
        'summary': 'mưa rào • 32°C • RH 70%',
        'advice': '☔ Có mưa — nhớ mang áo mưa/ô, bọc chống nước cho đồ điện.'
      }
    - Có API key: dùng /forecast (5-day/3-hour), gộp theo ngày.
    - Không có: fallback offline ngẫu nhiên.
    """
    days = max(1, min(int(days or 1), 7))

    if API_KEY:
        try:
            url = f"https://api.openweathermap.org/data/2.5/forecast?q={city},VN&appid={API_KEY}&units=metric&lang=vi"
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            data = r.json()
            lst = data.get("list", [])
            tz_shift = _safe_int(data.get("city", {}).get("timezone")) or 0

            # Gom theo ngày (UTC + tz_shift)
            daily: Dict[str, Dict[str, list]] = {}
            for item in lst:
                ts = _safe_int(item.get("dt"))
                if ts is None:
                    continue
                d = dt.datetime.utcfromtimestamp(ts + tz_shift).date()
                key = d.isoformat()

                bucket = daily.setdefault(key, {"temps": [], "humid": [], "descs": []})
                main = item.get("main", {})
                weather_arr = item.get("weather", [])

                t = _safe_float(main.get("temp"))
                h = _safe_int(main.get("humidity"))
                if t is not None:
                    bucket["temps"].append(t)
                if h is not None:
                    bucket["humid"].append(h)
                if weather_arr:
                    desc = (weather_arr[0].get("description") or "").strip()
                    if desc:
                        bucket["descs"].append(desc)

            # Build kết quả theo từng ngày từ hôm nay
            out: List[Dict] = []
            today = dt.date.today()
            for i in range(days):
                day_key = (today + dt.timedelta(days=i)).isoformat()
                bucket = daily.get(day_key)

                if not bucket:
                    # khi API 5d/3h không đủ xa hoặc thiếu slot → fallback ngày đó
                    temp = random.randint(24, 33)
                    hum = random.randint(55, 85)
                    desc = random.choice(["nắng nhẹ", "mây rải rác", "mưa rào", "âm u"])
                else:
                    temps = bucket["temps"]
                    hums = bucket["humid"]
                    descs = bucket["descs"]

                    temp = float(sum(temps) / len(temps)) if temps else None
                    hum = int(sum(hums) / len(hums)) if hums else None
                    # chọn mô tả xuất hiện nhiều nhất trong ngày
                    desc = max(descs, key=descs.count) if descs else ""

                summary = _fmt_summary(temp, hum, desc)
                out.append({
                    "date": day_key,
                    "temp": temp,
                    "humidity": hum,
                    "description": desc,
                    "summary": summary,
                    "advice": _advice(desc),
                })

            return out
        except Exception:
            pass

    # -------- fallback offline (không có API/ lỗi API) --------
    today = dt.date.today()
    out: List[Dict] = []
    for i in range(days):
        d = (today + dt.timedelta(days=i)).isoformat()
        desc = random.choice(["nắng nhẹ", "mây rải rác", "mưa rào", "âm u"])
        temp = random.randint(24, 33)
        hum = random.randint(55, 85)
        out.append({
            "date": d,
            "temp": temp,
            "humidity": hum,
            "description": desc,
            "summary": _fmt_summary(temp, hum, desc),
            "advice": _advice(desc),
        })
    return out
