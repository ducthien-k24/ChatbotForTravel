import os, json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def parse_prompt_to_params(prompt: str):
    """
    Trả về JSON: {city, budget_vnd, days, taste_tags, activity_tags, walk_tolerance_km, transport}
    """
    if not OPENAI_API_KEY:
        # fallback tối thiểu
        return {"city":"Hồ Chí Minh","budget_vnd":1_500_000,"days":2,"taste_tags":[],"activity_tags":[],"walk_tolerance_km":5.0,"transport":"xe máy/ô tô"}
    client = OpenAI(api_key=OPENAI_API_KEY)
    sys = """Bạn là module trích tham số cho TravelGPT+.
    Trả về JSON có các khóa: city (string), budget_vnd (int), days (int),
    taste_tags ([string]), activity_tags ([string]), walk_tolerance_km (float), transport (string).
    Chỉ trả JSON hợp lệ."""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":prompt}],
            temperature=0
        )
        return json.loads(r.choices[0].message.content)
    except Exception:
        return {"city":"Hồ Chí Minh","budget_vnd":1_500_000,"days":2,"taste_tags":[],"activity_tags":[],"walk_tolerance_km":5.0,"transport":"xe máy/ô tô"}
