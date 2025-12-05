import json
import google.generativeai as genai

DEFAULT_MODEL = "gemini-2.0-flash"


def analyze_user_preferences(params, model_name=DEFAULT_MODEL):
    """
    Use Gemini to determine the optimal distribution of POIs per day,
    following strict but intelligent travel rules.
    """

    max_poi = int(params.get("max_poi_per_day", 6))

    prompt = f"""
You are an expert travel planner AI. Your task is to determine how many POIs
from each category should be included PER DAY based on the user's preferences,
selected tags, and expected travel style.

User Inputs (JSON):
{json.dumps(params, ensure_ascii=False)}

VERY STRICT RULES:

1. total_per_day must equal EXACTLY {max_poi}.

2. FOOD:
   - Must ALWAYS be >= 2.
   - Cafe is a SEPARATE category and optional.
   - Cafe may be 0 or 1. Never more than 1.
   - Food should be considered essential.

3. SHOPPING:
   - Only allowed if user enabled "do_shopping".
   - Must be 0 or 1. Never more than 1.

4. ATTRACTION & ENTERTAINMENT:
   - If both enabled → split remaining slots as evenly as possible.
   - If only one enabled → allocate most remaining slots to that category.
   - These categories should take priority over cafe and shopping.

5. After assigning food >= 2, and optional cafe/shopping,
   distribute the remaining slots intelligently.

6. Output MUST be valid JSON only.

7. Distance realism rule:
       - Assume all POIs are within the same city.
       - Do NOT plan itineraries that would require traveling > 20km between any two POIs.
       - Favor clusters of POIs that are geographically close.

OUTPUT FORMAT (MANDATORY):
{{
  "total_per_day": number,
  "distribution": {{
    "food": number,
    "cafe": number,
    "attraction": number,
    "entertainment": number,
    "shopping": number
  }},
  "notes": "short explanation"
}}

Output ONLY valid JSON. No extra text.
"""


    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    text = response.text.strip()

    try:
        data = json.loads(text)
        return data
    except Exception:
        # fallback in case of parsing error
        return {
            "total_per_day": max_poi,
            "distribution": {
                "food": max(2, max_poi - 3),
                "cafe": 0,
                "attraction": 1 if params.get("do_attraction") else 0,
                "entertainment": 1 if params.get("do_entertainment") else 0,
                "shopping": 1 if params.get("do_shopping") else 0,
            },
            "notes": "Fallback distribution."
        }
