import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Load Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Default model: fast + cheap
GEMINI_MODEL = "gemini-2.0-flash"

# Configure Gemini if API key exists
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def compose_plan_response(plan_raw, params):
    """
    Uses Gemini to write a polished travel itinerary with weather notes,
    travel time, cost hints, and indoor suggestions for rainy days.
    Falls back to a rule-based summary if no API key is available.
    """

    if not GEMINI_API_KEY:
        lines = [f"⛅ Weather: {plan_raw.get('weather', 'n/a')}"]
        for i, day in enumerate(plan_raw.get("days", []), 1):
            names = ", ".join(p["name"] for p in day["order"])
            lines.append(f"Day {i}: {names} (≈ {day['distance_km']} km)")
        return "\n".join(lines)

    system_text = (
        "You are an expert travel planner. Write clearly, concisely, and in a friendly tone."
    )

    user_text = f"""
    Parameters: {json.dumps(params, ensure_ascii=False)}
    RawPlan: {json.dumps(plan_raw, ensure_ascii=False)}

    Requirements:
    - Write a {params.get('days', 2)}-day travel itinerary.
    - Each day should include 4–6 attractions.
    - Mention the weather conditions.
    - Estimate travel distance per day.
    - Suggest visit duration, break times, and pacing.
    - If the weather indicates rain, prioritize indoor activities.
    """

    prompt = f"{system_text}\n\n{user_text}".strip()

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        print("Gemini error:", e)
        return "Failed to generate itinerary. Please try again."


def generate_day_summary(day_number, day_data):
    """
    Generates a 3–4 sentence English summary for a single day of the itinerary.
    Uses Gemini; falls back to a simple rule-based summary if Gemini is unavailable.
    """

    if not GEMINI_API_KEY:
        names = ", ".join([p.get("name", "") for p in day_data.get("order", [])])
        return f"Day {day_number}: Visit {names}."

    places = day_data.get("order", [])
    names = [p.get("name", "") for p in places]
    descriptions = [p.get("description", "") for p in places if p.get("description")]

    prompt = f"""
    You are a professional travel writer.
    Write a short and engaging English summary (3–4 sentences)
    for Day {day_number} of a travel itinerary.

    Places: {', '.join(names)}
    Descriptions: {'; '.join(descriptions)}

    Requirements:
    - Natural and smooth flow.
    - Friendly, positive tone.
    - Read like a travel recommendation.
    """

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        print("Gemini error:", e)
        return f"Day {day_number}: Summary not generated."
