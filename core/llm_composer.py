import os, json
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def compose_plan_response(plan_raw, params):
    """Dùng LLM viết lịch trình “đẹp”, có gợi ý thời tiết/chi phí/di chuyển. Fallback rule-based nếu không có API."""
    if not OPENAI_API_KEY:
        lines = [f"⛅ Thời tiết: {plan_raw.get('weather','n/a')}"]
        for i, day in enumerate(plan_raw.get("days", []), 1):
            names = ", ".join(p['name'] for p in day["order"])
            lines.append(f"Ngày {i}: {names} (≈ {day['distance_km']} km)")
        return "\n".join(lines)

    client = OpenAI(api_key=OPENAI_API_KEY)
    sys = "Bạn là travel planner xuất sắc, trả lời tiếng Việt, rõ ràng, cô đọng."
    user = f"""
    Params: {json.dumps(params, ensure_ascii=False)}
    PlanRaw: {json.dumps(plan_raw, ensure_ascii=False)}
    Yêu cầu: Viết lịch trình {params.get('days',2)} ngày; mỗi ngày 4-6 điểm; nhắc thời tiết; ước lượng quãng đường/ngày;
    gợi ý thời gian tham quan & nghỉ; lưu ý mưa thì ưu tiên trong nhà.
    """
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.6
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return "Không tạo được văn bản lịch trình. Hãy thử lại."
    
def generate_day_summary(day_number, day_data):
    if not OPENAI_API_KEY:
        names = ", ".join([p.get("name", "") for p in day_data.get("order", [])])
        return f"Day {day_number}: Visit {names}."

    client = OpenAI(api_key=OPENAI_API_KEY)
    places = day_data.get("order", [])
    names = [p.get("name", "") for p in places]
    descs = [p.get("description", "") for p in places if p.get("description")]

    prompt = f"""
    You are a professional travel writer.
    Write a short and engaging English summary (3–4 sentences) for Day {day_number} of a travel itinerary.
    Base your writing on these places and their short descriptions:

    Places: {', '.join(names)}
    Descriptions: {'; '.join(descs)}.

    Write in natural English, positive tone, and make it flow like a travel recommendation.
    """

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return f"Day {day_number}: Summary not generated."
