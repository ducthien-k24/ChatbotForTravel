# core/llm_orchestrator.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

def ask_llm(prompt, system_prompt="Bạn là TravelGPT – trợ lý du lịch thông minh."):
    if not OPENAI_KEY:
        # Fallback chat đơn giản
        return "Mình đang ở chế độ đơn giản (không có API), bạn có thể hỏi mình về địa điểm, thời tiết, hay lịch trình cơ bản nhé!"
    try:
        client = OpenAI(api_key=OPENAI_KEY)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return "OpenAI đang gặp sự cố, vui lòng thử lại sau."
