from fpdf import FPDF
from pathlib import Path
import datetime
import google.generativeai as genai
import json
import re
import unicodedata
import urllib.request
import tempfile
import os

DEFAULT_MODEL = "gemini-2.0-flash"


# ==================================================
# 🧹 TEXT CLEANING
# ==================================================

def strip_emoji(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\xa0", " ").replace("\u200b", "")
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002700-\U000027BF"
        u"\U0001F900-\U0001F9FF"
        u"\u200b"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub("", text)

def _safe_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = strip_emoji(s)

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # Remove URLs + markdown
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"www\.\S+", "", s)
    s = re.sub(r"[*_`#>\[\]\(\)\"“”‘’•]+", "", s)
    s = re.sub(r"[^\x20-\x7E]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


# ==================================================
# 📦 SAFE CELL HELPERS
# ==================================================

def safe_multicell(pdf: FPDF, w, h, text, **kwargs):
    if not text:
        return
    text = _safe_text(text)
    if w == 0:
        w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.multi_cell(w, h, text, **kwargs)
    pdf.ln(0.8)


def safe_cell(pdf: FPDF, w, h, text, **kwargs):
    if not text:
        return
    text = _safe_text(text)
    if w == 0:
        w = pdf.w - pdf.l_margin - pdf.r_margin

    if pdf.get_string_width(text) > w:
        pdf.multi_cell(w, h, text, **kwargs)
        pdf.ln(0.8)
    else:
        pdf.cell(w, h, text, **kwargs)


# ==================================================
# 🤖 GEMINI HELPERS
# ==================================================

def summarize_day_parts(day_data, model_name=DEFAULT_MODEL):
    """
    New improved prompt → very stable JSON
    """
    prompt = f"""
You are a professional travel planner.
Divide the following POIs into 4 time-of-day groups:

- morning
- noon
- afternoon
- evening

Return ONLY a valid JSON object with EXACT structure:
{{
  "morning": [...],
  "noon": [...],
  "afternoon": [...],
  "evening": [...]
}}

Input POIs: {day_data}
"""
    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return json.loads(resp.text.strip())
    except:
        return {"morning": [], "noon": [], "afternoon": [], "evening": []}


def translate_to_english(text, model_name=DEFAULT_MODEL):
    if not text:
        return ""
    text = strip_emoji(text)
    prompt = f"""
Translate the following text into concise English.
Return ONLY the translated text (no explanation):

{text}
"""
    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return _safe_text(resp.text.strip())
    except:
        return _safe_text(text)


# ==================================================
# 🖼️ DOWNLOAD IMAGE HELPER
# ==================================================

def download_image(url: str) -> str | None:
    """
    Download remote image → return local temp path
    """
    try:
        tmpdir = tempfile.gettempdir()
        ext = url.split(".")[-1][:4]
        filename = f"poi_{abs(hash(url))}.{ext}"
        path = os.path.join(tmpdir, filename)
        urllib.request.urlretrieve(url, path)
        return path
    except:
        return None


# ==================================================
# 🎯 OCEAN BLUE UI CONSTANTS
# ==================================================

COLOR_BLUE = (30, 136, 229)     # #1E88E5
COLOR_BLUE_LIGHT = (144, 202, 249)  # #90CAF9
COLOR_TEXT = (0, 0, 0)


# ==================================================
# 📄 PDF EXPORT
# ==================================================

def export_itinerary_to_pdf(out_days, filename="itinerary.pdf"):
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    fonts_dir = Path("fonts")
    pdf.add_font("Noto", "", str(fonts_dir / "NotoSans-Regular.ttf"), uni=True)
    pdf.add_font("Noto", "B", str(fonts_dir / "NotoSans-Bold.ttf"), uni=True)
    pdf.add_font("Noto", "I", str(fonts_dir / "NotoSans-Italic.ttf"), uni=True)

    pdf.set_font("Noto", "", 12)

    # For each day
    for day_index, day in enumerate(out_days, start=1):

        # ===== HEADER BLUE BOX =====
        pdf.set_fill_color(*COLOR_BLUE)
        pdf.set_text_color(255, 255, 255)

        title = translate_to_english(day.get("title", f"Day {day_index}"))
        pdf.set_font("Noto", "B", 15)
        pdf.cell(0, 12, f"DAY {day_index} — {title}", ln=True, fill=True)

        pdf.ln(3)

        pdf.set_text_color(*COLOR_TEXT)
        pdf.set_font("Noto", "", 11)

        weather = translate_to_english(day.get("weather", ""))
        distance = day.get("distance", 0.0)

        if weather:
            safe_multicell(pdf, 0, 6, f"Weather: {weather}")
        safe_multicell(pdf, 0, 6, f"Distance: {distance:.2f} km")

        # separator line
        pdf.set_draw_color(*COLOR_BLUE_LIGHT)
        pdf.set_line_width(0.8)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(4)

        # ===== TIME-OF-DAY SLOTS =====
        pois = day.get("pois", [])
        summary = summarize_day_parts(json.dumps([p["name"] for p in pois], ensure_ascii=False))

        slots = [
            ("Morning", summary.get("morning", [])),
            ("Noon", summary.get("noon", [])),
            ("Afternoon", summary.get("afternoon", [])),
            ("Evening", summary.get("evening", [])),
        ]

        # normalize helper
        def _norm(s):
            if not s:
                return ""
            s = unicodedata.normalize("NFKD", s)
            s = "".join(ch for ch in s if not unicodedata.combining(ch))
            return re.sub(r"[^a-zA-Z0-9 ]", "", s).strip().lower()

        slot_map = [set(_norm(name) for name in lst) for _, lst in slots]
        has_slots = any(len(s) > 0 for s in slot_map)

        # if no slots → render all POIs flat
        if not has_slots:
            pdf.set_font("Noto", "B", 13)
            safe_multicell(pdf, 0, 7, "Attractions")
            pdf.set_font("Noto", "", 11)
            for p in pois:
                _render_poi_block(pdf, p)
        else:
            for (slot_name, _), nameset in zip(slots, slot_map):
                if not nameset:
                    continue

                pdf.set_font("Noto", "B", 13)
                safe_multicell(pdf, 0, 7, slot_name)
                pdf.set_font("Noto", "", 11)

                for p in pois:
                    raw = p.get("name", "")
                    trans = translate_to_english(raw)
                    if _norm(raw) in nameset or _norm(trans) in nameset:
                        _render_poi_block(pdf, p)

                pdf.ln(2)

        pdf.ln(5)

    out_path = Path(filename)
    pdf.output(str(out_path))
    return str(out_path)


# ==================================================
# 🏙️ POI RENDERER (WITH IMAGE, NO EMOJI)
# ==================================================
import requests
from mimetypes import guess_extension


import re

def fix_google_img(url: str):
    """Convert Google Maps /p/ID links into real JPEG links."""
    if not isinstance(url, str) or not url:
        return None

    # MATCH /p/<PHOTO_ID>
    m = re.search(r"googleusercontent\.com\/p\/([^=?]+)", url)
    if m:
        photo_id = m.group(1)
        # Convert to real downloadable image
        return f"https://lh3.googleusercontent.com/p/{photo_id}=s1200"

    # Other googleusercontent links → KEEP ORIGINAL (because proxy breaks PDF)
    return url



def download_image(url: str) -> str | None:
    """Download ảnh bằng requests + detect mime-type + save đúng định dạng."""
    try:
        url = fix_google_img(url)
        if not url:
            return None

        headers = {"User-Agent": "Mozilla/5.0"}

        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None

        content_type = resp.headers.get("content-type", "")
        ext = ".jpg"

        if "png" in content_type:
            ext = ".png"
        elif "jpeg" in content_type or "jpg" in content_type:
            ext = ".jpg"
        else:
            # fallback đoán extension
            ext2 = guess_extension(content_type)
            if ext2:
                ext = ext2

        tmpdir = tempfile.gettempdir()
        filename = f"poi_{abs(hash(url))}{ext}"
        path = os.path.join(tmpdir, filename)

        with open(path, "wb") as f:
            f.write(resp.content)

        return path

    except Exception as e:
        print("IMG ERROR:", e)
        return None


def _render_poi_block(pdf: FPDF, p: dict):
    # ==== DATA EXTRACTION ====
    name = translate_to_english(p.get("name", "Unnamed Location"))
    desc = translate_to_english(p.get("description", ""))
    address = translate_to_english(p.get("address", ""))
    rating = p.get("rating")
    cost = p.get("avg_cost")
    tag = p.get("tag", "")


    # Auto-select best image
    img_url = (
        p.get("image_url1") or
        p.get("image_url2") or
        p.get("image_url")      # fallback nếu DB dùng field này
    )
    img_url = fix_google_img(img_url)

    # ==== TITLE ====
    pdf.set_font("Noto", "B", 12)
    pdf.set_text_color(30, 136, 229)  # Ocean blue
    safe_multicell(pdf, 0, 6, name)
    pdf.set_text_color(0, 0, 0)

    # ==== IMAGE ====
    if img_url:
        img_path = download_image(img_url)
        if img_path:
            try:
                pdf.image(img_path, w=90)
                pdf.ln(3)
            except:
                pass

    # ==== INFO ROW ====
    info = []
    if rating:
        info.append(f"⭐ Rating: {rating}")
    if cost:
        try:
            info.append(f"💵 Avg cost: {int(cost):,} VND")
        except:
            pass
    if tag:
        info.append(f"🏷 Tag: {tag}")

    if info:
        pdf.set_font("Noto", "I", 10)
        safe_multicell(pdf, 0, 6, " | ".join(info))

    # ==== ADDRESS ====
    if address:
        pdf.set_font("Noto", "I", 10)
        safe_multicell(pdf, 0, 6, f"📍 Address: {address}")

    # ==== DESCRIPTION ====
    if desc:
        pdf.set_font("Noto", "", 10)
        safe_multicell(pdf, 0, 6, desc)

    # Separator line
    pdf.set_draw_color(220, 220, 220)
    pdf.set_line_width(0.3)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())

    pdf.ln(4)
