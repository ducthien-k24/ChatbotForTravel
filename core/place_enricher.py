import pandas as pd
import requests

def _get_wikipedia_image(name: str):
    """Lấy ảnh minh họa và mô tả từ Wikipedia (miễn phí, không cần API key)."""
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "pageimages|extracts",
            "titles": name,
            "exintro": True,
            "explaintext": True,
            "pithumbsize": 600,
        }
        r = requests.get(url, params=params, timeout=6)
        data = r.json().get("query", {}).get("pages", {})
        for _, page in data.items():
            img = page.get("thumbnail", {}).get("source")
            desc = page.get("extract", "")
            return img, desc
        return None, ""
    except Exception:
        return None, ""

def enrich_list_with_images(pois):
    """Bổ sung ảnh + mô tả từ Wikipedia (nhưng KHÔNG lọc bỏ địa điểm thiếu ảnh)."""
    df = pd.DataFrame(pois)
    imgs, descs = [], []
    for _, row in df.iterrows():
        img, desc = _get_wikipedia_image(str(row["name"]))
        imgs.append(img)
        descs.append(desc or row.get("description", ""))
    df["image"] = imgs
    df["description"] = descs
    df["opening_hours"] = "Không rõ"
    df["address"] = df.get("city", "")
    # ❌ KHÔNG lọc bỏ các dòng không có ảnh nữa
    return df
