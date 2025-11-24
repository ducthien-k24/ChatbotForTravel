import pandas as pd
import os

files = [
    "data/pois_hcm_food.csv",
    "data/pois_hcm_cafe.csv",
    "data/pois_hcm_entertainment.csv",
    "data/pois_hcm_shopping.csv",
    "data/pois_hcm_attraction.csv"
]

def fix_link(url):
    if isinstance(url, str) and "lh3.googleusercontent.com" in url:
        # Bỏ "https://" để chèn vào proxy
        core = url.split("//")[-1]
        return f"https://images.weserv.nl/?url={core}"
    return url

for f in files:
    if not os.path.exists(f):
        print(f"⚠️ Không tìm thấy {f}")
        continue
    df = pd.read_csv(f)
    for col in ["image_url1", "image_url2"]:
        if col in df.columns:
            df[col] = df[col].apply(fix_link)
    df.to_csv(f, index=False)
    print(f"✅ Đã cập nhật link ảnh trong: {f}")
