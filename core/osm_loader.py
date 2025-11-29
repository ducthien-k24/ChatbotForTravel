import os
import pandas as pd
import glob
from pathlib import Path

# ==============================
#  BẢN CHẠY OFFLINE DEMO (HCM)
# ==============================

def load_local_pois(data_dir: str = "data/") -> pd.DataFrame:
    """
    Đọc toàn bộ dữ liệu POI từ 5 file CSV offline.
    Dùng cho demo Hồ Chí Minh, không gọi Overpass/OSM.
    """
    pattern = os.path.join(data_dir, "pois_hcm_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file CSV nào trong {data_dir}")

    dfs = []
    for path in files:
        try:
            df = pd.read_csv(path)
            df["source_file"] = os.path.basename(path)
            # Chuẩn hoá cột
            if "tag" not in df.columns:
                df["tag"] = "unknown"
            if "lat" not in df.columns or "lon" not in df.columns:
                continue  # bỏ file không hợp lệ
            # Điền city cố định là Hồ Chí Minh
            df["city"] = "Ho Chi Minh"
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc {path}: {e}")

    if not dfs:
        raise ValueError("Không load được dữ liệu POI nào!")

    all_pois = pd.concat(dfs, ignore_index=True)
    # Giữ các cột cần thiết cho planner/recommender
    keep_cols = [
        c for c in [
            "name", "tag", "description", "lat", "lon", "avg_cost",
            "rating", "reviews", "address", "opening_hours",
            "image_url1", "image_url2", "city", "source_file"
        ] if c in all_pois.columns
    ]
    return all_pois[keep_cols].dropna(subset=["name", "lat", "lon"])

def ensure_poi_dataset(city: str) -> pd.DataFrame:
    """
    Load dữ liệu POI offline cho thành phố được chọn.
    Demo hiện tại chỉ hỗ trợ Hồ Chí Minh.

    - Tự động thêm cột `category` cho từng loại.
    - Gộp 5 file CSV trong thư mục data/.
    """
    city_key = city.lower().strip()
    if city_key not in ["ho chi minh", "hồ chí minh", "hcm", "ho chi minh city"]:
        raise ValueError("🧭 Demo only supports Ho Chi Minh City.")

    data_dir = Path("data")
    mapping = {
        "food": "pois_hcm_food.csv",
        "cafe": "pois_hcm_cafe.csv",
        "entertainment": "pois_hcm_entertainment.csv",
        "shopping": "pois_hcm_shopping.csv",
        "attraction": "pois_hcm_attraction.csv",
    }

    frames = []
    for cat, filename in mapping.items():
        fpath = data_dir / filename
        if not fpath.exists():
            print(f"⚠️ Missing file: {fpath}")
            continue

        try:
            df = pd.read_csv(fpath)
            df["category"] = cat  # 👈 thêm cột để itinerary nhận biết loại
            frames.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {fpath}: {e}")

    if not frames:
        raise FileNotFoundError("❌ No POI CSV files found in /data directory.")

    all_pois = pd.concat(frames, ignore_index=True)
    all_pois.drop_duplicates(subset="name", inplace=True)
    return all_pois



def load_category_data(city: str, category: str, base_dir="data/") -> pd.DataFrame:
    """
    Tải dữ liệu offline tương ứng với category người dùng chọn (food, cafe, shopping, attraction, entertainment...).
    """
    category = category.lower()
    mapping = {
        "food": "pois_hcm_food.csv",
        "cafe": "pois_hcm_cafe.csv",
        "entertainment": "pois_hcm_entertainment.csv",
        "shopping": "pois_hcm_shopping.csv",
        "attraction": "pois_hcm_attraction.csv",
    }

    file_name = mapping.get(category)
    if not file_name:
        raise ValueError(f"Không có dữ liệu cho category: {category}")

    path = Path(base_dir) / file_name
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    df = pd.read_csv(path)
    df["city"] = city
    df["source_file"] = file_name

    # Chuẩn hóa cột
    if "category" not in df.columns:
        df["category"] = category
    df["category"] = df["category"].fillna(category).astype(str).str.lower()

    return df
