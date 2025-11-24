import os
import pandas as pd
import glob

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
    Với demo offline:
    - Nếu người dùng chọn Hồ Chí Minh → load CSV local
    - Thành phố khác → cảnh báo demo chỉ hỗ trợ HCM
    """
    city_key = city.lower().strip()
    if city_key not in ["ho chi minh", "hồ chí minh", "hcm"]:
        raise ValueError("🧭 Demo chỉ hỗ trợ thành phố Hồ Chí Minh.")
    return load_local_pois("data/")
