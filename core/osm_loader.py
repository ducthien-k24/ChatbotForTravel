import os
import osmnx as ox
import pandas as pd

# --- Bật cache ---
ox.settings.use_cache = True
ox.settings.cache_folder = "data/osmnx_cache"
ox.settings.log_console = True


def _download_osm_pois(city: str) -> pd.DataFrame:
    tags = {
        "amenity": ["restaurant", "cafe", "bar", "fast_food"],
        "tourism": ["attraction", "museum", "hotel", "guest_house", "hostel", "gallery"],
        "leisure": ["park", "garden"],
    }

    bbox_by_city = {
        "ho chi minh": (10.85, 10.70, 106.83, 106.63),
        "đà lạt": (11.97, 11.90, 108.47, 108.40),
        "hà nội": (21.08, 20.95, 105.90, 105.75),
        "đà nẵng": (16.10, 15.90, 108.30, 108.10),
        "huế": (16.50, 16.42, 107.63, 107.52),
        "nha trang": (12.28, 12.18, 109.22, 109.12),
    }

    city_key = city.lower().strip()
    if city_key in bbox_by_city:
        north, south, east, west = bbox_by_city[city_key]
        gdf = ox.features_from_bbox(
            north=north,
            south=south,
            east=east,
            west=west,
            tags=tags
        )
    else:
        gdf = ox.features_from_place(city + ", Vietnam", tags)

    if gdf.empty:
        raise ValueError(f"Không tìm thấy POI cho {city}")

    gdf = gdf.to_crs(epsg=4326)
    gdf["lat"] = gdf.geometry.centroid.y
    gdf["lon"] = gdf.geometry.centroid.x

    def detect_category(row):
        for key in ["amenity", "tourism", "leisure"]:
            if key in row and pd.notna(row[key]):
                return str(row[key])
        return "other"

    gdf["category"] = gdf.apply(detect_category, axis=1)
    df = gdf[["name", "category", "lat", "lon"]].dropna(subset=["name"])
    df["city"] = city
    df["avg_cost"] = 100000
    df["description"] = df["category"].map({
        "restaurant": "Nhà hàng nổi tiếng với ẩm thực địa phương.",
        "cafe": "Quán cà phê yên tĩnh, thích hợp để thư giãn.",
        "hotel": "Khách sạn thuận tiện cho du khách.",
        "park": "Không gian xanh mát, lý tưởng để đi dạo.",
        "museum": "Nơi lưu giữ nhiều giá trị văn hóa, lịch sử.",
    }).fillna("Địa điểm du lịch được yêu thích.")

    return df


def ensure_poi_dataset(city: str) -> pd.DataFrame:
    """Tự động cache dataset POI theo thành phố."""
    os.makedirs("data", exist_ok=True)
    cache_path = f"data/pois_cache_{city.lower().replace(' ', '_')}.csv"
    if os.path.exists(cache_path):
        print(f"⚡ Đang load dữ liệu POI từ cache: {cache_path}")
        return pd.read_csv(cache_path)
    df = _download_osm_pois(city)
    df.to_csv(cache_path, index=False)
    print(f"💾 Đã lưu cache POI: {cache_path}")
    return df
