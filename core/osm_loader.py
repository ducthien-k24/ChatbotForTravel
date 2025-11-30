# core/osm_loader.py
"""
Thin facade để lấy POI cho app.
- Không gọi Overpass / OSM.
- Uỷ quyền toàn bộ cho core.datasource (CSV hoặc API tuỳ ENV).
ENV:
  DATA_PROVIDER = "api" | "csv"
  API_BASE_URL, API_TOKEN (nếu DATA_PROVIDER=api)
"""

from typing import List
import pandas as pd

# Adapter chính: CSV/API -> DataFrame chuẩn
from .datasource import (
    load_all_categories,
    load_category_df,
)

# Danh sách category chuẩn mà app dùng
_ALL_CATEGORIES: List[str] = ["food", "cafe", "entertainment", "shopping", "attraction"]


def ensure_poi_dataset(city: str) -> pd.DataFrame:
    """
    Trả về DataFrame gộp đủ 5 category cho thành phố `city`.
    Datasource sẽ tự chọn API hay CSV theo ENV (DATA_PROVIDER).
    """
    return load_all_categories(city, _ALL_CATEGORIES)


def load_category_data(city: str, category: str, base_dir: str = "data/") -> pd.DataFrame:
    """
    Trả về DataFrame 1 category cho `city` (để recommender).
    Vẫn gọi qua datasource để tôn trọng ENV và chuẩn schema.
    """
    return load_category_df(city, category)


# (Tuỳ chọn) Backward-compat: nếu code cũ còn gọi hàm này ở nơi khác
def load_local_pois(data_dir: str = "data/") -> pd.DataFrame:
    """
    GIỮ TƯƠNG THÍCH: thay vì tự đọc CSV ở đây, ta ủy quyền lại cho datasource.
    Mặc định lấy đủ 5 category cho 'Ho Chi Minh'.
    """
    return load_all_categories("Ho Chi Minh", _ALL_CATEGORIES)
