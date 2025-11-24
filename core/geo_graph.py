import os
import math
import osmnx as ox
import networkx as nx
import numpy as np

# --- Cấu hình cache ---
ox.settings.use_cache = True
ox.settings.cache_folder = "data/osmnx_cache"
ox.settings.log_console = True


def haversine_dist(lat1, lon1, lat2, lon2):
    """Tính khoảng cách địa lý (m) giữa 2 tọa độ lat/lon (haversine)."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _get_graph_cache_path(city: str) -> str:
    os.makedirs("data", exist_ok=True)
    return f"data/{city.lower().replace(' ', '_')}_graph.graphml"


def road_graph_for_city(city: str) -> nx.MultiDiGraph:
    """
    Tải graph đường (drive) cho city và cache lại để lần sau load nhanh hơn.
    """
    cache_path = _get_graph_cache_path(city)
    if os.path.exists(cache_path):
        print(f"⚡ Đang tải graph từ cache: {cache_path}")
        return ox.load_graphml(cache_path)

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
        G = ox.graph_from_bbox(
            north=north, south=south, east=east, west=west,
            network_type="drive", simplify=True
        )
    else:
        G = ox.graph_from_place(city + ", Vietnam", network_type="drive", simplify=True)

    # Đảm bảo cạnh nào cũng có length
    for u, v, k, data in G.edges(keys=True, data=True):
        if "length" not in data:
            if "geometry" in data:
                data["length"] = data["geometry"].length
            else:
                x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
                x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
                data["length"] = haversine_dist(y1, x1, y2, x2)

    ox.save_graphml(G, cache_path)
    print(f"💾 Graph được lưu cache tại: {cache_path}")
    return G


def shortest_distance_km(G: nx.MultiDiGraph, src, dst) -> float:
    """Tính khoảng cách ngắn nhất (km) giữa 2 tọa độ (lat, lon), có kiểm tra lỗi."""
    try:
        lat1, lon1 = float(src[0]), float(src[1])
        lat2, lon2 = float(dst[0]), float(dst[1])
        if any(np.isnan([lat1, lon1, lat2, lon2])):
            raise ValueError("Toạ độ NaN")
    except Exception as e:
        print(f"❌ Lỗi toạ độ không hợp lệ: {e}")
        return float("inf")

    try:
        u = ox.distance.nearest_nodes(G, lon1, lat1)
        v = ox.distance.nearest_nodes(G, lon2, lat2)
        length_m = nx.shortest_path_length(G, u, v, weight="length", method="dijkstra")
        return length_m / 1000.0
    except nx.NetworkXNoPath:
        return float("inf")
    except Exception as e:
        print(f"❌ Lỗi khi tính khoảng cách: {e}")
        return float("inf")
