# core/geo_graph.py — dùng osmnx offline 100%
import math
from pathlib import Path
from typing import Optional, Tuple

import osmnx as ox            # bắt buộc có osmnx
import networkx as nx         # bắt buộc có networkx


# Map city -> file graphml local (đổi nếu bạn dùng tên khác)
_OFFLINE_GRAPH_FILES = {
    "hồ chí minh": "hồ_chí_minh_graph.graphml",
    "ho chi minh": "hồ_chí_minh_graph.graphml",
    "hcm": "hồ_chí_minh_graph.graphml",
}

# Cache trong RAM để không đọc file lặp lại
_GRAPH_CACHE: dict[str, Optional[nx.MultiDiGraph]] = {}


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Khoảng cách đường thẳng (m)."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _load_graphml_from_disk(city: str) -> Optional[nx.MultiDiGraph]:
    """
    Chỉ đọc file .graphml trong ./data, KHÔNG gọi Overpass.
    Ưu tiên theo map city -> filename; nếu không khớp, lấy file .graphml đầu tiên.
    """
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    key = (city or "").strip().lower()
    fname = _OFFLINE_GRAPH_FILES.get(key)

    if not fname:
        first = next(data_dir.glob("*.graphml"), None)
        fname = first.name if first else None
    if not fname:
        return None

    path = data_dir / fname
    if not path.exists():
        return None

    # osmnx load (local), không qua mạng
    G = ox.load_graphml(path)
    # ép về MultiDiGraph nếu cần
    if not isinstance(G, nx.MultiDiGraph):
        G = nx.MultiDiGraph(G)

    # đảm bảo mỗi cạnh có 'length' (m)
    for u, v, k, data in G.edges(keys=True, data=True):
        if "length" not in data:
            if "geometry" in data and data["geometry"] is not None:
                # geometry là Linestring với CRS WGS84 → ước lượng theo đoạn thẳng các điểm
                try:
                    coords = list(data["geometry"].coords)
                    dist = 0.0
                    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
                        dist += _haversine_m(y1, x1, y2, x2)
                    data["length"] = dist
                except Exception:
                    x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
                    x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
                    data["length"] = _haversine_m(y1, x1, y2, x2)
            else:
                x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
                x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
                data["length"] = _haversine_m(y1, x1, y2, x2)

    return G


def road_graph_for_city(city: str) -> Optional[nx.MultiDiGraph]:
    """
    Trả về graph đường từ file .graphml local (offline).
    """
    key = (city or "").strip().lower()
    if key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]
    G = _load_graphml_from_disk(city)
    _GRAPH_CACHE[key] = G
    return G


def _nearest_node(G: nx.MultiDiGraph, lat: float, lon: float):
    """
    nearest_nodes của osmnx — chạy local hoàn toàn (không gọi mạng).
    """
    return ox.distance.nearest_nodes(G, lon, lat)


def shortest_distance_km(G: Optional[nx.MultiDiGraph],
                         src: Tuple[float, float],
                         dst: Tuple[float, float]) -> float:
    """
    Khoảng cách ngắn nhất (km) theo graph nếu có; nếu thiếu dữ liệu → fallback Haversine.
    """
    try:
        lat1, lon1 = float(src[0]), float(src[1])
        lat2, lon2 = float(dst[0]), float(dst[1])
    except Exception:
        # toạ độ không hợp lệ → 0 km
        return 0.0

    if G is None:
        return _haversine_m(lat1, lon1, lat2, lon2) / 1000.0

    try:
        u = _nearest_node(G, lat1, lon1)
        v = _nearest_node(G, lat2, lon2)
        if u is None or v is None:
            return _haversine_m(lat1, lon1, lat2, lon2) / 1000.0

        try:
            length_m = nx.shortest_path_length(G, u, v, weight="length", method="dijkstra")
            return float(length_m) / 1000.0
        except nx.NetworkXNoPath:
            return _haversine_m(lat1, lon1, lat2, lon2) / 1000.0
        except Exception:
            # không có weight → ước lượng theo số cạnh (~80m/cạnh)
            steps = nx.shortest_path_length(G, u, v, weight=None)
            return float(steps) * 0.08
    except Exception:
        return _haversine_m(lat1, lon1, lat2, lon2) / 1000.0