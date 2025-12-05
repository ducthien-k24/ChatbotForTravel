import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
from pathlib import Path
import time

from core.datasource import load_all_categories
from core.recommender import recommend_pois
from core.itinerary import build_itinerary
from core.weather import get_weather
from core.ui_plan_renderer import render_plan_card
from streamlit_js_eval import streamlit_js_eval


# --- Cấu hình trang ---
st.set_page_config(page_title="TravelGPT+ (Offline Demo)", page_icon="🌍", layout="wide")

# --- CSS căn giữa ---
st.markdown("""
<style>
div[data-testid="column"] {
    display: flex;
    justify-content: center;
    align-items: center;
}
.center-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 75%;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

# --- Tiêu đề ---
st.title("🌍 TravelGPT+ — Demo Offline Hồ Chí Minh")


# --- Helper: Lấy danh sách tag (chuẩn hoá từ CSV) ---
def get_tags_for_category(category: str):
    mapping = {
        "food": "pois_hcm_food.csv",
        "cafe": "pois_hcm_cafe.csv",
        "entertainment": "pois_hcm_entertainment.csv",
        "shopping": "pois_hcm_shopping.csv",
        "attraction": "pois_hcm_attraction.csv",
    }
    file_name = mapping.get(category)
    if not file_name:
        return []
    path = Path(f"data/{file_name}")
    if not path.exists():
        return []

    df = pd.read_csv(path)

    # Hợp nhất các biến thể cột tag về 'tag'
    tag_col = None
    for c in ["tag", "tags", "keywords", "Labels", "labels"]:
        if c in df.columns:
            tag_col = c
            break
    if tag_col is None:
        return []

    # Tách thành list: phân tách bằng , ; |
    def split_tags(s: str):
        if not isinstance(s, str):
            return []
        s = s.lower()
        for ch in [";", "|"]:
            s = s.replace(ch, ",")
        return [t.strip() for t in s.split(",") if t.strip()]

    uniq = set()
    for s in df[tag_col].fillna(""):
        uniq.update(split_tags(s))

    return sorted([t for t in uniq if t])


# --- Sidebar cấu hình ---
with st.sidebar:
    st.header("⚙️ Cá nhân hoá chuyến đi")
    mode = st.radio("Chọn chế độ hoạt động:", ["Gợi ý địa điểm", "Lập lịch trình"], index=0)

    city = st.selectbox("Thành phố:", ["Hồ Chí Minh", "Đà Lạt", "Hà Nội", "Huế", "Đà Nẵng"], index=0)
    if city != "Hồ Chí Minh":
        st.error("🧭 Demo chỉ hỗ trợ thành phố Hồ Chí Minh.")
        st.stop()

    budget = st.number_input("💸 Ngân sách/ngày (VND)", 100_000, 10_000_000, 1_500_000, 100_000)
    walk_tolerance_km = st.slider("🚶‍♂️ Chịu đi bộ (km/ngày)", 0.5, 15.0, 5.0, 0.5)
    transport = st.selectbox("🚗 Phương tiện chính", ["Xe máy / Ô tô", "Đi bộ"], index=0)

    # --- Thêm lựa chọn vị trí hiện tại ---
    use_current_location = st.checkbox("📍 Ưu tiên địa điểm gần vị trí hiện tại", value=False)

    if use_current_location:
        st.markdown("#### 📡 Lấy vị trí hiện tại (GPS)")
        if "user_location" not in st.session_state or not st.session_state["user_location"]:

            coords = streamlit_js_eval(
                js_expressions="""
                new Promise((resolve, reject) => {
                    if (navigator.geolocation) {
                        navigator.geolocation.getCurrentPosition(
                            pos => {
                                resolve(pos.coords.latitude + ',' + pos.coords.longitude);
                            },
                            err => {
                                console.log("Geolocation error:", err);
                                resolve(null);
                            }
                        );
                    } else {
                        console.log("Geolocation not supported");
                        resolve(null);
                    }
                }).then(res => res);
                """,
                key="get_location_once",
                want_output=True,
            )

            if coords and isinstance(coords, str) and "," in coords:
                st.session_state["user_location"] = coords.strip()
                st.success(f"📍 Vị trí hiện tại: {coords}")
            else:
                st.info("Đang dò vị trí... (hãy bật quyền truy cập vị trí trong trình duyệt)")
                
        else:
            st.success(f"📍 Vị trí hiện tại: {st.session_state['user_location']}")

    # --- Gợi ý địa điểm ---
    if mode == "Gợi ý địa điểm":
        st.markdown("### 🎯 Chọn loại địa điểm")

        selected_category = st.selectbox(
            "Loại địa điểm:",
            ["food", "cafe", "entertainment", "shopping", "attraction"],
            index=0
        )
        available_tags = get_tags_for_category(selected_category)
        if selected_category != "shopping" and available_tags:
            selected_tags = st.multiselect("🏷️ Chọn tag (nếu muốn):", available_tags)
        else:
            selected_tags = []

    # --- Lập lịch trình ---
    else:
        st.markdown("### 🧭 Cá nhân hoá lịch trình du lịch")

        # 1️⃣ Ăn uống
        food_tags = get_tags_for_category("food")
        selected_food_tags = st.multiselect("🍽️ Bạn thích ăn kiểu nào?", food_tags, default=["vietnamese"])

        # 2️⃣ Shopping
        do_shopping = st.checkbox("🛍️ Có đi shopping không?", value=False)

        # 3️⃣ Entertainment
        do_entertainment = st.checkbox("🎭 Có đi giải trí không?", value=False)
        if do_entertainment:
            entertainment_tags = get_tags_for_category("entertainment")
            selected_entertainment_tags = st.multiselect(
                "🎬 Hoạt động giải trí bạn thích:",
                entertainment_tags,
                default=[]
            )
        else:
            selected_entertainment_tags = []

        # 4️⃣ Attraction
        do_attraction = st.checkbox("🏞️ Có đi tham quan không?", value=True)
        if do_attraction:
            attraction_tags = get_tags_for_category("attraction")
            selected_attraction_tags = st.multiselect(
                "📸 Loại hình tham quan bạn muốn:",
                attraction_tags,
                default=[]
            )
        else:
            selected_attraction_tags = []

        # 5️⃣ Số ngày + số điểm
        days = st.number_input("📅 Số ngày hành trình", 1, 10, 2)
        max_poi_per_day = st.slider("📍 Số địa điểm mỗi ngày", 3, 10, 10, 1)


# --- Hiển thị thông tin tổng quát ---
st.caption(f"📍 **{city}** • 💸 {budget:,}đ/ngày • 🚶 {walk_tolerance_km}km/ngày")

# --- Load dữ liệu POI ---
if "poi_df" not in st.session_state or st.session_state.get("poi_city") != city:
    with st.spinner("🗺️ Đang tải dữ liệu địa điểm (API/CSV adapter)..."):
        st.session_state["poi_df"] = load_all_categories(city, ["food","cafe","entertainment","shopping","attraction"])
        st.session_state["poi_city"] = city
poi_df = st.session_state["poi_df"]

if "weather_now" not in st.session_state or st.session_state.get("weather_city") != city:
    st.session_state["weather_now"] = get_weather(city)
    st.session_state["weather_city"] = city
weather_now = st.session_state["weather_now"]



# --- Các nút chính ---
col_space, col1, col2, col3, col_space2 = st.columns([1, 2, 2, 2, 1])

if mode == "Gợi ý địa điểm":
    with col1:
        if st.button("🔎 Gợi ý địa điểm", key="btn_recommend"):
            user_loc = st.session_state.get("user_location") if use_current_location else None

            pois = recommend_pois(
                city=city,
                category=selected_category,
                user_query="",
                taste_tags=[],
                activity_tags=[],
                budget_per_day=budget,
                walk_tolerance_km=walk_tolerance_km,
                weather_desc=weather_now["description"],
                tag_filter=selected_tags,
                user_location=user_loc   # 🔹 truyền vị trí vào recommender
            ) or []

            want = selected_category.lower()
            filtered = [p for p in pois if want in str(p.get("category", "")).lower()]
            pois = filtered if len(filtered) >= 3 else pois

            st.session_state["pois"] = pois
            st.session_state.pop("plan_raw", None)

else:
    with col3:
        if st.button(f"🧭 Tạo lịch trình {days} ngày", key="btn_plan"):
            params = {
                "city": city,
                "budget_vnd": budget,
                "days": days,
                "walk_tolerance_km": walk_tolerance_km,
                "transport": transport,
                "max_poi_per_day": max_poi_per_day,
                "food_tags": selected_food_tags,
                "do_shopping": do_shopping,
                "do_entertainment": do_entertainment,
                "do_attraction": do_attraction,
                "entertainment_tags": selected_entertainment_tags,
                "attraction_tags": selected_attraction_tags,
            }

            progress = st.progress(0)
            msg = st.empty()
            msg.text("🚀 Đang khởi tạo lịch trình...")

            for pct in range(0, 101, 25):
                time.sleep(0.3)
                progress.progress(pct)
                msg.text(f"🧭 Đang tạo lịch trình du lịch... {pct}%")

            plan_raw = build_itinerary(params, poi_df, weather_now)
            st.session_state["plan_raw"] = plan_raw
            st.session_state.pop("pois", None)

            progress.empty()
            msg.empty()
            st.success("✨ Lịch trình đã được tạo thành công! Kéo xuống để xem chi tiết.")


# --- Hiển thị kết quả ---
def render_poi_card(p):
    st.markdown(f"### 🏙️ {p.get('name', 'Chưa rõ tên')}")
    def fix_img(url):
        if not isinstance(url, str):
            return None
        if "lh3.googleusercontent.com" in url:
            return f"https://images.weserv.nl/?url={url}"
        return url

    imgs = [fix_img(p.get("image_url1")), fix_img(p.get("image_url2"))]
    imgs = [u for u in imgs if u and u.startswith("http")]
    if len(imgs) == 2:
        cols = st.columns(2)
        cols[0].image(imgs[0], width=450)
        cols[1].image(imgs[1], width=450)
    elif len(imgs) == 1:
        st.image(imgs[0], width=600)

    info = []
    if p.get("tag"):
        info.append(f"🏷️ {p['tag']}")
    if p.get("avg_cost"):
        try:
            info.append(f"💵 {int(p['avg_cost']):,}đ")
        except Exception:
            pass
    if p.get("rating"):
        info.append(f"⭐ {p['rating']}")
        
    if p.get("distance_km") is not None:
        try:
            info.append(f"📏 {float(p['distance_km']):.2f} km từ vị trí của bạn")
        except Exception:
            pass
        
    if info:
        st.caption(" | ".join(info))

    if p.get("description"):
        st.write(p["description"])
    if p.get("address"):
        st.info(f"📍 {p['address']}")
    st.divider()


def render_pois(pois):
    if not pois:
        st.warning("Không tìm thấy địa điểm phù hợp.")
        return

    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    st.subheader(f"🎯 Gợi ý {len(pois)} địa điểm phù hợp:")

    for p in pois:
        render_poi_card(p)

    coords = [(float(p["lat"]), float(p["lon"])) for p in pois
              if pd.notna(p.get("lat")) and pd.notna(p.get("lon"))]
    if not coords:
        st.warning("⚠️ Không thể hiển thị bản đồ vì thiếu tọa độ hợp lệ.")
        return

    lat_center = sum(lat for lat, _ in coords) / len(coords)
    lon_center = sum(lon for _, lon in coords) / len(coords)
    fmap = folium.Map(location=[lat_center, lon_center], zoom_start=13)

    for p in pois:
        try:
            lat, lon = float(p["lat"]), float(p["lon"])
            folium.Marker([lat, lon], popup=p["name"], tooltip=p["name"]).add_to(fmap)
        except Exception:
            continue

    st_folium(fmap, width=900, height=500, key=f"map_{city}_{int(time.time())}")
    st.markdown('</div>', unsafe_allow_html=True)


if "pois" in st.session_state:
    render_pois(st.session_state["pois"])
elif "plan_raw" in st.session_state:
    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    st.success("✨ Lịch trình đã sẵn sàng! Dưới đây là gợi ý chi tiết:")
    for i, day in enumerate(st.session_state["plan_raw"]):
        render_plan_card(i, day)
    st.markdown('</div>', unsafe_allow_html=True)
