import streamlit as st
import folium
from streamlit_folium import st_folium
import time

from core.intent_detector import detect_intent
from core.llm_parser import parse_prompt_to_params
from core.llm_composer import compose_plan_response
from core.osm_loader import ensure_poi_dataset
from core.recommender import recommend_pois
from core.itinerary import build_itinerary
from core.weather import get_weather
from core.ui_plan_renderer import render_plan_card

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

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Cá nhân hoá chuyến đi")
    city = st.selectbox("Thành phố:", ["Hồ Chí Minh", "Đà Lạt", "Hà Nội", "Huế", "Đà Nẵng"], index=0)
    if city != "Hồ Chí Minh":
        st.error("🧭 Demo chỉ hỗ trợ thành phố Hồ Chí Minh. Hãy chọn lại.")
        st.stop()

    budget = st.number_input("Ngân sách/ngày (VND)", 100_000, 10_000_000, 1_500_000, 100_000)
    walk_tolerance_km = st.slider("Chịu đi bộ (km/ngày)", 0.5, 15.0, 5.0, 0.5)
    transport = st.selectbox("Phương tiện chính", ["xe máy/ô tô", "đi bộ"], index=0)

    st.markdown("### 🎯 Loại địa điểm")
    category_filter = st.multiselect(
        "Chọn loại bạn muốn khám phá:",
        ["🍽 Ẩm thực", "☕ Cafe", "🎢 Giải trí", "🏛 Tham quan", "🛍 Mua sắm"],
        default=["🍽 Ẩm thực", "🏛 Tham quan"]
    )

    taste = st.multiselect("Khẩu vị ẩm thực", ["Vietnamese", "Japanese", "Italian", "Cafe", "Seafood", "Vegetarian"], default=["Vietnamese", "Cafe"])
    interests = st.multiselect("Sở thích/Hoạt động", ["attraction", "park", "museum", "shopping", "nightlife", "food"], default=["attraction", "food"])
    days = st.number_input("Số ngày hành trình", 1, 10, 2)

st.caption(f"📍 **{city}** • 💸 {budget:,}đ/ngày • 🚶 {walk_tolerance_km}km/ngày")

# --- Cache dữ liệu ---
with st.spinner("Đang tải dữ liệu địa điểm offline..."):
    poi_df = ensure_poi_dataset(city)
weather_now = get_weather(city)


# --- Hiển thị thẻ địa điểm ---
def render_poi_card(p):
    st.markdown(f"### 🏙️ {p.get('name', 'Chưa rõ tên')}")

    def fix_google_img(url):
        if not isinstance(url, str):
            return None
        if "lh3.googleusercontent.com" in url:
            return f"https://images.weserv.nl/?url={url}"
        return url

    images = [fix_google_img(p.get("image_url1")), fix_google_img(p.get("image_url2"))]
    images = [u for u in images if u and u.startswith("http")]

    if len(images) == 2:
        cols = st.columns(2)
        with cols[0]:
            st.image(images[0], use_container_width=True)
        with cols[1]:
            st.image(images[1], use_container_width=True)
    elif len(images) == 1:
        st.image(images[0], use_container_width=True)

    info_parts = []
    if p.get("tag"):
        info_parts.append(f"🏷️ {p['tag']}")
    if p.get("avg_cost"):
        info_parts.append(f"💵 {int(p['avg_cost']):,}đ")
    if p.get("rating"):
        info_parts.append(f"⭐ {p['rating']}")
    if info_parts:
        st.caption(" | ".join(info_parts))

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

    st.markdown("### 🗺️ Bản đồ vị trí các địa điểm")
    coords = [(float(p["lat"]), float(p["lon"])) for p in pois if str(p.get("lat")).replace('.', '', 1).isdigit() and str(p.get("lon")).replace('.', '', 1).isdigit()]
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

    st_folium(fmap, width=900, height=500, key=f"map_{city}")
    st.markdown('</div>', unsafe_allow_html=True)


# --- Các nút chức năng nhanh ---
col_space, col1, col2, col3, col_space2 = st.columns([1, 2, 2, 2, 1])

with col1:
    if st.button("🔎 Gợi ý địa điểm theo cá nhân hoá"):
        if "plan_raw" in st.session_state:
            del st.session_state["plan_raw"]

        category_map = {
            "🍽 Ẩm thực": "food",
            "☕ Cafe": "cafe",
            "🎢 Giải trí": "entertainment",
            "🏛 Tham quan": "attraction",
            "🛍 Mua sắm": "shopping",
        }
        chosen = [category_map[c] for c in category_filter if c in category_map]
        pois = []
        for cat in chosen:
            pois.extend(recommend_pois(
                city=city,
                category=cat,
                user_query="",
                taste_tags=taste,
                activity_tags=interests,
                budget_per_day=budget,
                walk_tolerance_km=walk_tolerance_km,
                weather_desc=weather_now["description"],
            ))
        st.session_state["pois"] = pois
        render_pois(pois)

with col2:
    if st.button("⛅ Xem thời tiết hiện tại"):
        st.info(f"⛅ {city}: {weather_now['description']}, {weather_now['temp']}°C")

with col3:
    if st.button(f"🧭 Lập lịch trình {days} ngày (auto)"):
        # Xóa gợi ý cũ
        if "pois" in st.session_state:
            del st.session_state["pois"]

        params = {
            "city": city,
            "budget_vnd": budget,
            "days": days,
            "taste_tags": taste,
            "activity_tags": interests,
            "walk_tolerance_km": walk_tolerance_km,
            "transport": transport,
        }

        progress_text = st.empty()
        progress_bar = st.progress(0)
        progress_text.text("🔍 Đang tải dữ liệu bản đồ...")

        for pct in range(0, 101, 25):
            time.sleep(0.3)
            progress_bar.progress(pct)
            progress_text.text(f"🧭 Đang tạo lịch trình du lịch... {pct}%")

        plan_raw = build_itinerary(params, poi_df, weather_now)
        st.session_state["plan_raw"] = plan_raw

        progress_bar.empty()
        progress_text.empty()

        st.markdown('<div class="center-container">', unsafe_allow_html=True)
        st.success("✨ Lịch trình đã sẵn sàng! Dưới đây là gợi ý chi tiết:")
        for i, day in enumerate(plan_raw):
            render_plan_card(i, day)
        st.markdown('</div>', unsafe_allow_html=True)


# --- Chat input ---
user_input = st.chat_input("Nhập yêu cầu (vd: 'Gợi ý quán cà phê', 'Lịch trình 3 ngày')")

if user_input:
    st.chat_message("user").write(user_input)
    intent = detect_intent(user_input)

    if intent == "weather":
        st.chat_message("assistant").write(
            f"⛅ Thời tiết {city}: **{weather_now['description']}**, "
            f"{weather_now['temp']}°C, độ ẩm {weather_now.get('humidity', '?')}%."
        )

    elif intent == "lookup":
        if "plan_raw" in st.session_state:
            del st.session_state["plan_raw"]

        category_map = {
            "🍽 Ẩm thực": "food",
            "☕ Cafe": "cafe",
            "🎢 Giải trí": "entertainment",
            "🏛 Tham quan": "attraction",
            "🛍 Mua sắm": "shopping",
        }
        chosen = [category_map[c] for c in category_filter if c in category_map]
        pois = []
        for cat in chosen:
            pois.extend(recommend_pois(
                city=city,
                category=cat,
                user_query=user_input,
                taste_tags=taste,
                activity_tags=interests,
                budget_per_day=budget,
                walk_tolerance_km=walk_tolerance_km,
                weather_desc=weather_now["description"],
            ))
        st.session_state["pois"] = pois
        st.chat_message("assistant").write("🔎 Đây là danh sách địa điểm gợi ý:")
        render_pois(pois)

    elif intent == "plan":
        if "pois" in st.session_state:
            del st.session_state["pois"]

        params = parse_prompt_to_params(user_input)
        params.update({
            "city": city,
            "budget_vnd": budget,
            "days": days,
            "taste_tags": taste,
            "activity_tags": interests,
            "walk_tolerance_km": walk_tolerance_km,
            "transport": transport,
        })

        progress_text = st.empty()
        progress_bar = st.progress(0)
        progress_text.text("🔍 Đang tải dữ liệu bản đồ...")

        for pct in range(0, 101, 25):
            time.sleep(0.3)
            progress_bar.progress(pct)
            progress_text.text(f"🧭 Đang tạo lịch trình du lịch... {pct}%")

        plan_raw = build_itinerary(params, poi_df, weather_now)
        st.session_state["plan_raw"] = plan_raw

        progress_bar.empty()
        progress_text.empty()

        st.markdown('<div class="center-container">', unsafe_allow_html=True)
        st.success("✨ Lịch trình đã sẵn sàng! Dưới đây là gợi ý chi tiết:")
        for i, day in enumerate(plan_raw):
            render_plan_card(i, day)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.chat_message("assistant").write(
            "💡 Gợi ý: *gợi ý địa điểm*, *xem thời tiết*, hoặc *lên lịch trình nhiều ngày*."
        )


# --- Hiển thị POIs hoặc Plan nếu có ---
if "pois" in st.session_state and not user_input:
    render_pois(st.session_state["pois"])
elif "plan_raw" in st.session_state and not user_input:
    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    st.success("✨ Lịch trình đã sẵn sàng! Dưới đây là gợi ý chi tiết:")
    for i, day in enumerate(st.session_state["plan_raw"]):
        render_plan_card(i, day)
    st.markdown('</div>', unsafe_allow_html=True)
