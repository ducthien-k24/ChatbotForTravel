import streamlit as st
from core.intent_detector import detect_intent
from core.llm_parser import parse_prompt_to_params
from core.llm_composer import compose_plan_response
from core.osm_loader import ensure_poi_dataset
from core.recommender import recommend_pois
from core.itinerary import build_itinerary
from core.weather import get_weather

st.set_page_config(page_title="TravelGPT+", page_icon="🌍", layout="wide")
st.title("🌍 TravelGPT+ — Trợ lý du lịch toàn diện")

# --- Cá nhân hoá & thành phố ---
with st.sidebar:
    st.header("⚙️ Cấu hình & Cá nhân hoá")
    city = st.selectbox("Thành phố:", ["Hồ Chí Minh", "Đà Lạt", "Hà Nội", "Huế", "Đà Nẵng"], index=0)
    budget = st.number_input("Ngân sách/ngày (VND)", min_value=100_000, max_value=10_000_000, value=1_500_000, step=100_000)
    walk_tolerance_km = st.slider("Chịu đi bộ (km/ngày)", 0.5, 15.0, 5.0, 0.5)
    transport = st.selectbox("Phương tiện chính", ["xe máy/ô tô", "đi bộ"], index=0)
    taste = st.multiselect("Khẩu vị ẩm thực", ["Vietnamese", "Japanese", "Italian", "Cafe", "Seafood", "Vegetarian"], default=["Vietnamese","Cafe"])
    interests = st.multiselect("Sở thích/Hoạt động", ["attraction", "park", "museum", "shopping", "nightlife", "food"], default=["attraction","food"])
    days = st.number_input("Số ngày hành trình", min_value=1, max_value=10, value=2)

st.caption(f"📍 Thành phố hiện tại: **{city}** • 💸 Ngân sách: **{budget:,}đ/ngày** • 🚶 Tolerate: **{walk_tolerance_km}km/ngày**")

# --- Bảo đảm có dữ liệu POI (OSM) & thời tiết ---
with st.spinner("Đang tải dữ liệu POI từ OpenStreetMap (cache nếu có)…"):
    poi_df = ensure_poi_dataset(city)

weather_now = get_weather(city)  # dict: {city, temp, humidity, description} (fallback nếu thiếu API)

# --- Ô chat nhập tự do ---
user_input = st.chat_input("Nhập yêu cầu (ví dụ: 'Gợi ý địa điểm tham quan', 'Lên lịch trình 3 ngày', 'Thời tiết hôm nay')")

def _render_pois(pois):
    if not pois:
        st.warning("Không tìm thấy địa điểm phù hợp.")
        return
    st.write(f"**Gợi ý {len(pois)} địa điểm phù hợp:**")
    for p in pois:
        name = str(p.get("name", ""))
        category = str(p.get("category", ""))
        cost = int(p.get("avg_cost", 0))
        desc = str(p.get("description", ""))[:120]  # ✅ ép kiểu để tránh lỗi NaN
        lat = round(float(p.get("lat", 0)), 6)
        lon = round(float(p.get("lon", 0)), 6)
        st.markdown(
            f"- **{name}** · *{category}* · {cost:,}đ  \n"
            f"  {desc}…  \n"
            f"  ↳ (lat: {lat}, lon: {lon})"
        )


if user_input:
    st.chat_message("user").write(user_input)
    with st.spinner("Đang xử lý…"):
        intent = detect_intent(user_input)
        if intent == "weather":
            st.chat_message("assistant").write(
                f"⛅ Thời tiết {city}: **{weather_now['description']}**, {weather_now['temp']}°C, "
                f"độ ẩm {weather_now.get('humidity','?')}%."
            )
        elif intent == "lookup":
            pois = recommend_pois(
                city=city,
                poi_df=poi_df,
                user_query=user_input,
                taste_tags=taste,
                activity_tags=interests,
                budget_per_day=budget,
                walk_tolerance_km=walk_tolerance_km
            )
            st.chat_message("assistant").write("🔎 Mình đã lọc theo sở thích & cá nhân hoá của bạn:")
            _render_pois(pois)
        elif intent == "plan":
            # người dùng nói tự nhiên → trích tham số
            params = parse_prompt_to_params(user_input)
            # ghi đè theo sidebar (vì bạn muốn app điều khiển)
            params.update({
                "city": city,
                "budget_vnd": budget,
                "days": days,
                "taste_tags": taste,
                "activity_tags": interests,
                "walk_tolerance_km": walk_tolerance_km,
                "transport": transport
            })
            plan_raw = build_itinerary(params, poi_df, weather_now)  # tính tuyến theo Dijkstra/MST + chấm điểm
            plan_text = compose_plan_response(plan_raw, params)      # LLM “đánh bóng” (fallback nếu thiếu API)
            st.chat_message("assistant").write(plan_text)
        else:
            # chat tự nhiên, hoặc ý định mơ hồ
            st.chat_message("assistant").write(
                "Bạn có thể yêu cầu: *gợi ý địa điểm*, *xem thời tiết*, hoặc *lên lịch trình nhiều ngày*.\n"
                "Ví dụ: “Lên lịch trình Đà Lạt 3 ngày với hoạt động ngoài trời và ít đi bộ”."
            )

# Khu vực chạy nhanh KHÔNG CHAT: 3 nút demo chức năng
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔎 Gợi ý địa điểm theo cá nhân hoá"):
        pois = recommend_pois(
            city=city, poi_df=poi_df, user_query="",
            taste_tags=taste, activity_tags=interests,
            budget_per_day=budget, walk_tolerance_km=walk_tolerance_km
        )
        _render_pois(pois)
with col2:
    if st.button("⛅ Xem thời tiết hiện tại"):
        st.info(f"⛅ {city}: {weather_now['description']}, {weather_now['temp']}°C")
with col3:
    if st.button("🧭 Lập lịch trình {days} ngày (auto)"):
        params = {
            "city": city, "budget_vnd": budget, "days": days,
            "taste_tags": taste, "activity_tags": interests,
            "walk_tolerance_km": walk_tolerance_km, "transport": transport
        }
        plan_raw = build_itinerary(params, poi_df, weather_now)
        st.write(compose_plan_response(plan_raw, params))
