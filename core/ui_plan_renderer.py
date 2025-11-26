import streamlit as st
import folium
from streamlit_folium import st_folium

# CSS hiệu ứng + style card
st.markdown("""
<style>
.poi-card {
    background-color: #f9f9f9;
    border-radius: 15px;
    padding: 15px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: transform 0.2s ease-in-out;
}
.poi-card:hover {
    transform: scale(1.01);
}
.arrow-text {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin: 6px 0;
}
[data-testid="stModal"] {
    animation: fadeIn 0.4s ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)


# --- Hàm xử lý link ảnh Google Maps ---
def fix_google_img(url: str):
    """Chuyển link Google Maps (lh3.googleusercontent) thành ảnh có thể load được."""
    if not isinstance(url, str) or not url:
        return None
    if "lh3.googleusercontent.com" in url:
        return f"https://images.weserv.nl/?url={url}"
    return url


def render_plan_card(day_idx, plan_day):
    """Hiển thị 1 ngày hành trình dạng thẻ đẹp + ảnh + bản đồ."""
    st.markdown(f"## 🗓️ Ngày {day_idx+1}: {plan_day.get('title', 'Khám phá')}")
    st.caption(f"🌤 {plan_day.get('weather', 'Không rõ')} • 🚗 {plan_day.get('distance', 0):.2f} km tổng quãng đường")
    st.divider()

    pois = plan_day.get("pois", [])
    if not pois:
        st.warning("Không có địa điểm nào trong ngày này.")
        return

    for i, poi in enumerate(pois):
        with st.container():
            st.markdown('<div class="poi-card">', unsafe_allow_html=True)
            cols = st.columns([1, 2])
            with cols[0]:
                # Ảnh
                raw_img = poi.get("image_url1") or poi.get("image_url2")
                img = fix_google_img(raw_img) or "https://via.placeholder.com/300x200?text=No+Image"
                st.image(img, width="stretch")

                # Chi tiết địa điểm
                with st.expander("🔍 Xem chi tiết"):
                    st.image(img, width="stretch")
                    st.markdown(f"### 🏙️ {poi.get('name', 'Địa điểm chưa rõ')}")
                    st.markdown(f"**📍 Địa chỉ:** {poi.get('address', 'Không rõ')}")
                    st.markdown(f"**💰 Giá trung bình:** {int(poi.get('avg_cost', 0)):,} VND")
                    st.markdown(f"**⭐ Đánh giá:** {poi.get('rating', 'N/A')}")
                    st.markdown(f"**🕒 Thời gian:** {poi.get('time', 'Không có')}")
                    desc = poi.get('description', '')
                    if desc:
                        st.markdown("### 📝 Mô tả chi tiết")
                        st.write(desc)

            with cols[1]:
                st.markdown(f"### 🏙️ {poi.get('name', 'Địa điểm chưa rõ')}")
                st.caption(f"📍 {poi.get('address', 'Không rõ địa chỉ')}")
                st.caption(f"💰 {int(poi.get('avg_cost', 0)):,} VND • ⭐ {poi.get('rating', 'N/A')}")
                desc = poi.get('description', '')
                if desc:
                    short = desc[:150] + "..." if len(desc) > 150 else desc
                    st.write(short)
            st.markdown('</div>', unsafe_allow_html=True)

        # Hiển thị khoảng cách giữa các điểm
        if i < len(pois) - 1:
            next_km = poi.get('next_distance_km', '?')
            st.markdown(f"<div class='arrow-text'>⬇️ Cách {next_km} km ⬇️</div>", unsafe_allow_html=True)

    st.divider()

    # Mini map trong ngày
    valid_coords = [p for p in pois if isinstance(p.get('lat'), (int, float)) and isinstance(p.get('lon'), (int, float))]
    if len(valid_coords) >= 2:
        lat_center = sum(p['lat'] for p in valid_coords) / len(valid_coords)
        lon_center = sum(p['lon'] for p in valid_coords) / len(valid_coords)
        fmap = folium.Map(location=[lat_center, lon_center], zoom_start=13)
        for p in valid_coords:
            folium.Marker([p['lat'], p['lon']], tooltip=p['name']).add_to(fmap)
        st_folium(fmap, width=850, height=400, key=f"map_day_{day_idx}")
