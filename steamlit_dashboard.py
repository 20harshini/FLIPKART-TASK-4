import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from route_optimization import optimize_route

st.set_page_config(page_title="AI Logistics Dashboard", layout="wide")
st.title("🚚 AI Logistics Dashboard")

# ===========================
# Load Dataset
# ===========================
df = pd.read_excel("data/logistics_smart_500.xlsx")

# ===========================
# KPIs
# ===========================
total_deliveries = len(df)
delayed_deliveries = int(df["delay"].sum())
on_time_deliveries = total_deliveries - delayed_deliveries

col1, col2, col3 = st.columns(3)
col1.metric("Total Deliveries", total_deliveries)
col2.metric("Delayed Deliveries", delayed_deliveries)
col3.metric("On-Time Deliveries", on_time_deliveries)

st.divider()

# ===========================
# Driver Wise Performance
# ===========================
st.subheader("👨‍✈️ Driver-wise Performance Metrics")

driver_stats = df.groupby("driver_id").agg(
    total_deliveries=("delivery_id", "count"),
    delayed=("delay", "sum")
).reset_index()

driver_stats["delay_rate(%)"] = round((driver_stats["delayed"] / driver_stats["total_deliveries"]) * 100, 2)

st.dataframe(driver_stats.sort_values("delay_rate(%)", ascending=False), use_container_width=True)

st.divider()

# ===========================
# Traffic KPI + Heatmap
# ===========================
st.subheader("🚦 Traffic Level Overview")

traffic_counts = df["traffic_level"].value_counts().reset_index()
traffic_counts.columns = ["traffic_level", "count"]

col4, col5 = st.columns(2)

with col4:
    st.bar_chart(traffic_counts.set_index("traffic_level"))

with col5:
    st.write("### 🔥 Traffic Heatmap (Map)")
    
    # Heatmap Data: use delivery points weighted by traffic
    traffic_weight = {"Low": 1, "Medium": 2, "High": 3}

    heat_data = [
        [row["dest_lat"], row["dest_lon"], traffic_weight[row["traffic_level"]]]
        for _, row in df.sample(200, random_state=42).iterrows()
    ]

    heat_map = folium.Map(location=[17.3850, 78.4867], zoom_start=11)
    HeatMap(heat_data, radius=12).add_to(heat_map)

    st_folium(heat_map, width=550, height=400)

st.divider()

# ===========================
# Real-Time Delay Prediction + Route
# ===========================
st.subheader("⚡ Real-Time Delay Prediction + Optimized Route")

model = joblib.load("models/xgboost.pkl")

colA, colB = st.columns(2)

with colA:
    st.write("### Enter Delivery Details")

    distance_km = st.slider("Distance (km)", 2.0, 40.0, 12.0)
    package_weight = st.slider("Package Weight (kg)", 0.5, 30.0, 8.0)
    driver_id = st.selectbox("Driver ID", sorted(df["driver_id"].unique()))

    vehicle_type = st.selectbox("Vehicle Type (Encoded)", [0, 1, 2])
    traffic_level = st.selectbox("Traffic Level (Encoded)", [0, 1, 2])
    weather = st.selectbox("Weather (Encoded)", [0, 1, 2])
    road_type = st.selectbox("Road Type (Encoded)", [0, 1, 2])

    order_hour = st.slider("Order Hour", 0, 23, 14)
    scheduled_hour = st.slider("Scheduled Hour", 0, 23, 16)

    # source + destination (fixed sample Hyderabad)
    src_lat, src_lon = 17.38, 78.48
    dest_lat, dest_lon = 17.39, 78.50

with colB:
    st.write("### Prediction Output")

    if st.button("Predict Delay + Show Route"):
        sample = np.array([[
            src_lat, src_lon, dest_lat, dest_lon,
            distance_km,
            vehicle_type,
            driver_id,
            package_weight,
            traffic_level,
            weather,
            road_type,
            order_hour,
            scheduled_hour
        ]])

        delay_prob = model.predict_proba(sample)[0][1]
        st.success(f"✅ Delay Probability: {round(delay_prob*100, 2)}%")

        # Route Optimization
        route, dist = optimize_route()
        st.info(f"🛣 Optimized Route: {route}")
        st.info(f"📏 Estimated Distance: {dist}")

        # ===========================
        # Map Showing Optimized Route
        # ===========================
        st.write("### 🗺 Map Showing Optimized Route")

        route_map = folium.Map(location=[src_lat, src_lon], zoom_start=12)

        folium.Marker([src_lat, src_lon], popup="Source", icon=folium.Icon(color="green")).add_to(route_map)
        folium.Marker([dest_lat, dest_lon], popup="Destination", icon=folium.Icon(color="red")).add_to(route_map)

        # Draw line between source and destination (route visual demo)
        folium.PolyLine(locations=[[src_lat, src_lon], [dest_lat, dest_lon]], weight=5).add_to(route_map)

        st_folium(route_map, width=700, height=450)
