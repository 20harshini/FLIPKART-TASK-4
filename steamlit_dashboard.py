import streamlit as st
import pandas as pd
import joblib

st.title("🚚 AI Logistics Dashboard")

df = pd.read_excel("data/logistics_smart_500.xlsx")



st.metric("Total Deliveries", len(df))
st.metric("Delayed Deliveries", df["delay"].sum())

model = joblib.load("models/XGBoost.pkl")

st.subheader("Live Prediction")
if st.button("Predict Delay"):
    sample =sample = [[
    17.38, 78.48, 17.39, 78.50,   # src/dest lat lon (4)
    12.0,                         # distance_km
    1,                            # vehicle_type (encoded)
    5,                            # driver_id
    8.0,                          # package_weight
    2,                            # traffic_level (encoded)
    1,                            # weather (encoded)
    0,                            # road_type (encoded)
    14,                           # order_hour
    16                            # scheduled_hour
]]

    prob = model.predict_proba(sample)[0][1]
    st.success(f"Delay Probability: {round(prob*100,2)}%")
      # ✅ Route optimization

