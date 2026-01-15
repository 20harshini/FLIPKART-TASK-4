import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("data/logistics_data.csv")

# Time features
df["order_hour"] = pd.to_datetime(df["order_time"]).dt.hour
df["scheduled_hour"] = pd.to_datetime(df["scheduled_time"]).dt.hour

# Encode categorical features
le = LabelEncoder()
for col in ["vehicle_type", "traffic_level", "weather", "road_type"]:
    df[col] = le.fit_transform(df[col])

# Drop unused columns
df.drop(["delivery_id", "order_time", "scheduled_time", "actual_time"], axis=1, inplace=True)

X = df.drop("delay", axis=1)
y = df["delay"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Preprocessing completed")
