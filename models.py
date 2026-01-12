import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Load dataset
df = pd.read_excel("data/logistics_data_500.xlsx")

# Feature engineering
df["order_hour"] = pd.to_datetime(df["order_time"]).dt.hour
df["scheduled_hour"] = pd.to_datetime(df["scheduled_time"]).dt.hour

# Encode categorical features
le = LabelEncoder()
for col in ["vehicle_type", "traffic_level", "weather", "road_type"]:
    df[col] = le.fit_transform(df[col])

# Prepare features and target
X = df.drop(
    ["delivery_id", "order_time", "scheduled_time", "actual_time", "delay"],
    axis=1
)
y = df["delay"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Initialize models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(random_state=42)
xgb = XGBClassifier(eval_metric="logloss", random_state=42)

# Train models
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Save models
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(gb, "models/gradient_boost.pkl")
joblib.dump(xgb, "models/xgboost.pkl")

print("✅ All models trained and saved successfully!")
