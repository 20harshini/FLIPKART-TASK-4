import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# ================================
# 1. Load Dataset
# ================================
df = pd.read_excel("data/logistics_smart_500.xlsx")

print("\n✅ Delay Distribution:")
print(df["delay"].value_counts())
print(df["delay"].value_counts(normalize=True))

# ================================
# 2. Feature Engineering
# ================================
df["order_hour"] = pd.to_datetime(df["order_time"]).dt.hour
df["scheduled_hour"] = pd.to_datetime(df["scheduled_time"]).dt.hour

# ================================
# 3. Encode categorical columns
# ================================
label_cols = ["vehicle_type", "traffic_level", "weather", "road_type"]
le = LabelEncoder()

for col in label_cols:
    df[col] = le.fit_transform(df[col])

# ================================
# 4. Prepare X and y
# ================================
X = df.drop(["delivery_id", "order_time", "scheduled_time", "actual_time", "delay"], axis=1)
y = df["delay"]

# ================================
# 5. Scaling
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# 6. Stratified Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================================
# 7. Train Models (With imbalance handling)
# ================================
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
gb = GradientBoostingClassifier(random_state=42)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_weight = neg / pos

xgb = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=scale_weight
)

models = {
    "RandomForest": rf,
    "GradientBoost": gb,
    "XGBoost": xgb
}

# ================================
# 8. Train + Evaluate
# ================================
for name, model in models.items():
    print(f"\n================ {name} Results ================\n")
    model.fit(X_train, y_train)

    # probability prediction
    proba = model.predict_proba(X_test)[:, 1]

    # threshold tuning (important)
    threshold = 0.45
    preds = (proba >= threshold).astype(int)

    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))
    print("ROC AUC:", roc_auc_score(y_test, proba))

# ================================
# 9. Save Models
# ================================
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(gb, "models/gradient_boost.pkl")
joblib.dump(xgb, "models/xgboost.pkl")

print("\n✅ Models trained and saved successfully in models/ folder!")
