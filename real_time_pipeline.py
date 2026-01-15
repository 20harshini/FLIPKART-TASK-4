import joblib
import numpy as np
from route_optimization import optimize_route

model = joblib.load("models/XGBoost.pkl")

# Sample live input
new_delivery = np.array([[17.38, 78.48, 17.39, 78.50,
                           12, 1, 5, 8,
                           2, 1, 0, 14, 16]])

delay_prob = model.predict_proba(new_delivery)[0][1]
route, distance = optimize_route()

print("Delay Probability:", round(delay_prob * 100, 2), "%")
print("Optimized Route:", route)
print("Estimated Distance:", distance)
if delay_prob >= 0.60:
    print("⚠️ High Delay Risk! Suggested alternate optimized route.")
else:
    print("✅ Low delay risk. Normal route is fine.")

