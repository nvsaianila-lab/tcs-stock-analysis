
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create output folder
Path("outputs").mkdir(exist_ok=True)

# Load historical stock data
df = pd.read_csv("data/tcs_stock_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Create features using historical information
for lag in [1, 2, 3, 5, 10]:
    df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)

df["MA_5"] = df["Close"].rolling(window=5).mean()
df["MA_10"] = df["Close"].rolling(window=10).mean()
df["Return_1D"] = df["Close"].pct_change()

# Target: next trading day's closing price
df["Target_Next_Close"] = df["Close"].shift(-1)
df["Target_Next_Date"] = df["Date"].shift(-1)

features = [
    "Close",
    "Volume",
    "Close_Lag_1",
    "Close_Lag_2",
    "Close_Lag_3",
    "Close_Lag_5",
    "Close_Lag_10",
    "MA_5",
    "MA_10",
    "Return_1D"
]

# Remove rows with missing features or target
model_data = df.dropna(
    subset=features + ["Target_Next_Close", "Target_Next_Date"]
).copy()

X = model_data[features]
y = model_data["Target_Next_Close"]

# Chronological split: first 80% train, last 20% test
split_index = int(len(model_data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test-period prices
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Baseline: tomorrow's price equals today's close
baseline_predictions = model_data["Close"].iloc[
    split_index:
].to_numpy()

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

print("TCS STOCK PRICE PREDICTION")
print("-" * 40)
print("Model: Linear Regression")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(f"Model MAE: Rs. {mae:.2f}")
print(f"Model RMSE: Rs. {rmse:.2f}")
print(f"Model R-squared: {r2:.4f}")
print(f"Baseline MAE: Rs. {baseline_mae:.2f}")

if mae < baseline_mae:
    print("Result: Model outperformed the baseline.")
else:
    print("Result: Model did not outperform the baseline.")

# Save test predictions with the correct target dates
results = pd.DataFrame({
    "Date": model_data["Target_Next_Date"].iloc[
        split_index:
    ].to_numpy(),
    "Actual_Next_Close": y_test.to_numpy(),
    "Predicted_Next_Close": predictions
})

results.to_csv(
    "outputs/tcs_test_predictions.csv",
    index=False
)

# Plot actual versus predicted prices
plt.figure(figsize=(12, 6))

plt.plot(
    results["Date"],
    results["Actual_Next_Close"],
    label="Actual Next-Day Close"
)

plt.plot(
    results["Date"],
    results["Predicted_Next_Close"],
    label="Predicted Next-Day Close"
)

plt.title("TCS: Actual vs Predicted Next-Day Closing Price")
plt.xlabel("Target Trading Date")
plt.ylabel("Adjusted Price (INR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/tcs_prediction_comparison.png",
    dpi=150
)
plt.close()

# Train on all available labelled historical records
# for the next-trading-day estimate
final_model = LinearRegression()
final_model.fit(X, y)

# Latest row with all required features
latest_features = df.dropna(
    subset=features
).iloc[[-1]][features]

next_estimate = final_model.predict(latest_features)[0]

print(
    "\nEstimated next-trading-day adjusted close: Rs.",
    round(next_estimate, 2)
)
print("This is a model estimate, not a guaranteed market price.")

print("\nSaved files:")
print("- outputs/tcs_test_predictions.csv")
print("- outputs/tcs_prediction_comparison.png")
