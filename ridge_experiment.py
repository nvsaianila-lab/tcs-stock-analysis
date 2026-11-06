
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Input and output files
data_path = Path("data/tcs_stock_data.csv")
linear_path = Path("outputs/tcs_test_predictions.csv")
ridge_path = Path("outputs/tcs_ridge_test_predictions.csv")

# Never overwrite existing Ridge results
if ridge_path.exists():
    raise FileExistsError(
        f"{ridge_path} already exists. "
        "Rename it or choose another output filename first."
    )

# Check required input files
if not data_path.exists() or not linear_path.exists():
    raise FileNotFoundError(
        "Required input file is missing. Check the data and outputs folders."
    )

# Load and prepare historical data
df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Create the same features as prediction.py
for lag in [1, 2, 3, 5, 10]:
    df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)

df["MA_5"] = df["Close"].rolling(window=5).mean()
df["MA_10"] = df["Close"].rolling(window=10).mean()
df["Return_1D"] = df["Close"].pct_change()

# Predict the next trading day's closing price
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

# Match the original Linear Regression preparation
model_data = df.dropna(
    subset=features + ["Target_Next_Close", "Target_Next_Date"]
).copy()

X = model_data[features]
y = model_data["Target_Next_Close"]

# Same chronological 80/20 split
split_index = int(len(model_data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Scale features and train Ridge Regression
model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate Ridge
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Baseline: tomorrow's price equals today's close
baseline_predictions = model_data["Close"].iloc[
    split_index:
].to_numpy()

baseline_mae = mean_absolute_error(y_test, baseline_predictions)

# Load existing Linear Regression results for validation
linear_results = pd.read_csv(linear_path)
linear_results["Date"] = pd.to_datetime(linear_results["Date"])

expected_dates = pd.to_datetime(
    model_data["Target_Next_Date"].iloc[split_index:]
).reset_index(drop=True)

saved_dates = linear_results["Date"].reset_index(drop=True)

dates_match = (
    len(linear_results) == len(expected_dates)
    and saved_dates.equals(expected_dates)
)

print("TCS RIDGE REGRESSION EXPERIMENT")
print("-" * 40)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(f"Ridge MAE: Rs. {mae:.2f}")
print(f"Ridge RMSE: Rs. {rmse:.2f}")
print(f"Ridge R-squared: {r2:.4f}")
print(f"Baseline MAE: Rs. {baseline_mae:.2f}")
print("Linear prediction dates match:", dates_match)

if not dates_match:
    print(
        "\nWARNING: Existing Linear Regression test dates do not match. "
        "Do not compare the model scores until this is investigated."
    )
else:
    print("\nExisting Linear Regression results can be checked on the same dates.")

# Save to a new file only
results = pd.DataFrame({
    "Date": expected_dates,
    "Actual_Next_Close": y_test.reset_index(drop=True),
    "Ridge_Predicted_Next_Close": predictions
})

results.to_csv(ridge_path, index=False)

print("\nSaved new file:", ridge_path)
print("Existing project files were not modified.")
