
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

# Create the data folder if it does not exist
Path("data").mkdir(exist_ok=True)

# TCS stock listed on the National Stock Exchange of India
ticker = "TCS.NS"

print("Downloading historical TCS stock data...")

# Download five years of daily stock data
data = yf.download(
    ticker,
    period="5y",
    interval="1d",
    auto_adjust=True,
    progress=False
)

# Check whether data was received
if data.empty:
    raise RuntimeError(
        "No data received. Check your internet connection and try again."
    )

# Convert multi-level columns to simple column names if necessary
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Prepare and organise the data
data = data.reset_index()
data = data.drop_duplicates()
data = data.sort_values("Date")

# Save the original collected data
output_file = Path("data") / "tcs_stock_data.csv"
data.to_csv(output_file, index=False)

# Display a summary
print("\nData downloaded successfully!")
print("Number of rows:", len(data))
print("Columns:", list(data.columns))
print("\nFirst five records:")
print(data.head())
print("\nDate range:")
print(data["Date"].min(), "to", data["Date"].max())
print("\nMissing values:")
print(data.isnull().sum())
print("\nFile saved to:", output_file)
print("Collection date:", datetime.now().strftime("%Y-%m-%d"))
