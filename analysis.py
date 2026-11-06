
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Create a folder for analysis results
Path("outputs").mkdir(exist_ok=True)

# Load the collected stock data
df = pd.read_csv("data/tcs_stock_data.csv")

# Prepare data
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Calculate daily percentage return
df["Daily_Return"] = df["Close"].pct_change() * 100

# Calculate moving averages
df["MA_20"] = df["Close"].rolling(window=20).mean()
df["MA_50"] = df["Close"].rolling(window=50).mean()

# Display basic statistics
print("TCS STOCK DATA ANALYSIS")
print("-" * 40)
print("Total trading records:", len(df))
print("Start date:", df["Date"].min().date())
print("End date:", df["Date"].max().date())
print("Missing values:\n", df.isnull().sum())

print("\nClosing Price Statistics:")
print(df["Close"].describe())

print("\nAverage daily return (%):", round(df["Daily_Return"].mean(), 4))
print("Highest closing price:", round(df["Close"].max(), 2))
print("Lowest closing price:", round(df["Close"].min(), 2))
print("Average trading volume:", round(df["Volume"].mean(), 0))

# Save analysis data
df.to_csv("outputs/tcs_analysis_data.csv", index=False)

# Graph 1: Historical closing price
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Close"], label="Closing Price")
plt.title("TCS Historical Closing Price")
plt.xlabel("Date")
plt.ylabel("Adjusted Price (INR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/tcs_closing_price.png", dpi=150)
plt.close()

# Graph 2: Closing price with moving averages
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Close"], label="Closing Price", alpha=0.7)
plt.plot(df["Date"], df["MA_20"], label="20-day Moving Average")
plt.plot(df["Date"], df["MA_50"], label="50-day Moving Average")
plt.title("TCS Closing Price and Moving Averages")
plt.xlabel("Date")
plt.ylabel("Adjusted Price (INR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/tcs_moving_averages.png", dpi=150)
plt.close()

# Graph 3: Daily returns
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Daily_Return"], linewidth=0.7)
plt.axhline(0, linewidth=1)
plt.title("TCS Daily Percentage Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/tcs_daily_returns.png", dpi=150)
plt.close()

print("\nAnalysis completed successfully!")
print("Analysis data and graphs are saved in the outputs folder.")
