# TCS Stock Market Analysis and Prediction

## 1. Project Overview
This project analyzes historical Tata Consultancy Services (TCS) stock data from the National Stock Exchange of India (NSE). It explores historical price movements, daily returns, moving averages, and next-trading-day closing-price prediction.

This is an educational data analysis and machine learning project, not financial advice.

## 2. Objectives
- Collect historical TCS stock data.
- Clean and analyze stock-market data using Python and Pandas.
- Visualize closing prices, daily returns, and moving averages.
- Build a Linear Regression prediction model.
- Compare Linear Regression and Ridge Regression with a simple baseline.
- Evaluate prediction errors and explain the limitations.

## 3. Data Source
- Company: Tata Consultancy Services (TCS)
- NSE ticker: `TCS.NS`
- Source: Yahoo Finance through `yfinance`
- Frequency: Daily
- Requested historical period: Approximately five years

The exact dates and number of records depend on the downloaded dataset. The analysis uses historical data and does not guarantee future prices.

## 4. Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- yfinance
- PowerShell / Command Prompt

## 5. Project Structure

```text
TCS_Stock_Analysis/
├── data/
│   └── tcs_stock_data.csv
├── outputs/
│   ├── tcs_analysis_data.csv
│   ├── tcs_closing_price.png
│   ├── tcs_daily_returns.png
│   ├── tcs_moving_averages.png
│   ├── tcs_prediction_comparison.png
│   ├── tcs_test_predictions.csv
│   └── tcs_ridge_test_predictions.csv
├── collect_data.py
├── analysis.py
├── prediction.py
├── ridge_experiment.py
└── README.md
```

`README_updated.md` is the proposed replacement documentation. `README_backup_before_ridge.md` is a backup of the original README.

## 6. Methodology

### Step 1: Data Collection
Historical TCS stock data is collected using `yfinance` and saved to `data/tcs_stock_data.csv`.

### Step 2: Data Analysis
Pandas is used to inspect and process the data. Charts visualize closing prices, daily returns, and moving averages.

### Step 3: Feature Engineering
The prediction workflow uses previous closing prices, trading volume, lagged closing prices, moving averages, and a daily-return feature to predict the next trading day's closing price.

### Step 4: Chronological Train-Test Split
The data is split in time order, using 80% for training and 20% for testing. This avoids randomly mixing earlier and later dates.

### Step 5: Model Comparison
Three approaches are evaluated:
- **Baseline:** predicts the next closing price using the current closing price.
- **Linear Regression:** fits a linear model to the selected features.
- **Ridge Regression:** uses feature scaling and L2 regularization.

The Ridge experiment uses the same test dates as the existing Linear Regression predictions.

## 7. Model Results

Results from the current experiment:

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---:|---:|---:|
| Baseline | 33.07 | Not calculated | Not calculated |
| Linear Regression | 34.61 | 47.66 | 0.9834 |
| Ridge Regression | 35.29 | 49.11 | 0.9823 |

### Interpretation
- MAE measures the average absolute difference between predictions and actual prices. Lower is better.
- RMSE penalizes larger prediction errors. Lower is better.
- R² describes how much variation the model explains relative to a mean-based reference. Higher is generally better, but it should not be used alone.

The baseline has the lowest MAE in this experiment. Ridge Regression did not outperform Linear Regression on MAE or RMSE. The high R² values do not, by themselves, prove that the models can reliably forecast future stock prices.

## 8. Limitations
- Stock prices are affected by many factors not included in these features.
- Historical patterns may not continue in the future.
- Results depend on the chosen stock, data period, features, and test period.
- A model estimate is not a guaranteed future market price.
- This project is for learning and analysis, not investment advice.

## 9. Conclusion
This project demonstrates a basic stock-data workflow, including data collection, analysis, visualization, feature engineering, chronological model evaluation, and comparison with a baseline.

The current experiment shows that the more complex regression models did not beat the simple baseline on MAE. This is an important finding and suggests that future improvements should be tested using fair evaluation rather than assuming a more complex model is automatically better.

## 10. How to Run

Run these commands from the project folder after installing the dependencies used by the scripts:

```powershell
python collect_data.py
python analysis.py
python prediction.py
python ridge_experiment.py
```

Run the scripts in this order because the Ridge experiment compares its predictions with the saved Linear Regression results. Running the collection script may download fresh data, so retain a copy of important data before recollecting it.

## Disclaimer
This project is for educational purposes only. It does not provide financial advice or guarantee future stock prices.
