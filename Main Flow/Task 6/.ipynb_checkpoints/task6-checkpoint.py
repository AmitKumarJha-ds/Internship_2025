import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

import numpy as np

# Load dataset
df = pd.read_csv("Superstore.csv", parse_dates=["Order Date"], dayfirst=True)
df = df.rename(columns={"Order Date": "Date", "Sales": "Sales"})  # Ensure correct column names

# Aggregate sales by date
df = df.groupby("Date")["Sales"].sum().reset_index()

# Sort values by date
df = df.sort_values("Date")

# Visualizing Sales Trend
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Sales"], label="Sales", color="blue")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Trend Over Time")
plt.legend()
plt.show()

# Moving Average (7-day) Trend
plt.figure(figsize=(12, 6))
df["Sales"].rolling(window=7).mean().plot(label='7-day Moving Average', color='red')
plt.plot(df["Date"], df["Sales"], label="Sales", color="blue", alpha=0.5)
plt.legend()
plt.title("Sales with Moving Average")
plt.show()

# ARIMA Model Training
train_size = int(len(df) * 0.8)
train, test = df["Sales"][:train_size], df["Sales"][train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(5,1,0))  # (p,d,q) values can be tuned
model_fit = model.fit()

# Forecasting
forecast = model_fit.forecast(steps=len(test))

# Model Evaluation
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot Forecast vs Actual Sales
plt.figure(figsize=(12, 6))
plt.plot(df["Date"][train_size:], test, label="Actual Sales", color="blue")
plt.plot(df["Date"][train_size:], forecast, label="Forecasted Sales", color="red")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Actual vs Forecasted Sales")
plt.legend()
plt.show()

# Future Forecast for next 30 days
future_forecast = model_fit.forecast(steps=30)
future_dates = pd.date_range(start=df["Date"].max(), periods=31, freq='D')[1:]

# Create Forecasted Table
forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Sales": future_forecast})
print(forecast_df)
