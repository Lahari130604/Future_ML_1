import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load dataset
df = pd.read_csv("sales_data.csv")

# Create a date column
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-01-' + df['Day'].astype(str))
df.rename(columns={'Sum_of_Sales': 'y', 'Date': 'ds'}, inplace=True)

# Prophet forecasting
model = Prophet()
model.fit(df[['ds', 'y']])

future = model.make_future_dataframe(periods=6)  # forecast 6 more days
forecast = model.predict(future)

# Plot actual sales
plt.figure(figsize=(10,5))
plt.bar(df['ds'], df['y'], color='blue', label='Actual Sales')
plt.title("Sales by Day - January")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()

# Plot forecast
fig2 = model.plot(forecast)
plt.show()

# Show forecast for 16 Jan 2023
forecast_row = forecast[forecast['ds'] == '2023-01-16']
if not forecast_row.empty:
    print("Forecast:", round(forecast_row['yhat'].values[0]))
    print("Upper Bound:", round(forecast_row['yhat_upper'].values[0]))
    print("Lower Bound:", round(forecast_row['yhat_lower'].values[0]))