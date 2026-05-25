# Ex.No: 6               HOLT WINTERS METHOD




### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load Dataset
file_path = "DailyDelhiClimate.csv"

data = pd.read_csv(file_path)

# Convert date column
data['date'] = pd.to_datetime(data['date'])

# Set date as index
data = data.set_index('date')

# Select temperature column
series = data['meantemp']

# Remove missing values
series = series.dropna()

# Train-Test Split
train_size = int(0.9 * len(series))

train_data = series[:train_size]

test_data = series[train_size:]

# Holt-Winters Model
fitted_model = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='add',
    seasonal_periods=7
).fit()

# Predictions
test_predictions = fitted_model.forecast(len(test_data))

# Plot
plt.figure(figsize=(12, 8))

train_data.plot(label='Train Data')

test_data.plot(label='Test Data')

test_predictions.plot(label='Predicted Data')

plt.title('Delhi Climate Forecast using Holt-Winters')

plt.xlabel('Date')

plt.ylabel('Mean Temperature')

plt.legend()

plt.show()

# Error Metrics
mae = mean_absolute_error(test_data, test_predictions)

mse = mean_squared_error(test_data, test_predictions)

print(f"Mean Absolute Error = {mae:.4f}")

print(f"Mean Squared Error = {mse:.4f}")

# Final Model
final_model = ExponentialSmoothing(
    series,
    trend='add',
    seasonal='add',
    seasonal_periods=7
).fit()

# Forecast next 30 days
forecast_predictions = final_model.forecast(steps=30)

# Plot Forecast
plt.figure(figsize=(12, 8))

series.plot(label='Original Data')

forecast_predictions.plot(
    label='Forecasted Data',
    color='red'
)

plt.title('Future Forecast')

plt.xlabel('Date')

plt.ylabel('Mean Temperature')

plt.legend()

plt.show()

# Print Forecast
print("\nForecasted Values:")

print(forecast_predictions)

```
### OUTPUT:
<img width="996" height="717" alt="image" src="https://github.com/user-attachments/assets/e84e528b-454d-4bcb-810f-aa61a4bd0119" />


TEST_PREDICTION



FINAL_PREDICTION
<img width="996" height="717" alt="image" src="https://github.com/user-attachments/assets/b6d13005-de1c-486e-ab44-627af7bb9d5b" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
