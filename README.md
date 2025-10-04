# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date:- 04-10-2025



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

data = pd.read_csv("/content/co2_gr_mlo.csv", comment="#")

result = adfuller(data['ann inc'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

x = int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]

lag_order = 13
model = AutoReg(train_data['ann inc'], lags=lag_order)
model_fit = model.fit()

plt.figure(figsize=(10, 6))
plot_acf(data['ann inc'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(data['ann inc'], lags=30, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)
mse = mean_squared_error(test_data['ann inc'], predictions)
print('Mean Squared Error (MSE):', mse)

plt.figure(figsize=(12, 6))
plt.plot(test_data['ann inc'], label='Test Data - Emissions')
plt.plot(predictions, label='Predictions - Emissions', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Amound of annual increase')
plt.title('AR Model Predictions vs Test Data - Emissions')
plt.legend()
plt.grid()
plt.show()
```
### OUTPUT:

GIVEN DATA
<img width="227" height="231" alt="image" src="https://github.com/user-attachments/assets/e01fed08-d3ad-40ae-b167-20c0bb72eca2" />

PACF - ACF
<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/9444402c-5d03-41fa-b7eb-2ec364e161f7" />

<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/0d583911-953b-4e17-8a46-557dfcc43699" />

PREDICTION
<img width="1010" height="547" alt="image" src="https://github.com/user-attachments/assets/fe49a76b-72b2-49e1-bef2-0c42a68a36de" />

### RESULT:
Thus we have successfully implemented the auto regression function using python.
