import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('us_monthly_temperature_cleaned_1900_2023.csv')
data.dropna(subset=['Year', 'Month', 'Avg_Temperature_Celsius'], inplace=True)
data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))
data = data[['Date', 'Avg_Temperature_Celsius']]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Avg_Temperature_Celsius'].values.reshape(-1, 1))

# Function to create dataset with time_steps
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Create dataset
time_step = 12
X, y = create_dataset(scaled_data, time_step)

# Split the data into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).reshape(X_test.shape[0], X_test.shape[1], 1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Initialize the model, criterion, and optimizer
model = LSTMModel(input_size=1, hidden_layer_size=30, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 5
batch_size = 32
for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i + batch_size]
        batch_y = y_train_tensor[i:i + batch_size]
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluation and plotting results
model.eval()
with torch.no_grad():
    predicted_temp = model(X_test_tensor)

# Inverse transform the predictions and actual values
predicted_temp = scaler.inverse_transform(predicted_temp.numpy())
y_test_actual = scaler.inverse_transform(y_test_tensor.numpy().reshape(-1, 1))

# Calculate MSE
mse = mean_squared_error(y_test_actual, predicted_temp)
print('Mean Squared Error (MSE):', mse)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(data['Date'][-len(y_test_actual):].values, y_test_actual, label='Actual Temperature', color='blue')
plt.plot(data['Date'][-len(predicted_temp):].values, predicted_temp, label='Predicted Temperature', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Actual vs Predicted Temperature')
plt.legend()
plt.grid(True)
plt.show()

# Predicting the next month's temperature
last_12_months = scaled_data[-time_step:].reshape(1, -1)
last_12_months = last_12_months.reshape((1, time_step, 1))
future_pred = model(torch.tensor(last_12_months, dtype=torch.float32))
future_temp = scaler.inverse_transform(future_pred.detach().numpy())
print(f'Predicted Temperature for Next Month: {future_temp[0][0]:.2f} °C')
