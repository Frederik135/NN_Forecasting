import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt


# Download the past 5 years of daily Apple stock prices
aapl_df = yf.download('AAPL', start='2022-01-01', end='2024-01-01')

print(aapl_df)
# Normalize the prices
close_prices = aapl_df[['Close']].values
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(close_prices)

print(close_prices)

# Creating sequences for training
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10  # Number of days to look back
X, y = create_sequences(data_normalized, seq_length)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

# Step 3: Building and Training the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, seq_length):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_length, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3)  # compressing to 3-dimensional latent space
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, seq_length),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder(seq_length)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, X)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Step 4: Forecasting the Next Three Days
# Using the last sequence from the data as the input
last_sequence = data_normalized[-seq_length:]
last_sequence = torch.from_numpy(last_sequence).float()
model.eval()
with torch.no_grad():
    predicted_normalized = model(last_sequence.view(1, seq_length)).numpy()
predicted_prices = scaler.inverse_transform(predicted_normalized).flatten()

# Step 5: Plotting the forecast with the actual prices
plt.figure(figsize=(10,6))
predicted_days = np.arange(1, 4, 1)
plt.plot(predicted_days, predicted_prices[:3], label='Forecasted Price')
plt.legend()
plt.title('Apple Stock Price Forecast')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()