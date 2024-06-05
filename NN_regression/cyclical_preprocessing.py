import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import seq_length

def adjust_to_three_sigma(feature_list):
    percentile_1st = np.percentile(feature_list, 0.3)
    percentile_99th = np.percentile(feature_list, 99.7)
    feature_list_capped = [max(min(x, percentile_99th), percentile_1st) for x in feature_list]
    return feature_list_capped

def adjust_zeropointone_pct(feature_list):
    percentile_1st = np.percentile(feature_list, 0.1)
    percentile_99th = np.percentile(feature_list, 99.9)
    feature_list_capped = [max(min(x, percentile_99th), percentile_1st) for x in feature_list]
    return feature_list_capped

# Standardize with zero mean and 1 variance
def standardize(feature_list):
    feature_list = np.array(feature_list)
    mean = np.mean(feature_list)
    std_dev = np.std(feature_list)
    feature_list_standardized = (feature_list - mean) / std_dev
    return feature_list_standardized

# Number of data points
n_points = 10000

# Generate a time array
time = np.linspace(0, 10000, n_points)

# Generate a cyclical time-series using a sinusoidal function with an increased period
# Increase the period from 50 to 100 to space out the cycles
data = 30 * np.sin(2 * np.pi * time / 200) + 40  # Adjusted period

# Adding some noise
noise = np.random.normal(0, 1, n_points)  # Reduced standard deviation
data_with_noise = data + noise


stock_df = pd.DataFrame()
stock_df["close"] = data_with_noise

features_df = pd.DataFrame(index=stock_df.index)


labels = stock_df['close'].values
labels = np.log(labels[1:]/labels[:-1])
labels = np.insert(labels, 0, 0)


# 1 Log returns
log_returns = labels
log_returns_adj = adjust_to_three_sigma(log_returns)
log_returns_adj_std = standardize(log_returns_adj)
features_df["log_returns_adj_std"] = log_returns_adj_std



"""
# 2 Rolling standard deviation of log returns
# Choose a window size, e.g., 10 days
window_size = 10

# Calculate rolling standard deviation of log returns
df['historical_volatility'] = df['log_returns'].rolling(window=window_size).std()


# 3 EMA of log returns
# Set the span for EMA
span = 20  # This is equivalent to what is often used as the 'half-life' in financial contexts

# Calculate EMA of the absolute log returns as a measure of volatility
df['volatility_ema'] = df['log_returns'].abs().ewm(span=span, adjust=False).mean()

df.dropna(inplace=True)

"""



"""
# 1. Order Book Imbalance
df['order_book_imbalance'] = (df['bid_sum_volume'] - df['ask_sum_volume']) / (df['bid_sum_volume'] + df['ask_sum_volume'])
features_df['order_book_imbalance'] = standardize(df['order_book_imbalance'])


2. Volume Data Features

# Log of volume to normalize large values
df['log_volume'] = np.log(df['volume'] + 1)  # add 1 to avoid log(0)
features_df['log_volume_std'] = standardize(df['log_volume'])

3. Weighted Price and Volume Features

# 3.1 Weighted price average
df['weighted_price_std'] = standardize(df['weighted_price_average'])
features_df['weighted_price_std'] = df['weighted_price_std']

# 3.2 Paid volume as a percentage of total volume
df['paid_volume_pct'] = df['paid_volume'] / df['volume']
features_df['paid_volume_pct_std'] = standardize(df['paid_volume_pct'])

# 3.3 Give volume as percentage of total volume
df['give_volume_pct'] = df['give_volume'] / df['volume']
features_df['give_volume_pct_std'] = standardize(df['give_volume_pct'])


# 4. Best Bid and Ask Prices
# Spread between best bid and ask prices
df['bid_ask_spread'] = df['ask_best_price'] - df['bid_best_price']
features_df['bid_ask_spread_std'] = standardize(df['bid_ask_spread'])

# 5. Moving Averages and Relative Strength Index (RSI)
# Calculate simple moving average (SMA)
sma_window = 15
df['sma'] = df['close'].rolling(window=sma_window).mean()
# Standardize and add to features
features_df['sma_std'] = standardize(df['sma'])

# 6. Calculate Relative Strength Index (RSI)
def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = rsi(df['close'])
features_df['rsi_std'] = standardize(df['rsi'])

# 7 Calculate spread for 10MW and 30MW
df['spread_10MW'] = df['ask_best_10MW_price'] - df['bid_best_10MW_price']
df['spread_30MW'] = df['ask_best_30MW_price'] - df['bid_best_30MW_price']
# Standardize and add to features DataFrame
features_df['spread_10MW_std'] = standardize(df['spread_10MW'])
features_df['spread_30MW_std'] = standardize(df['spread_30MW'])

# 8 Relative Spreads at 10MW and 30MW
# Calculate average prices at 10MW and 30MW
df['avg_price_10MW'] = (df['ask_best_10MW_price'] + df['bid_best_10MW_price']) / 2
df['avg_price_30MW'] = (df['ask_best_30MW_price'] + df['bid_best_30MW_price']) / 2
# Calculate relative spreads
df['rel_spread_10MW'] = df['spread_10MW'] / df['avg_price_10MW']
df['rel_spread_30MW'] = df['spread_30MW'] / df['avg_price_30MW']
# Standardize and add to features DataFrame
features_df['rel_spread_10MW_std'] = standardize(df['rel_spread_10MW'])
features_df['rel_spread_30MW_std'] = standardize(df['rel_spread_30MW'])


# 9 Volume at 10MW and 30MW depth
# Checking and standardizing volumes at 10MW depth
if 'bid_best_10MW_volume' in df.columns and 'ask_best_10MW_volume' in df.columns:
    features_df['bid_vol_10MW_std'] = standardize(df['bid_best_10MW_volume'])
    features_df['ask_vol_10MW_std'] = standardize(df['ask_best_10MW_volume'])

# Checking and standardizing volumes at 30MW depth
if 'bid_best_30MW_volume' in df.columns and 'ask_best_30MW_volume' in df.columns:
    features_df['bid_vol_30MW_std'] = standardize(df['bid_best_30MW_volume'])
    features_df['ask_vol_30MW_std'] = standardize(df['ask_best_30MW_volume'])

# 10 Calculate the log of the ratio of paid_volume to given_volume
# Adding a small constant (epsilon) to avoid division by zero or log(0)
epsilon = 1e-9
df['log_paid_given_ratio'] = np.log((df['paid_volume'] + epsilon) / (df['given_volume'] + epsilon))
# Standardize the new feature before adding to features DataFrame
features_df['log_paid_given_ratio_std'] = standardize(df['log_paid_given_ratio'])
"""



"""
# Create correlation matrix to measure which features might be redundant. More of the same features could lead to overfitting

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming features_df is your DataFrame containing all the features
correlation_matrix = features_df.corr()

# Visualizing the correlation matrix using seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()

"""


"""
Use tree-based models like Random Forest to evaluate feature importance

from sklearn.ensemble import RandomForestRegressor

# Assume you have a target variable y
model = RandomForestRegressor()
model.fit(features_df, y)
importances = model.feature_importances_

# Plotting feature importance
plt.figure(figsize=(10, 8))
feature_names = features_df.columns
indices = np.argsort(importances)[::-1]
plt.title('Feature Importances by Random Forest')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
"""


"""
# Permutation Importance - we can use this with LSTMs and GRU models

from sklearn.inspection import permutation_importance

# Fit your LSTM/GRU model first and ensure it is capable of scoring (accuracy, R^2, etc.)
result = permutation_importance(model, features_df, y, n_repeats=10, random_state=42, scoring='accuracy')

# Plotting permutation importance
perm_sorted_idx = result.importances_mean.argsort()
plt.figure(figsize=(10, 8))
plt.boxplot(result.importances[perm_sorted_idx].T, vert=False, labels=features_df.columns[perm_sorted_idx])
plt.title("Permutation Importance (test set)")
plt.tight_layout()
plt.show()
"""




def create_sequences(data, labels):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = labels[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys).reshape(-1, 1)

total_length = len(features_df)
split_idx = int(total_length * 0.8) + seq_length
val_test_idx = int(total_length * 0.9) + seq_length

train_df = features_df.iloc[:split_idx]
val_df = features_df.iloc[split_idx:val_test_idx]
test_df = features_df.iloc[val_test_idx:]

train_labels = labels[:split_idx]
val_labels = labels[split_idx:val_test_idx]
test_labels = labels[val_test_idx:]

feature_scaler = {}
train_normalized = train_df.copy()
val_normalized = val_df.copy()
test_normalized = test_df.copy()

for column in train_df.columns:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_normalized[column] = scaler.fit_transform(train_df[column].values.reshape(-1, 1)).flatten()
    val_normalized[column] = scaler.transform(val_df[column].values.reshape(-1, 1)).flatten()
    test_normalized[column] = scaler.transform(test_df[column].values.reshape(-1, 1)).flatten()
    feature_scaler[column] = scaler

label_scaler = MinMaxScaler(feature_range=(-1, 1))
train_labels_scaled = label_scaler.fit_transform(train_labels.reshape(-1, 1))
val_labels_scaled = label_scaler.transform(val_labels.reshape(-1, 1))
test_labels_scaled = label_scaler.transform(test_labels.reshape(-1, 1))

X_train, y_train = create_sequences(train_normalized.values, train_labels_scaled.flatten())
X_val, y_val = create_sequences(val_normalized.values, val_labels_scaled.flatten())
X_test, y_test = create_sequences(test_normalized.values, test_labels_scaled.flatten())

# num_workers = 15 (home pc), max_workers = 10 (mac)
train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=64, shuffle=False, 
                                            num_workers=10, persistent_workers=True)
val_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), batch_size=64, shuffle=False, 
                                            num_workers=10, persistent_workers=True)
test_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), batch_size=64, shuffle=False, 
                                            num_workers=10, persistent_workers=True)