import torch
import yfinance as yf
import pandas as pd
import numpy as np
from config import stock_ticker, start_date, end_date
from sklearn.preprocessing import MinMaxScaler
from config import seq_length
from pmdarima import auto_arima

def adjust_to_three_sigma(feature_list):
    percentile_1st = np.percentile(feature_list, 0.3)
    percentile_99th = np.percentile(feature_list, 99.7)
    feature_list_capped = [max(min(x, percentile_99th), percentile_1st) for x in feature_list]
    return feature_list_capped

def adjust_top_one_pct(feature_list):
    percentile_1st = np.percentile(feature_list, 1)
    percentile_99th = np.percentile(feature_list, 99)
    feature_list_capped = [max(min(x, percentile_99th), percentile_1st) for x in feature_list]
    return feature_list_capped

# Standardize with zero mean and unit variance
def standardize(feature_list):
    feature_list = np.array(feature_list)
    mean = np.mean(feature_list)
    std_dev = np.std(feature_list)
    feature_list_standardized = (feature_list - mean) / std_dev
    return feature_list_standardized
    
def flatten_cyclic_features(df, feature_name):
    df[f'{feature_name}_cos'] = df[feature_name].apply(lambda x: x[0])
    df[f'{feature_name}_sin'] = df[feature_name].apply(lambda x: x[1])
    df.drop(feature_name, axis=1, inplace=True)
    return df

stock_df = yf.download(stock_ticker, start=start_date, end=end_date)

# Remove the 29th of February
stock_df = stock_df[~((stock_df.index.month == 2) & (stock_df.index.day == 29))]
features_df = pd.DataFrame(index=stock_df.index)

labels = stock_df['Close'].values
labels = [1.0] + [labels[i]/labels[i-1] for i in range(1, len(labels))]
labels = np.array(labels)

# 1. Relative change (relative change will is also used for the labels)
curr_close_prev_close_rel = [1.0] + [stock_df['Close'].iloc[i] / stock_df['Close'].iloc[i-1] for i in range(1,len(stock_df))]
curr_close_prev_close_rel_std = standardize(curr_close_prev_close_rel)
features_df['curr_close_prev_close_rel_std'] = curr_close_prev_close_rel_std
"""
# 2. Absolute change
curr_close_prev_close_abs = [0] + [stock_df['Close'].iloc[i] - stock_df['Close'].iloc[i-1] for i in range(1,len(stock_df))]
curr_close_prev_close_abs_std = standardize(curr_close_prev_close_abs)
features_df['curr_close_prev_close_abs_std'] = curr_close_prev_close_abs_std

# 3. Current open minus previous close [%]
curr_open_prev_close = [(stock_df['Open'].iloc[i] - stock_df['Close'].iloc[i-1]) / stock_df['Close'].iloc[i-1] for i in range(1,len(stock_df))]
curr_open_prev_close = [0] + curr_open_prev_close
curr_open_prev_close_adj = adjust_to_three_sigma(curr_open_prev_close)
curr_open_prev_close_adj_std = standardize(curr_open_prev_close_adj)
features_df['curr_open_prev_close_adj_std'] = curr_open_prev_close_adj_std

# 4. Today's close minus today's open [%] | (daily movement)
t_close_t_open = [(stock_df['Close'].iloc[i] - stock_df['Open'].iloc[i]) / stock_df['Open'].iloc[i] for i in range(len(stock_df))]
t_close_t_open_adj = adjust_to_three_sigma(t_close_t_open)
t_close_t_open_adj_std = standardize(t_close_t_open_adj)
features_df['t_close_t_open_adj_std'] = t_close_t_open_adj_std

# 5. Today's high minus today's low [%] | (daily volatility)
t_high_t_low = [(stock_df['High'].iloc[i] - stock_df['Low'].iloc[i]) / stock_df['Low'].iloc[i] for i in range(len(stock_df))]
t_high_t_low_adj = adjust_to_three_sigma(t_high_t_low)
t_high_t_low_adj_std = standardize(t_high_t_low_adj)
features_df['t_high_t_low_adj_std'] = t_high_t_low_adj_std

# 6. Bid-Ask Spread [%]

# 7. Volume
volume = stock_df['Volume']
volume_adjusted = adjust_to_three_sigma(volume)
log_volume = [np.log(value) for value in stock_df['Volume']]
volume_log_std = standardize(log_volume)
features_df['volume_log_std'] = volume_log_std

# 8. Month of year
def month_of_year(month):
    radians = (month - 1) * (np.pi / 6)
    return [np.cos(radians), np.sin(radians)]
features_df['month_of_year'] = features_df.index.month.map(month_of_year)

# 9. Week of year
def week_of_year(week):
    radians = (week - 1) * (2 * np.pi / 52)
    return [np.cos(radians), np.sin(radians)]
features_df['week_of_year'] = features_df.index.isocalendar().week.map(week_of_year)

# 10. Day of year - leap years are removed from the data so we always divide by 365
def day_of_year(day):
    radians = (day - 1) * (2 * np.pi / 365)
    return [np.cos(radians), np.sin(radians)]
features_df['day_of_year'] = features_df.index.map(lambda x: day_of_year(x.dayofyear))

# 11. Day of month
# For day_of_month we normalize by the number of days in the specific month, since there is no additional timespan from 
# the 28th of February to the 1st of March or from the 30th of September to the 1st of October
def day_of_month(row_index):
    day = row_index.day
    month = row_index.month
    long_months = [1, 3, 5, 7, 8, 10, 12]
    shorter_months = [2, 4, 6, 9, 11]
    if month in long_months:
        radians = (day - 1) * (2 * np.pi / 31)
    elif month in shorter_months:
        if month == 2:
            radians = (day - 1) * (2 * np.pi / 28)
        else:
            radians = (day - 1) * (2 * np.pi / 30)
    return [np.cos(radians), np.sin(radians)]
features_df['day_of_month'] = features_df.index.map(day_of_month)

# 12. Day of week
def day_of_week(day):
    radians = (day - 1) * (2 * np.pi / 7)
    return [np.cos(radians), np.sin(radians)]
features_df['day_of_week'] = features_df.index.isocalendar().day.map(day_of_week)

# 13. Hour of day


# Split time encodings into separate columns for cos and sin
cyclic_features = ['month_of_year', 'week_of_year', 'day_of_year', 'day_of_month', 'day_of_week']
for feature in cyclic_features:
    features_df = flatten_cyclic_features(features_df, feature)
"""


"""
# Classification Task: Create Labels
def calculate_moving_average(data, window=7):
    return data.rolling(window=window).mean()

def assign_labels(stock_df):
    # Calculate future moving average
    future_ma = calculate_moving_average(stock_df['Close'].shift(-7), window=7)
    
    # Calculate percentage change from current price to future moving average
    pct_change = ((future_ma - stock_df['Close']) / stock_df['Close']) * 100

    # Assign labels based on percentage change
    labels = np.zeros((len(stock_df), 3))  # Initialize a matrix of zeros with shape (n_samples, 3)
    labels[pct_change > 3, 0] = 1  # Buy
    labels[(pct_change <= 3) & (pct_change >= -3), 1] = 1  # Hold
    labels[pct_change < -3, 2] = 1  # Sell

    return labels

labels = assign_labels(stock_df)
"""


def create_sequences(data, labels):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = labels[i + seq_length - 1]
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

# p: autoregression (number of lag observations included in the model), q: moving average, d: degree of differencing; 
auto_arima_model = auto_arima(train_labels, start_p=0, start_q=0, max_p=15, max_q=15, max_d=5, 
                                trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)

feature_scaler = {}
train_normalized = train_df.copy()
val_normalized = val_df.copy()
test_normalized = test_df.copy()

# last closing price in the validation set (needed for plotting the predictions of the test set)
# initial_actual_close = stock_df['Close'].values[val_test_idx - 1]

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

test_dates = features_df.index[-(len(X_test) + seq_length):].tolist()
test_dates = test_dates[seq_length:]

# num_workers = 15 (home pc), max_workers = 10 (mac)
train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=64, shuffle=False, 
                                            num_workers=10, persistent_workers=True)
val_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), batch_size=64, shuffle=False, 
                                            num_workers=10, persistent_workers=True)
test_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), batch_size=64, shuffle=False, 
                                            num_workers=10, persistent_workers=True)