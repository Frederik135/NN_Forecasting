import torch
import yfinance as yf
import pandas as pd
import numpy as np
from config import stock_ticker, start_date, end_date
from sklearn.preprocessing import MinMaxScaler
from config import seq_length

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

stock_df.loc[:, 'rel_change'] = stock_df['Close'].pct_change().fillna(0)
labels = stock_df['rel_change'].apply(lambda x: 2 if x > 0.005 else (0 if x < -0.005 else 1)).values

# Features
features_df['rel_change'] = stock_df['Close'].pct_change().fillna(0)


def create_sequences(data, labels):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = labels[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

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

X_train, y_train = create_sequences(train_normalized.values, train_labels)
X_val, y_val = create_sequences(val_normalized.values, val_labels)
X_test, y_test = create_sequences(test_normalized.values, test_labels)

test_dates = features_df.index[-(len(X_test) + seq_length):].tolist()
test_dates = test_dates[seq_length:]

# num_workers = 15 (home pc), max_workers = 10 (mac)
train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=64, shuffle=True, 
                                           num_workers=10, persistent_workers=True)
val_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=64, shuffle=False, 
                                         num_workers=10, persistent_workers=True)
test_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=64, shuffle=False, 
                                          num_workers=10, persistent_workers=True)