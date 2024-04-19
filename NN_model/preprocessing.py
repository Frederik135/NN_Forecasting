import yfinance as yf
import pandas as pd
import numpy as np
from config import stock_ticker, start_date, end_date

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

# 1. Current open minus previous close [%]
curr_open_prev_close = [(stock_df['Open'].iloc[i] - stock_df['Close'].iloc[i-1]) * 100 / stock_df['Close'].iloc[i-1] for i in range(1,len(stock_df))]
curr_open_prev_close = [0] + curr_open_prev_close
curr_open_prev_close_adj = adjust_to_three_sigma(curr_open_prev_close)
curr_open_prev_close_adj_std = standardize(curr_open_prev_close_adj)
features_df['curr_open_prev_close_adj_std'] = curr_open_prev_close_adj_std

# 2. Today's close minus today's open [%] | (daily movement)
t_close_t_open = [(stock_df['Close'].iloc[i] - stock_df['Open'].iloc[i]) * 100 / stock_df['Open'].iloc[i] for i in range(len(stock_df))]
t_close_t_open_adj = adjust_to_three_sigma(t_close_t_open)
t_close_t_open_adj_std = standardize(t_close_t_open_adj)
features_df['t_close_t_open_adj_std'] = t_close_t_open_adj_std

# 3. Today's high minus today's low [%] | (daily volatility)
t_high_t_low = [(stock_df['High'].iloc[i] - stock_df['Low'].iloc[i]) * 100 / stock_df['Low'].iloc[i] for i in range(len(stock_df))]
t_high_t_low_adj = adjust_to_three_sigma(t_high_t_low)
t_high_t_low_adj_std = standardize(t_high_t_low_adj)
features_df['t_high_t_low_adj_std'] = t_high_t_low_adj_std

# 4. Bid-Ask Spread [%]

# 5. Volume
volume = stock_df['Volume']
volume_adjusted = adjust_to_three_sigma(volume)
log_volume = [np.log(value) for value in stock_df['Volume']]
volume_log_std = standardize(log_volume)
features_df['volume_log_std'] = volume_log_std

# 6. Month of year
def month_of_year(month):
    radians = (month - 1) * (np.pi / 6)
    return [np.cos(radians), np.sin(radians)]
features_df['month_of_year'] = features_df.index.month.map(month_of_year)

# 7. Week of year
def week_of_year(week):
    radians = (week - 1) * (2 * np.pi / 52)
    return [np.cos(radians), np.sin(radians)]
features_df['week_of_year'] = features_df.index.isocalendar().week.map(week_of_year)

# 8. Day of year - leap years are removed from the data so we always divide by 365
def day_of_year(day):
    radians = (day - 1) * (2 * np.pi / 365)
    return [np.cos(radians), np.sin(radians)]
features_df['day_of_year'] = features_df.index.map(lambda x: day_of_year(x.dayofyear))

# 9. Day of month
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

# 10. Day of week
def day_of_week(day):
    radians = (day - 1) * (2 * np.pi / 7)
    return [np.cos(radians), np.sin(radians)]
features_df['day_of_week'] = features_df.index.isocalendar().day.map(day_of_week)

# 11. Hour of day


# Split time encodings into separate columns for cos and sin
cyclic_features = ['month_of_year', 'week_of_year', 'day_of_year', 'day_of_month', 'day_of_week']
for feature in cyclic_features:
    features_df = flatten_cyclic_features(features_df, feature)