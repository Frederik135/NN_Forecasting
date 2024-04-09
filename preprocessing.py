import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from configuration import stock_ticker, start_date, end_date

stock_df = yf.download(stock_ticker, start=start_date, end=end_date)
features_df = pd.DataFrame(index=stock_df.index)

t_open_y_close = [(stock_df['Open'].iloc[i] - stock_df['Close'].iloc[i-1]) / stock_df['Close'].iloc[i-1] for i in range(1,len(stock_df))]
t_open_y_close = [0] + t_open_y_close
