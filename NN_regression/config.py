import torch
import torch.nn as nn
import yfinance as yf
from models import RNN, LSTM, GRU, FCNN

# Wichtigste Hyperparameter: number_units, num_layers, seq_length, learning_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stock_ticker = "KO"
ticker = yf.Ticker(stock_ticker)
company_name = ticker.info.get('longName', 'Company name not found')

price_features = [1]
time_features = []
num_features = len(price_features) + 2 * len(time_features)
num_units = 50
num_layers = 1
dropout_prob = 0
seq_length = 64                        # sequence length of sliding windows

model_config = {
    "input_size": num_features,         # Number of features (Currently: 4 price features and 5 time features (each with two columns))
    "hidden_layer_size": num_units,     # Number of neurons in hidden layer
    "num_layers": num_layers,           # Number of hidden layers
    "output_size": 1,                   # Number of output neurons
    "dropout_prob": dropout_prob        # Dropout probability (usually between 0.2 and 0.5; only apply when using >= 2 layers)
}
model = LSTM(**model_config).to(device)             # Select RNN model

"""
fcnn_config = {
    "seq_length": seq_length,                    # Number of time steps
    "num_features": num_features,                # Number of features
    "l_1": 64,                                   # Number of neurons in the first layer
    "l_2": 32,                                   # Number of neurons in the second layer
    "n_out": 1                                   # Output size
}
fcnn_model = FCNN(**model_config).to(device)
"""

architecture = str(model).split("(")[0]         # Selection of the RNN model
start_date = '1980-01-01'                       # Start date of the complete dataframe
end_date = '2024-01-01'                         # End date of the complete dataframe
num_epochs = 80                                 # Number of epochs
learning_rate = 0.0003                         # Learning rate of the optimizer

wandb_config = {
    "dataset": f"{company_name} closing prices",
    "architecture": architecture,
    "features": price_features + time_features,
    "num_units": num_units,
    "num_layers": num_layers,
    "dropout": dropout_prob,
    "seq_length": seq_length,
    "start_date": start_date,
    "end_date": end_date,
    "epochs": num_epochs,
    "learning_rate": learning_rate
}


