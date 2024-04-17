import torch
import torch.nn as nn
import yfinance as yf
from models import RNNModel, LSTMModel, GRUModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stock_ticker = "KO"
ticker = yf.Ticker(stock_ticker)
company_name = ticker.info.get('longName', 'Company name not found')

num_units = 40
num_layers = 1
dropout_prob = 0

model_config = {
    "input_size": 14,                   # Number of features (Currently: 4 price features and 5 time features (each with two columns))
    "hidden_layer_size": num_units,     # Number of neurons in hidden layer
    "num_layers": num_layers,           # Number of hidden layers
    "output_size": 1,                   # Number of output neurons
    "dropout_prob": dropout_prob        # Dropout probability (usually between 0.2 and 0.5; only apply when using >= 2 layers)
}

model = RNNModel(**model_config).to(device)     # Select Deep Learning model

architecture = str(model).split("(")[0].replace("Model", "")       # Selection of the RNN model
seq_length = 13                 # Number of time steps
start_date = '1990-01-01'       # Start date of the complete dataframe
end_date = '2024-01-01'         # End date of the complete dataframe
num_epochs = 100                 # Number of epochs
learning_rate = 0.001           # Learning rate of the optimizer

wandb_config = {
    "dataset": f"{company_name} closing prices",
    "architecture": architecture,
    "num_units": num_units,
    "num_layers": num_layers,
    "dropout": dropout_prob,
    "seq_length": seq_length,
    "start_date": start_date,
    "end_date": end_date,
    "epochs": num_epochs,
    "learning_rate": learning_rate
}


