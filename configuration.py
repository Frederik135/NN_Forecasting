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
    "input_size": 1,                    # Number of features
    "hidden_layer_size": num_units,     # Number of neurons in hidden layer
    "num_layers": num_layers,           # Number of hidden layers
    "output_size": 1,                   # Number of output neurons
    "dropout_prob": dropout_prob        # Dropout probability (usually between 0.2 and 0.5; only apply when using >= 2 layers)
}

model = LSTMModel(**model_config).to(device)     # Select Deep Learning model

architecture = str(model).split("(")[0].replace("Model", " ")       # Selection of the RNN model
seq_length = 13                 # Number of time steps
start_date = '2009-01-01'       # Start date of the complete dataframe
end_date = '2024-01-01'         # End date of the complete dataframe
num_epochs = 150                # Number of epochs
learning_rate = 0.001           # Learning rate of the optimizer



