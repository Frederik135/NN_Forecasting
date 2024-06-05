import torch
import torch.nn as nn
import yfinance as yf
from models import TransformerModel

# Wichtigste Hyperparameter: number_units, num_layers, seq_length, learning_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

price_features = [1]
time_features = []
num_features = len(price_features) + 2 * len(time_features)
num_units = 52
num_layers = 1
num_heads = 4
dropout_prob = 0
seq_length = 64                       

# Transformer model

model_config = {
    "input_size": num_features,
    "hidden_size": num_units,
    "num_layers": num_layers,
    "num_heads": num_heads,
    "dropout": dropout_prob,
    "output_size": 1
}
model = TransformerModel(**model_config).to(device)


architecture = str(model).split("(")[0]         # Selection of the RNN model
num_epochs = 70                                 # Number of epochs
learning_rate = 0.0003                         # Learning rate of the optimizer

wandb_config = {
    "architecture": architecture,
    "features": price_features + time_features,
    "num_units": num_units,
    "num_layers": num_layers,
    "num_heads": num_heads,
    "dropout": dropout_prob,
    "seq_length": seq_length,
    "epochs": num_epochs,
    "learning_rate": learning_rate
}




