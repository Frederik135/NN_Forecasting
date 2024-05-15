import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, dropout_prob):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    def forward(self, input_seq):
        rnn_out, _ = self.rnn(input_seq)
        return self.linear(rnn_out[:, -1])

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, dropout_prob):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        return self.linear(lstm_out[:, -1])

class GRU(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, dropout_prob):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    def forward(self, input_seq):
        gru_out, _ = self.gru(input_seq)
        return self.linear(gru_out[:, -1])

# Previous FCNN model
class FCNN(nn.Module):
    def __init__(self, seq_length, num_features, l_1, l_2, n_out):
        super(FCNN, self).__init__()
        self.seq_length = seq_length
        self.num_features = num_features
        self.lin1 = nn.Linear(seq_length * num_features, l_1)
        self.lin2 = nn.Linear(l_1, l_2)
        self.lin3 = nn.Linear(l_2, n_out)
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x1 = F.relu(self.lin1(x))
        x2 = F.relu(self.lin2(x1))
        y = self.lin3(x2)
        return y
    
# FCNN for hyperparameter tuning
class FCNN_model(nn.Module):
    def __init__(self, seq_length, num_features, hidden_layers, n_out, dropout_prob):
        super(FCNN_model, self).__init__()
        self.seq_length = seq_length
        self.num_features = num_features
        self.layers = nn.ModuleList()
        
        input_size = seq_length * num_features
        for layer_size in hidden_layers:
            self.layers.append(nn.Linear(input_size, layer_size))
            self.layers.append(nn.Dropout(dropout_prob))
            input_size = layer_size
        
        self.layers.append(nn.Linear(input_size, n_out))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x