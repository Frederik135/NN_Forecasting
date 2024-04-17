import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, dropout_prob):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    def forward(self, input_seq):
        rnn_out, _ = self.rnn(input_seq)
        return self.linear(rnn_out[:, -1])

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, dropout_prob):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        return self.linear(lstm_out[:, -1])

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, dropout_prob):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    def forward(self, input_seq):
        gru_out, _ = self.gru(input_seq)
        return self.linear(gru_out[:, -1])