import torch
import torch.nn as nn
import torch.nn.functional as F
    
class LSTM_classification(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, dropout_prob, output_size):
        super(LSTM_classification, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        logits = self.linear(lstm_out[:, -1])
        return logits
    def predict(self, input_seq):
        logits = self.forward(input_seq)
        probabilities = F.softmax(logits, dim=1)
        return torch.max(probabilities, 1)[1]