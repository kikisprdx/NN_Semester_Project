import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size) # convert output to 9 class logits
        self.softmax = nn.Softmax() # convert logits to probabilities for each class

    def forward(self, inputs):
        out, _ = self.lstm(inputs)
        out = self.fc(out[-1, :]) # select output from last time step
        out = self.softmax(out)
        return out
